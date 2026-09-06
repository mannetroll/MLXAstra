#include <metal_stdlib>
using namespace metal;

struct AstraUniforms {
    uint gridSize;
    uint palette;
    uint display;
    uint showFlowLines;
    float exposure;
    float vorticityScale;
    float speedScale;
    float reserved;
    float2 resolution;
    float2 cursor;
    float brushRadius;
    float cursorVisible;
    float cursorNegative;
    float padding;
};

struct FieldVertex {
    float4 position [[position]];
    float2 uv;
};

vertex FieldVertex astraFieldVertex(uint id [[vertex_id]]) {
    float2 p = float2((id << 1) & 2, id & 2);
    FieldVertex out;
    out.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    out.uv = p;
    return out;
}

// Samples wrap across the periodic domain. The buffer is MLX's own evaluated
// allocation, with vorticity, horizontal velocity, and vertical velocity planes.
float samplePlane(device const float *field, float2 uv, uint n, uint plane) {
    float2 p = fract(uv) * float(n);
    uint2 a = uint2(floor(p));
    uint2 b = (a + 1) % n;
    float2 f = fract(p);
    uint offset = plane * n * n;
    float lo = mix(field[offset + a.y * n + a.x], field[offset + a.y * n + b.x], f.x);
    float hi = mix(field[offset + b.y * n + a.x], field[offset + b.y * n + b.x], f.x);
    return mix(lo, hi, f.y);
}

float2 sampleVelocity(device const float *field, float2 uv, uint n) {
    return float2(samplePlane(field, uv, n, 1), samplePlane(field, uv, n, 2));
}

float3 signedColor(float value, uint palette) {
    float amount = saturate(abs(value));
    bool positive = value >= 0.0;
    float3 base = float3(0.017, 0.026, 0.055);
    float3 middle, bright;
    if (palette == 1) {
        middle = positive ? float3(0.92, 0.19, 0.065) : float3(0.28, 0.17, 0.69);
        bright = positive ? float3(1.0, 0.88, 0.43) : float3(0.76, 0.64, 1.0);
    } else if (palette == 2) {
        middle = positive ? float3(0.01, 0.46, 0.78) : float3(0.14, 0.24, 0.68);
        bright = positive ? float3(0.70, 1.0, 0.99) : float3(0.70, 0.79, 1.0);
    } else {
        middle = positive ? float3(0.025, 0.60, 0.58) : float3(0.48, 0.20, 0.79);
        bright = positive ? float3(0.48, 1.0, 0.84) : float3(0.94, 0.67, 1.0);
    }
    float3 color = mix(base, middle, smoothstep(0.0, 0.60, amount));
    color = mix(color, bright, smoothstep(0.34, 1.0, amount));
    return color;
}

float3 speedColor(float value, uint palette) {
    float amount = saturate(value);
    float3 low = float3(0.019, 0.028, 0.063);
    float3 mid = palette == 1 ? float3(0.67, 0.10, 0.17)
        : palette == 2 ? float3(0.02, 0.33, 0.72) : float3(0.30, 0.17, 0.63);
    float3 high = palette == 1 ? float3(1.0, 0.86, 0.41)
        : palette == 2 ? float3(0.73, 0.97, 1.0) : float3(0.36, 1.0, 0.80);
    float3 color = mix(low, mid, smoothstep(0.0, 0.50, amount));
    return mix(color, high, smoothstep(0.32, 1.0, amount));
}

fragment float4 astraFieldFragment(FieldVertex in [[stage_in]],
                                    device const float *field [[buffer(0)]],
                                    constant AstraUniforms &u [[buffer(1)]]) {
    float3 color;
    if (u.display == 0) {
        float omega = samplePlane(field, in.uv, u.gridSize, 0);
        // Soft compression retains delicate filaments without clipping vortex cores.
        float value = tanh(omega / u.vorticityScale * max(u.exposure, 0.02) * 2.4);
        color = signedColor(value, u.palette);
    } else {
        float speed = length(sampleVelocity(field, in.uv, u.gridSize));
        float value = 1.0 - exp(-speed / u.speedScale * max(u.exposure, 0.02) * 2.1);
        color = speedColor(value, u.palette);
    }

    if (u.cursorVisible > 0.5) {
        float d = length(in.uv - u.cursor);
        float pixel = 1.0 / max(min(u.resolution.x, u.resolution.y), 1.0);
        float ring = 1.0 - smoothstep(0.65 * pixel, 1.65 * pixel, abs(d - u.brushRadius));
        float dot = 1.0 - smoothstep(pixel, 2.2 * pixel, d);
        float3 brushColor = u.cursorNegative > 0.5 ? float3(0.95, 0.77, 1.0) : float3(0.66, 1.0, 0.92);
        color = mix(color, brushColor, max(ring * 0.85, dot * 0.75));
    }
    return float4(color, 1.0);
}

struct FlowVertex {
    float4 position [[position]];
    float opacity;
};

float seedHash(float2 value) {
    return fract(sin(dot(value, float2(127.1, 311.7))) * 43758.5453);
}

// Each short streamline is traced through the actual instantaneous velocity,
// with midpoint integration. Geometry scales with the seed count, not pixels.
vertex FlowVertex astraFlowVertex(uint id [[vertex_id]],
                                  device const float *field [[buffer(0)]],
                                  constant AstraUniforms &u [[buffer(1)]]) {
    const uint segments = 18;
    uint seed = id / (segments * 2);
    uint endpoint = (id % (segments * 2)) / 2 + (id & 1);
    float2 cell = float2(seed % 32, seed / 32);
    float2 jitter = float2(seedHash(cell), seedHash(cell + 39.4));
    float2 origin = (cell + 0.25 + 0.5 * jitter) / 32.0;
    float2 p = origin;
    int steps = int(endpoint) - 9;
    float dt = (steps < 0 ? -1.0 : 1.0) * 0.010 / max(u.speedScale, 0.0001);
    for (int i = 0; i < abs(steps); ++i) {
        float2 v0 = sampleVelocity(field, p, u.gridSize);
        float2 midpoint = p + 0.5 * dt * v0;
        p += dt * sampleVelocity(field, midpoint, u.gridSize);
    }
    float speed = length(sampleVelocity(field, p, u.gridSize)) / max(u.speedScale, 0.0001);
    float taper = pow(max(sin(float(endpoint) / float(segments) * M_PI_F), 0.0), 0.7);
    FlowVertex out;
    // Keep geometry continuous at boundaries; the viewport clips excess length.
    out.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    out.opacity = taper * smoothstep(0.005, 0.15, speed) * 0.25;
    return out;
}

fragment float4 astraFlowFragment(FlowVertex in [[stage_in]]) {
    return float4(0.76, 0.93, 1.0, in.opacity);
}
