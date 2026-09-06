import Foundation
import MLX

/// An immutable, evaluated field. Planes are vorticity, horizontal velocity,
/// and vertical velocity; rows run upward from the bottom of the periodic domain.
public struct SimulationSnapshot {
    public let field: MLXArray
    public let statistics: SimulationStatistics
    public let gridSize: Int
}

/// GPU pseudospectral incompressible Navier–Stokes on [0, 2π)².
///
/// Own and call this object on one serial worker queue. No grid-sized data is
/// copied to the CPU during evolution; only CFL and diagnostic scalars are read.
public final class MLXTurbulenceSolver {
    private var configuration: SimulationConfiguration
    private var size = 0
    private var spectrum = MLXArray(Float(0))
    private var waveSquared = MLXArray(Float(0))
    private var mask = MLXArray(Float(0))
    private var velocityX = MLXArray(Float(0))
    private var velocityY = MLXArray(Float(0))
    private var derivatives = MLXArray(Float(0))
    private var x = MLXArray(Float(0))
    private var y = MLXArray(Float(0))
    private var forcingA = MLXArray(Float(0))
    private var forcingB = MLXArray(Float(0))
    private var statistics = SimulationStatistics()
    private var lastSnapshot: SimulationSnapshot?
    private let period: Float = 2 * .pi

    public init(configuration: SimulationConfiguration, seed: UInt64 = 42) {
        self.configuration = configuration
        reset(configuration: configuration, seed: seed)
    }

    /// Initializes a prescribed physical vorticity field, useful for numerical
    /// experiments. Projection removes its mean and unresolved Fourier modes.
    public convenience init(configuration: SimulationConfiguration, vorticity: [Float]) {
        self.init(configuration: configuration)
        precondition(vorticity.count == size * size)
        spectrum = rfft2(MLXArray(vorticity, [size, size])) * mask
        spectrum.eval()
        lastSnapshot = nil
        _ = snapshot()
    }

    public func reset(configuration: SimulationConfiguration, seed: UInt64 = 42) {
        precondition(configuration.gridSize >= 16 && configuration.gridSize.isMultiple(of: 2),
                     "The Fourier grid must have an even size of at least 16.")
        self.configuration = configuration
        size = configuration.gridSize
        statistics = SimulationStatistics()
        lastSnapshot = nil
        let n = size
        let half = n / 2 + 1
        let kx = MLXArray((0..<half).map { Float($0) }, [1, half])
        let ky = MLXArray((0..<n).map { Float($0 < n / 2 ? $0 : $0 - n) }, [n, 1])
        waveSquared = kx * kx + ky * ky
        // Strict inequality is essential when N is divisible by three: keeping
        // the boundary modes would let quadratic products alias onto that edge.
        var retained = [Float](repeating: 0, count: n * half)
        for row in 0..<n {
            let waveY = row < n / 2 ? row : row - n
            for column in 0..<half where 3 * abs(waveY) < n && 3 * column < n {
                if column != 0 || waveY != 0 { retained[row * half + column] = 1 }
            }
        }
        mask = MLXArray(retained, [n, half])
        let imaginary = MLXArray(real: 0, imaginary: 1)
        let inverseLaplacian = mask / maximum(waveSquared, 1)
        // ω = ∂x v − ∂y u, u = ∂y ψ, v = −∂x ψ, −Δψ = ω.
        velocityX = imaginary * ky * inverseLaplacian
        velocityY = -imaginary * kx * inverseLaplacian
        let derivativeX = imaginary * (kx + zeros([n, 1]))
        let derivativeY = imaginary * (ky + zeros([1, half]))
        derivatives = stacked([velocityX, velocityY, derivativeX, derivativeY])
        x = MLXArray((0..<n).map { period * Float($0) / Float(n) }, [1, n])
        y = MLXArray((0..<n).map { period * Float($0) / Float(n) }, [n, 1])
        var random = SplitMix64(state: seed)
        let initial = initialVorticity(configuration.preset, random: &random)
        spectrum = rfft2(initial) * mask
        let phase = random.unit() * period
        let force1 = sin(9 * x + 2 * y + phase) + sin(3 * x - 10 * y - phase)
        let force2 = cos(2 * x + 9 * y - phase) + cos(10 * x - 3 * y + phase)
        forcingA = rfft2(force1) * mask
        forcingB = rfft2(force2) * mask
        eval(spectrum, waveSquared, derivatives, forcingA, forcingB)
        _ = snapshot()
    }

    public func advance(configuration: SimulationConfiguration,
                        impulses: [VortexImpulse] = [], steps: Int = 1) -> SimulationSnapshot {
        let started = ProcessInfo.processInfo.systemUptime
        if configuration.gridSize != size || configuration.preset != self.configuration.preset {
            reset(configuration: configuration)
        }
        self.configuration = configuration
        if !impulses.isEmpty {
            var brush = zeros([size, size])
            for impulse in impulses where impulse.x.isFinite && impulse.y.isFinite
                && impulse.radius.isFinite && impulse.strength.isFinite {
                let centerX = (impulse.x - floor(impulse.x)) * period
                let centerY = (impulse.y - floor(impulse.y)) * period
                let radius = Swift.max(1.5 * period / Float(size),
                                       Swift.min(0.3, Swift.max(0, impulse.radius)) * period)
                brush = brush + impulse.strength * gaussian(x: centerX, y: centerY, radius: radius)
            }
            spectrum = (spectrum + rfft2(brush)) * mask
            lastSnapshot = nil
        }
        let timeScale = configuration.timeScale.isFinite ? Swift.max(0, configuration.timeScale) : 1
        let viscosity = configuration.viscosity.isFinite ? Swift.max(0, configuration.viscosity) : 0.00015
        let force = configuration.forcing.isFinite ? configuration.forcing : 0
        for _ in 0..<Swift.max(0, steps) where timeScale > 0 {
            // A single batched inverse transform gives velocity and gradients.
            let physical = physicalDerivatives(spectrum)
            let courantSpeed = MLX.max(MLX.abs(physical[0]) + MLX.abs(physical[1])).item(Float.self)
            guard courantSpeed.isFinite else { break }
            let dt = Swift.min(0.0125 * timeScale,
                               0.45 * period / (Float(size) * Swift.max(courantSpeed, 0.05)))
            let third = exp((-viscosity * dt / 3) * waveSquared)
            let twoThirds = third * third
            let full = twoThirds * third
            let time = Float(statistics.time)
            // Heun's third-order RK applied to the integrating-factor variable.
            // All diffusion exponents are nonpositive, so viscosity imposes no
            // extra stability restriction, even at the largest grid size.
            let a = nonlinear(physical, time: time, forcing: force)
            let stage1 = third * (spectrum + (dt / 3) * a)
            let b = nonlinear(physicalDerivatives(stage1), time: time + dt / 3, forcing: force)
            let stage2 = twoThirds * spectrum + (2 * dt / 3) * third * b
            let c = nonlinear(physicalDerivatives(stage2), time: time + 2 * dt / 3, forcing: force)
            spectrum = (full * (spectrum + (dt / 4) * a) + (3 * dt / 4) * third * c) * mask
            // End the lazy graph every step, bounding memory over long runs.
            spectrum.eval()
            statistics.time += Double(dt)
            statistics.step += 1
            lastSnapshot = nil
        }
        let result = snapshot()
        statistics.solverMilliseconds = (ProcessInfo.processInfo.systemUptime - started) * 1000
        let timed = SimulationSnapshot(field: result.field, statistics: statistics, gridSize: size)
        lastSnapshot = timed
        return timed
    }

    public func snapshot() -> SimulationSnapshot {
        if let lastSnapshot { return lastSnapshot }
        let spectra = stacked([spectrum, spectrum * velocityX, spectrum * velocityY])
        let field = contiguous(irfft2(spectra, s: [size, size]).asType(.float32))
        let speedSquared = field[1] * field[1] + field[2] * field[2]
        let diagnostics = stacked([0.5 * mean(speedSquared), 0.5 * mean(field[0] * field[0]),
                                   sqrt(MLX.max(speedSquared)), MLX.max(MLX.abs(field[0]))])
        eval(field, diagnostics)
        let values = diagnostics.asArray(Float.self)
        statistics.energy = values[0]
        statistics.enstrophy = values[1]
        statistics.maxSpeed = values[2]
        statistics.maxVorticity = values[3]
        statistics.isFinite = values.allSatisfy(\.isFinite)
        let result = SimulationSnapshot(field: field, statistics: statistics, gridSize: size)
        lastSnapshot = result
        return result
    }

    private func physicalDerivatives(_ value: MLXArray) -> MLXArray {
        irfft2(derivatives * value, s: [size, size])
    }

    private func nonlinear(_ fields: MLXArray, time: Float, forcing: Float) -> MLXArray {
        let advection = -(fields[0] * fields[2] + fields[1] * fields[3])
        var result = rfft2(advection) * mask
        if forcing != 0 {
            result = result + forcing * (cos(time * 0.7) * forcingA + sin(time * 0.7) * forcingB)
        }
        return result
    }

    private func gaussian(x centerX: Float, y centerY: Float, radius: Float) -> MLXArray {
        let rawX = MLX.abs(x - centerX)
        let rawY = MLX.abs(y - centerY)
        let dx = minimum(rawX, period - rawX)
        let dy = minimum(rawY, period - rawY)
        return exp(-(dx * dx + dy * dy) / (2 * radius * radius))
    }

    private func initialVorticity(_ preset: FlowPreset, random: inout SplitMix64) -> MLXArray {
        var result = zeros([size, size])
        switch preset {
        case .vortexDance:
            for index in 0..<10 {
                let angle = Float(index) * period / 10
                let ring: Float = index.isMultiple(of: 2) ? 1.65 : 1.1
                let centerX = Float.pi + ring * cos(angle)
                let centerY = Float.pi + ring * sin(angle)
                let strength: Float = index.isMultiple(of: 2) ? 10 : -10
                result = result + strength * gaussian(x: centerX, y: centerY, radius: 0.27 + 0.1 * random.unit())
            }
        case .shearLayer:
            let band = exp(-8 * cos(y) * cos(y))
            result = 7 * sin(y) * band + 0.9 * sin(3 * x + 0.3 * sin(y)) * band
            result = result + 0.3 * sin(7 * x + y + random.unit() * period)
        case .cascade, .decaying:
            for index in 0..<32 {
                let waveX = Float(1 + Int(random.unit() * 10))
                let waveY = Float(Int(random.unit() * 19) - 9)
                let amplitude: Float = preset == .cascade ? 0.65 : 0.85
                result = result + amplitude * sin(waveX * x + waveY * y + random.unit() * period)
                // Seed a few large eddies to make the cascade visible immediately.
                if index < 3 && preset == .cascade {
                    result = result + 0.75 * cos(Float(index + 1) * x + y + random.unit() * period)
                }
            }
        }
        return result
    }
}

private struct SplitMix64 {
    var state: UInt64
    mutating func unit() -> Float {
        state &+= 0x9e3779b97f4a7c15
        var value = state
        value = (value ^ (value >> 30)) &* 0xbf58476d1ce4e5b9
        value = (value ^ (value >> 27)) &* 0x94d049bb133111eb
        value ^= value >> 31
        return Float(value >> 40) / Float(1 << 24)
    }
}
