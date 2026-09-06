# MLX Astra

A native macOS laboratory for two-dimensional turbulence, built for Apple silicon.
MLX evolves an incompressible fluid on the GPU; a custom Metal renderer reveals
its vorticity, eddies, and velocity field in an interactive SwiftUI workspace.

![MLX Astra running on an M1 Max](Documentation/astra.png)

## Run

1. Open **`MLXAstra.xcodeproj`** in Xcode.
2. Select the **MLXAstra** scheme and **My Mac**.
3. Run with **⌘R**. The shared scheme uses Release optimization for interactive speed.

Requires an Apple silicon Mac, macOS 14 or later, and Xcode with the Metal compiler
component installed. Built and tested with Xcode 26.3 on an M1 Max.
Xcode resolves the official [`mlx-swift`](https://github.com/ml-explore/mlx-swift)
package automatically. Its version is pinned to **0.30.6** for reproducible builds
with the installed Swift toolchain. First builds need GitHub access and take longer
while MLX and its Metal shaders compile. No Python runtime, server, model weights,
or additional application is required.

Command-line build and launch:

```sh
bash Scripts/build.sh
open /tmp/MLXAstraDerived/Build/Products/Release/MLXAstra.app
```

Scripts place build products in `/tmp/MLXAstraDerived` to keep iCloud Desktop
metadata out of signed app bundles. Set `ASTRA_DERIVED_DATA` to override that path.
Xcode's normal Derived Data location also works. Keep build products outside synced
Desktop/Documents folders if signing reports “resource fork, Finder information,
or similar detritus not allowed.”

## Explore

- **Initial conditions:** inverse cascade, vortex dance, shear instability, and decaying turbulence.
- **Drag** to add a positive vortex; **Option-drag or right-drag** reverses its spin.
- Adjust brush radius, viscosity, energy injection, time scale, and resolution live.
- Choose **128², 256², 384², 512², 1024², 2048², or 4096²** grids. A resolution or preset change restarts the flow.
- **Aurora, Ember, and Glacier** palettes show signed vorticity or speed. Exposure adjusts contrast.
- **Velocity direction** traces short streamlines through the actual instantaneous velocity field.
- Live kinetic energy, enstrophy, mean time per step, steps per wall-clock second, and energy history track the simulation.

| Action | Shortcut |
| --- | --- |
| Pause / resume | Space |
| Advance exactly one step | → |
| Reset the current flow | R |
| Focus mode / show controls | F |
| Save a PNG of the field | ⌘S |

A paused flow can still be edited with the vortex brush. PNG export renders the
current field and palette at 1800 × 1800, without interface controls or the brush
cursor. The Help menu explains the controls and numerical model.

## Physics and performance

The solver advances the dimensionless vorticity equation on a periodic
`[0, 2π) × [0, 2π)` domain:

```text
∂t ω + u ∂x ω + v ∂y ω = ν Δω + f
−Δψ = ω,     u = ∂y ψ,     v = −∂x ψ
```

Real FFTs recover velocity and vorticity gradients. A strict 2/3 spectral filter
removes aliased nonlinear modes and the zero mode. A third-order integrating-factor
Runge–Kutta method handles advection, with exact exponential treatment of viscous
diffusion. The adaptive CFL bound limits the timestep as velocity or resolution
increases. Time scale changes the desired timestep; it cannot exceed that bound.

Energy injection drives a time-varying small-scale Fourier pattern. Decaying
turbulence starts with injection disabled; setting injection to zero makes any
preset unforced. Vortex impulses wrap across both periodic boundaries and have
zero spatial mean. This is an exploratory two-dimensional periodic model, without
walls or a three-dimensional turbulence closure.

Simulation work runs on one serial background queue with at most one pending
batch. An adaptive batch of 1–32 CFL-checked integration steps targets an 8 ms GPU
budget between display refreshes. Paused single-step commands always advance once. Evaluated MLX arrays pass to Metal through shared-memory buffers without
copying the grid to a Swift array. Frames retain those allocations until rendering
completes. Rendering happens on demand with at most two GPU frames in flight;
pausing stops integration, and hiding or minimizing the window suspends updates.
Only scalar diagnostics are read back during normal evolution. PNG export performs
one explicit image readback. MLX's reusable allocation cache is capped at 128 MiB.

The display refreshes at up to 60 Hz while the solver advances multiple steps per
refresh. **Simulation · steps/s** counts actual integration steps divided by elapsed
wall-clock seconds, independently of display refresh. **Solver · ms** shows mean
time per integration step, amortizing each batch’s field snapshot. Larger grids,
strong brush impulses, and other GPU workloads can lower the achieved rate.

Measured on this M1 Max in Release (600 updates after 60 warmup steps, default
cascade; solver and GPU-buffer handoff only):

| Grid | Mean update | 95th percentile | Unpaced throughput |
| --- | --- | --- | --- |
| 256² | 1.61 ms | 2.03 ms | 621 updates/s |
| 512² | 2.71 ms | 4.36 ms | 370 updates/s |

A separate 3,000-update run at 384² stayed finite, averaging 2.10 ms with a
2.52 ms 95th percentile and 55.6 MiB peak active MLX allocation. The live 256²
window was visually checked at its 60 Hz display cadence; solver throughput is
reported separately as steps/s.

Large-grid smoke checks on the same M1 Max also remained finite:

| Grid | Mean step | Steps/s | Peak active MLX memory |
| --- | --- | --- | --- |
| 1024² | 19.91 ms | 50.2 | 391 MiB |
| 2048² | 92.86 ms | 10.8 | 1,561 MiB |
| 4096² | 403.44 ms | 2.48 | 4,642 MiB |

These shorter checks used 50/30/12 measured steps after 5/3/3 warmup steps,
respectively; the 4096² check also rendered a PNG from the full GPU field.
Larger grids naturally refresh less frequently when a single integration step
exceeds the display interval; controls and the simulation remain on separate threads.

These are local measurements, not guarantees or displayed frame rates.

## Verify

```sh
bash Scripts/test.sh
/tmp/MLXAstraDerived/Build/Products/Release/MLXAstra.app/Contents/MacOS/MLXAstra --live-check
bash Scripts/benchmark.sh --grid 256 --steps 300 --warmup 30
bash Scripts/benchmark.sh --grid 512 --steps 300 --warmup 30
bash Scripts/benchmark.sh --grid 4096 --steps 20 --warmup 3
bash Scripts/benchmark.sh --preset vortexDance --steps 240 --export /tmp/astra-vortices.png
```

The live check exercises batched throughput, pause, exactly one step, reset, and
brushing while paused through the actual application worker. It prints JSON and
closes its test instance.

The six numerical tests verify analytic Fourier-mode diffusion, velocity divergence
and curl, inviscid energy/enstrophy conservation with nontrivial advection, periodic
vortex injection, strict dealiasing on a grid divisible by three, and deterministic
finite evolution of every preset. GPU tests require access to Metal and should be
run through Xcode or `Scripts/test.sh`; plain `swift test` is not the build path for
this native Xcode app.

The benchmark executes the shipping solver and GPU-buffer handoff without UI
pacing. JSON output includes throughput, mean and 95th-percentile step time,
physical diagnostics, and MLX allocation statistics. It excludes first-run warmup
and does not measure the display compositor. `--export` also validates the shipping
Metal pipeline and writes a PNG.

## Source map

- `Sources/MLXAstraCore/`: simulation parameters, presets, and MLX solver.
- `Sources/MLXAstra/`: app lifecycle, background worker, SwiftUI interface, Metal rendering, and diagnostics.
- `Tests/MLXAstraTests/`: numerical regression tests.
- `Resources/`: app metadata and native vector-derived icon assets.
- `Scripts/`: build, test, benchmark, and asset/project generation.

`MLXAstra.xcodeproj` is checked in and needs no project generator to open. After
changing the file list, update and run `python3 Scripts/generate_project.py` to
regenerate it without external tooling.
