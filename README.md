# MLX Astra

A native macOS laboratory for interactive two-dimensional turbulence. The app
integrates the incompressible vorticity equation on the MLX Metal backend and
renders the result with a custom Metal shader.

## Highlights

- Periodic 2D vorticity/stream-function solver with nonlinear advection
- Dense simulation math executed by MLX on Apple silicon's unified GPU
- 128², 256², and 384² live grids at a target of 60 FPS
- Four forcing regimes, three GPU colour spectra, and live viscosity controls
- Click-drag interaction to inject positive or negative vortices
- Native SwiftUI shell with an efficient `MTKView` renderer

## Run in Xcode

Requirements: an Apple-silicon Mac, macOS 14+, Xcode 16+, and internet access
the first time Swift Package Manager fetches MLX.

1. Open `Package.swift` in Xcode.
2. Select the **MLXAstra** scheme and **My Mac** destination.
3. Press **Run** (`⌘R`).

Xcode resolves the official `ml-explore/mlx-swift` package automatically. MLX
uses its Metal backend by default on macOS.

## Command-line verification

```sh
swift test
swift build -c release
swift run -c release MLXAstra
```

Use a Release build when benchmarking. At higher grid sizes, each frame includes
18 warm-started Jacobi iterations; the 256² default is tuned for smooth interaction
on an M1 Max.

## Controls

- **Drag right / left** over the image to add positive / negative vorticity.
- **Pause** freezes integration while preserving the rendered field.
- **Reset** reseeds the active forcing preset.
- The sidebar adjusts viscosity, forcing, simulation time scale, grid size, and palette.

The domain is periodic: structures leaving one edge re-enter at the opposite edge.
