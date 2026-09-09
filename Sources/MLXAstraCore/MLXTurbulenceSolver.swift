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
/// copied to the CPU during evolution. CFL stays on the GPU; diagnostic scalars
/// are read once at the end of each requested batch.
public final class MLXTurbulenceSolver {
    private var configuration: SimulationConfiguration
    private var size = 0
    // Spectral arrays use [kx, ky]. MLX's forward 2D real FFT already
    // produces this contiguous storage, exposed through a zero-copy transpose.
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
    private var unforcedStep: (([MLXArray]) -> [MLXArray])!
    private var forcedStep: (([MLXArray]) -> [MLXArray])!
    private let period: Float = 2 * .pi

    public init(configuration: SimulationConfiguration, seed: UInt64 = 42) {
        self.configuration = configuration
        reset(configuration: configuration, seed: seed)
    }

    /// Initializes a prescribed physical vorticity field, useful for numerical
    /// experiments. Projection removes its mean and the two Nyquist lines.
    public convenience init(configuration: SimulationConfiguration, vorticity: [Float]) {
        self.init(configuration: configuration)
        precondition(vorticity.count == size * size)
        spectrum = rfft2(MLXArray(vorticity, [size, size])).transposed() * mask
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
        let kx = MLXArray((0..<half).map { Float($0) }, [half, 1])
        let ky = MLXArray((0..<n).map { Float($0 < n / 2 ? $0 : $0 - n) }, [1, n])
        waveSquared = kx * kx + ky * ky
        // The nonlinear products use a separate 3/2-padded grid, so retain the
        // full N-grid band. Drop both Nyquist lines to keep derivatives and
        // Hermitian symmetry unambiguous when transferring between grids.
        var retained = [Float](repeating: 0, count: n * half)
        for row in 0..<n {
            let waveY = row < n / 2 ? row : row - n
            for column in 0..<half where abs(waveY) < n / 2 && column < n / 2 {
                if column != 0 || waveY != 0 { retained[column * n + row] = 1 }
            }
        }
        mask = MLXArray(retained, [half, n])
        let imaginary = MLXArray(real: 0, imaginary: 1)
        let inverseLaplacian = mask / maximum(waveSquared, 1)
        // ω = ∂x v − ∂y u, u = ∂y ψ, v = −∂x ψ, −Δψ = ω.
        velocityX = imaginary * ky * inverseLaplacian
        velocityY = -imaginary * kx * inverseLaplacian
        let derivativeX = imaginary * (kx + zeros([1, n]))
        let derivativeY = imaginary * (ky + zeros([half, 1]))
        derivatives = stacked([velocityX, velocityY, derivativeX, derivativeY])
        x = MLXArray((0..<n).map { period * Float($0) / Float(n) }, [1, n])
        y = MLXArray((0..<n).map { period * Float($0) / Float(n) }, [n, 1])
        var random = SplitMix64(state: seed)
        let initial = initialVorticity(configuration.preset, random: &random)
        spectrum = rfft2(initial).transposed() * mask
        let phase = random.unit() * period
        let force1 = sin(9 * x + 2 * y + phase) + sin(3 * x - 10 * y - phase)
        let force2 = cos(2 * x + 9 * y - phase) + cos(10 * x - 3 * y + phase)
        forcingA = rfft2(force1).transposed() * mask
        forcingB = rfft2(force2).transposed() * mask
        eval(spectrum, waveSquared, derivatives, forcingA, forcingB)
        unforcedStep = makeCompiledStep(forced: false)
        forcedStep = makeCompiledStep(forced: true)
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
            spectrum = (spectrum + rfft2(brush).transposed()) * mask
            lastSnapshot = nil
        }
        let timeScale = configuration.timeScale.isFinite ? Swift.max(0, configuration.timeScale) : 1
        let viscosity = configuration.viscosity.isFinite ? Swift.max(0, configuration.viscosity) : 0.00015
        let force = configuration.forcing.isFinite ? configuration.forcing : 0
        let stepCount = timeScale > 0 ? Swift.max(0, steps) : 0
        // Changing controls are array inputs, so the compiled graph sees their
        // current values without retracing or retaining old configuration.
        let parameters = [MLXArray(Float(statistics.time)), MLXArray(viscosity),
                          MLXArray(0.0125 * timeScale), MLXArray(force)]
        var state = [spectrum, MLXArray(Float(0)), MLXArray(Int32(0))]
        let step = force == 0 ? unforcedStep! : forcedStep!
        for _ in 0..<stepCount {
            state = step(state + parameters)
            // Retire each step before encoding the next so large FFT buffers
            // can be reused instead of accumulating across the batch.
            eval(state)
        }
        if stepCount > 0 {
            spectrum = state[0]
            lastSnapshot = nil
        }
        let result = makeSnapshot(elapsed: state[1], completed: state[2])
        statistics.solverMilliseconds = (ProcessInfo.processInfo.systemUptime - started) * 1000
        let timed = SimulationSnapshot(field: result.field, statistics: statistics, gridSize: size)
        lastSnapshot = timed
        return timed
    }

    public func snapshot() -> SimulationSnapshot {
        makeSnapshot()
    }

    private func makeSnapshot(elapsed: MLXArray? = nil,
                              completed: MLXArray? = nil) -> SimulationSnapshot {
        if let lastSnapshot { return lastSnapshot }
        let spectra = stacked([spectrum, spectrum * velocityX, spectrum * velocityY])
        let field = contiguous(Self.inversePlanes(spectra, size: size))
        let speedSquared = field[1] * field[1] + field[2] * field[2]
        let diagnostics = stacked([0.5 * mean(speedSquared), 0.5 * mean(field[0] * field[0]),
                                   sqrt(MLX.max(speedSquared)), MLX.max(MLX.abs(field[0]))])
        var evaluated = [field, diagnostics]
        if let elapsed { evaluated.append(elapsed) }
        if let completed { evaluated.append(completed) }
        eval(evaluated)
        if let elapsed { statistics.time += Double(elapsed.item(Float.self)) }
        if let completed { statistics.step += Int(completed.item(Int32.self)) }
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

    /// Invert [plane, kx, ky] into [plane, y, x]. The y transform reads
    /// contiguous rows. Moving the plane axis before the x transform makes
    /// MLX's required transpose produce contiguous display planes directly.
    private static func inversePlanes(_ spectra: MLXArray, size: Int) -> MLXArray {
        let alongY = ifft(spectra, axis: 2)
        let alongX = irfft(alongY.transposed(2, 0, 1), n: size, axis: 2)
        return alongX.transposed(1, 0, 2)
    }

    /// Embed [plane, kx, ky] spectra on the larger transform grid. Negative ky
    /// modes belong at the end of the padded axis, not next to the positive ones.
    /// MLX normalizes inverse FFTs by the transform size, so coefficients must
    /// grow by M²/N² to preserve physical amplitudes after padding.
    private static func paddedPlanes(_ spectra: MLXArray, size n: Int, paddedSize m: Int) -> MLXArray {
        let positive = spectra[0..., 0..., 0..<(n / 2)]
        let negative = spectra[0..., 0..., (n / 2)..<n]
        let gap = zeros([spectra.dim(0), n / 2 + 1, m - n], dtype: spectra.dtype)
        let alongY = concatenated([positive, gap, negative], axis: 2)
        let expanded = padded(alongY, widths: [0, [0, m / 2 - n / 2], 0])
        return expanded * (Float(m * m) / Float(n * n))
    }

    /// Return the retained N-grid coefficients of an M-grid real transform.
    /// The reciprocal padding scale restores the state's N² FFT convention.
    private static func truncatedSpectrum(_ spectrum: MLXArray, size n: Int,
                                          paddedSize m: Int) -> MLXArray {
        let positive = spectrum[0..<(n / 2 + 1), 0..<(n / 2)]
        let negative = spectrum[0..<(n / 2 + 1), (m - n / 2)..<m]
        return concatenated([positive, negative], axis: 1) * (Float(n * n) / Float(m * m))
    }

    private func makeCompiledStep(forced: Bool) -> ([MLXArray]) -> [MLXArray] {
        // Capture only immutable operators. Reset creates new closures for the
        // new grid/seed; evolving state and controls are explicit inputs.
        let n = size
        let m = configuration.paddedGridSize
        let operators = derivatives
        let kSquared = waveSquared
        let projection = mask
        let forceA = forcingA
        let forceB = forcingB
        let courantNumerator = 0.45 * period / Float(m)
        return MLX.compile { (input: [MLXArray]) -> [MLXArray] in
            let omega = input[0]
            let elapsed = input[1]
            let completed = input[2]
            let time = input[3] + elapsed
            let viscosity = input[4]
            let maximumStep = input[5]
            let forcing = input[6]

            func physical(_ value: MLXArray) -> MLXArray {
                let padded = Self.paddedPlanes(operators * value, size: n, paddedSize: m)
                return Self.inversePlanes(padded, size: m)
            }
            func nonlinear(_ fields: MLXArray, at time: MLXArray) -> MLXArray {
                let advection = -(fields[0] * fields[2] + fields[1] * fields[3])
                let transformed = rfft2(advection).transposed()
                var result = Self.truncatedSpectrum(transformed, size: n, paddedSize: m) * projection
                if forced {
                    let phase = time * 0.7
                    result = result + forcing * (cos(phase) * forceA + sin(phase) * forceB)
                }
                return result
            }

            // Bound CFL using velocity maxima and spacing on the padded grid.
            // This stays on the GPU between the transform and RK stages.
            let fields = physical(omega)
            let speed = MLX.max(MLX.abs(fields[0]) + MLX.abs(fields[1]))
            let valid = isFinite(speed)
            let proposedStep = minimum(maximumStep, courantNumerator / maximum(speed, 0.05))
            let dt = which(valid, proposedStep, 0)
            let third = exp((-viscosity * dt / 3) * kSquared)
            let twoThirds = third * third
            let full = twoThirds * third

            // Heun's third-order RK on the integrating-factor variable. Every
            // diffusion exponent stays nonpositive, including at large grids.
            let a = nonlinear(fields, at: time)
            let stage1 = third * (omega + (dt / 3) * a)
            let b = nonlinear(physical(stage1), at: time + dt / 3)
            let stage2 = twoThirds * omega + (2 * dt / 3) * third * b
            let c = nonlinear(physical(stage2), at: time + 2 * dt / 3)
            let next = (full * (omega + (dt / 4) * a) + (3 * dt / 4) * third * c) * projection
            // An invalid current field freezes progress, matching the previous
            // CPU-side CFL guard while keeping valid runs asynchronous.
            return [which(valid, next, omega), elapsed + dt,
                    completed + valid.asType(.int32)]
        }
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
