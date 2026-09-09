import XCTest
import MLX
@testable import MLXAstraCore

final class MLXTurbulenceSolverTests: XCTestCase {
    private func configuration(size: Int = 64, viscosity: Float = 0) -> SimulationConfiguration {
        var configuration = SimulationConfiguration()
        configuration.gridSize = size
        configuration.preset = .decaying
        configuration.viscosity = viscosity
        configuration.forcing = 0
        return configuration
    }

    private func field(size: Int, _ value: (Float, Float) -> Float) -> [Float] {
        (0..<size * size).map { index in
            value(2 * .pi * Float(index % size) / Float(size),
                  2 * .pi * Float(index / size) / Float(size))
        }
    }

    func testSingleFourierModeHasExactViscousDecay() {
        let configuration = configuration(viscosity: 0.08)
        let initial = field(size: 64) { x, y in 2 * cos(2 * x + 3 * y) }
        let solver = MLXTurbulenceSolver(configuration: configuration, vorticity: initial)
        let result = solver.advance(configuration: configuration, steps: 40)
        let decay = Float(exp(-13 * Double(configuration.viscosity) * result.statistics.time))
        let expected = MLXArray(initial, [64, 64]) * decay
        let error = MLX.max(MLX.abs(result.field[0] - expected)).item(Float.self)
        XCTAssertLessThan(error, 0.0001)
        XCTAssertEqual(result.statistics.energy, decay * decay / 13, accuracy: 0.00002)
        XCTAssertEqual(result.statistics.enstrophy, decay * decay, accuracy: 0.0001)
        XCTAssertEqual(result.statistics.step, 40)
        XCTAssertTrue(result.statistics.isFinite)
    }

    func testRecoveredVelocityIsDivergenceFreeAndHasCorrectCurl() {
        let n = 48
        let solver = MLXTurbulenceSolver(configuration: configuration(size: n))
        let result = solver.snapshot()
        let kx = MLXArray((0..<(n / 2 + 1)).map { Float($0) }, [1, n / 2 + 1])
        let ky = MLXArray((0..<n).map { Float($0 < n / 2 ? $0 : $0 - n) }, [n, 1])
        let imaginary = MLXArray(real: 0, imaginary: 1)
        let u = rfft2(result.field[1])
        let v = rfft2(result.field[2])
        let divergence = irfft2(imaginary * (kx * u + ky * v), s: [n, n])
        let curl = irfft2(imaginary * (kx * v - ky * u), s: [n, n])
        // Both residuals have vorticity units. Normalize by the field norm so
        // the criterion does not change when the initial amplitude changes.
        // Float32 mixed-radix FFT roundoff is amplified by spectral derivatives;
        // 1e-5 relative accuracy also matches MLX's own FFT validation tolerance.
        let scale = MLX.max(MLX.abs(result.field[0])).item(Float.self)
        XCTAssertGreaterThan(scale, 1)
        let divergenceResidual = MLX.max(MLX.abs(divergence)).item(Float.self) / scale
        let curlResidual = MLX.max(MLX.abs(curl - result.field[0])).item(Float.self) / scale
        XCTAssertLessThan(divergenceResidual, 0.00001)
        XCTAssertLessThan(curlResidual, 0.00001)
        XCTAssertEqual(result.field.shape, [3, n, n])
        XCTAssertEqual(result.field.dtype, .float32)
    }

    func testInviscidUnforcedFlowPreservesEnergyAndEnstrophy() {
        let configuration = configuration()
        let solver = MLXTurbulenceSolver(configuration: configuration, seed: 17)
        let initial = solver.snapshot()
        let result = solver.advance(configuration: configuration, steps: 100)
        XCTAssertTrue(result.statistics.isFinite)
        XCTAssertLessThan(abs(result.statistics.energy / initial.statistics.energy - 1), 0.003)
        XCTAssertLessThan(abs(result.statistics.enstrophy / initial.statistics.enstrophy - 1), 0.003)
        XCTAssertLessThan(abs(mean(result.field[0]).item(Float.self)), 0.000001)
        // Conservation must accompany nontrivial advection, not a frozen field.
        XCTAssertGreaterThan(mean(MLX.abs(result.field[0] - initial.field[0])).item(Float.self), 0.1)
    }

    func testImpulseWrapsAtPeriodicBoundaryAndHasZeroMean() {
        let n = 64
        let configuration = configuration(size: n)
        let zero = [Float](repeating: 0, count: n * n)
        let first = MLXTurbulenceSolver(configuration: configuration, vorticity: zero)
        let second = MLXTurbulenceSolver(configuration: configuration, vorticity: zero)
        let a = first.advance(configuration: configuration,
                              impulses: [VortexImpulse(x: 0, y: 0.5, strength: 12, radius: 0.07)], steps: 0)
        let b = second.advance(configuration: configuration,
                               impulses: [VortexImpulse(x: 1, y: 0.5, strength: 12, radius: 0.07)], steps: 0)
        XCTAssertLessThan(MLX.max(MLX.abs(a.field - b.field)).item(Float.self), 0.000001)
        XCTAssertLessThan(abs(mean(a.field[0]).item(Float.self)), 0.000001)
        let samples = a.field[0].asArray(Float.self)
        XCTAssertGreaterThan(samples[(n / 2) * n], 10)
        XCTAssertGreaterThan(samples[(n / 2) * n + n - 1], 10)
        XCTAssertEqual(a.statistics.step, 0)
    }

    func testProjectionRetainsFullBandAndRemovesMeanAndNyquistAxes() {
        // N = 34 also exercises an odd padded transform size, M = 51.
        for n in [48, 34] {
            let half = n / 2
            let retained: [FourierMode: Double] = [
                FourierMode(x: half - 1, y: -3): 0.5,
                FourierMode(x: 1 - half, y: 3): 0.5,
                FourierMode(x: 2, y: half - 1): 0.375,
                FourierMode(x: -2, y: 1 - half): 0.375
            ]
            var input = retained
            input[FourierMode(x: 0, y: 0)] = 3
            input[FourierMode(x: half, y: 2)] = 0.25
            input[FourierMode(x: -half, y: -2)] = 0.25
            input[FourierMode(x: 1, y: half)] = 0.125
            input[FourierMode(x: -1, y: -half)] = 0.125
            let solver = MLXTurbulenceSolver(configuration: configuration(size: n),
                                             vorticity: cosineField(input, size: n))
            let result = solver.snapshot()
            let expected = MLXArray(cosineField(retained, size: n), [n, n])
            XCTAssertLessThan(MLX.max(MLX.abs(result.field[0] - expected))
                .item(Float.self), 0.00001, "N = \(n)")
            XCTAssertEqual(result.field.shape, [3, n, n])
        }
    }

    func testPaddedHighModeAdvectionMatchesIndependentFourierConvolution() {
        for n in [32, 34] {
            var configuration = configuration(size: n)
            configuration.timeScale = 0.8
            // p + q exceeds the retained x band and must not wrap into a low
            // mode. p - q = (3, 5) remains resolved and has nonzero advection.
            // Negative ky tests the placement of the lower half when padding;
            // unequal |p| and |q| prevent this from being a steady eigenflow.
            let p = FourierMode(x: n / 2 - 2, y: 3)
            let q = FourierMode(x: n / 2 - 5, y: -2)
            let initial: [FourierMode: Double] = [
                p: 0.5, FourierMode(x: -p.x, y: -p.y): 0.5,
                q: 0.375, FourierMode(x: -q.x, y: -q.y): 0.375
            ]
            let initialField = cosineField(initial, size: n)
            let solver = MLXTurbulenceSolver(configuration: configuration, vorticity: initialField)
            let result = solver.advance(configuration: configuration)
            XCTAssertEqual(result.statistics.time, 0.01, accuracy: 0.00000001)

            // This reference directly convolves normalized Fourier-series
            // coefficients in Double. It uses neither FFTs nor padding and
            // therefore independently checks transform scaling and dealiasing.
            let expectedSpectrum = fourierHeunStep(initial, size: n, dt: result.statistics.time)
            let expected = MLXArray(cosineField(expectedSpectrum, size: n), [n, n])
            XCTAssertLessThan(MLX.max(MLX.abs(result.field[0] - expected))
                .item(Float.self), 0.00001, "N = \(n)")
            XCTAssertGreaterThan(MLX.max(MLX.abs(expected - MLXArray(initialField, [n, n])))
                .item(Float.self), 0.0003, "The reference must exercise nonlinear evolution")
            XCTAssertTrue(result.statistics.isFinite)
            XCTAssertEqual(result.statistics.step, 1)
        }
    }

    func testPresetsResetDeterministicallyAndRemainFinite() {
        var configuration = configuration(size: 32, viscosity: 0.00015)
        let solver = MLXTurbulenceSolver(configuration: configuration)
        for preset in FlowPreset.allCases {
            configuration.preset = preset
            configuration.forcing = 0.8
            solver.reset(configuration: configuration, seed: 91)
            let first = solver.snapshot()
            let advanced = solver.advance(configuration: configuration, steps: 8)
            XCTAssertTrue(advanced.statistics.isFinite, "\(preset)")
            XCTAssertGreaterThan(advanced.statistics.energy, 0)
            XCTAssertGreaterThan(advanced.statistics.time, 0)
            solver.reset(configuration: configuration, seed: 91)
            let reset = solver.snapshot()
            XCTAssertEqual(reset.statistics.time, 0)
            XCTAssertEqual(reset.statistics.step, 0)
            XCTAssertEqual(MLX.max(MLX.abs(first.field - reset.field)).item(Float.self), 0)
        }
    }

    func testForcedBatchesMatchSingleStepsAndUseUpdatedControls() {
        var initialConfiguration = configuration(size: 32, viscosity: 0.00015)
        initialConfiguration.forcing = 0.8
        let batched = MLXTurbulenceSolver(configuration: initialConfiguration, seed: 91)
        let singles = MLXTurbulenceSolver(configuration: initialConfiguration, seed: 91)
        let oldViscosity = MLXTurbulenceSolver(configuration: initialConfiguration, seed: 91)
        let oldForcing = MLXTurbulenceSolver(configuration: initialConfiguration, seed: 91)

        let initialBatch = batched.advance(configuration: initialConfiguration, steps: 16)
        for _ in 0..<16 { _ = singles.advance(configuration: initialConfiguration) }
        _ = oldViscosity.advance(configuration: initialConfiguration, steps: 16)
        _ = oldForcing.advance(configuration: initialConfiguration, steps: 16)
        XCTAssertLessThan(MLX.max(MLX.abs(initialBatch.field - singles.snapshot().field))
            .item(Float.self), 0.0001)
        XCTAssertEqual(initialBatch.statistics.time, singles.snapshot().statistics.time,
                       accuracy: 0.000001)
        XCTAssertEqual(initialBatch.statistics.step, singles.snapshot().statistics.step)

        // Both nonzero forcing values use the same compiled path. Compare with
        // controls held independently fixed to catch stale scalar captures.
        var updated = initialConfiguration
        updated.viscosity = 0.08
        updated.forcing = 2.4
        updated.timeScale = 0.4
        let batch = batched.advance(configuration: updated, steps: 8)
        for _ in 0..<8 { _ = singles.advance(configuration: updated) }
        let single = singles.snapshot()
        XCTAssertTrue(batch.statistics.isFinite)
        XCTAssertLessThan(MLX.max(MLX.abs(batch.field - single.field)).item(Float.self), 0.0001)
        XCTAssertEqual(batch.statistics.time, single.statistics.time, accuracy: 0.000001)
        XCTAssertEqual(batch.statistics.step, 24)
        XCTAssertEqual(batch.statistics.step, single.statistics.step)

        var unchangedViscosity = updated
        unchangedViscosity.viscosity = initialConfiguration.viscosity
        let viscosityReference = oldViscosity.advance(configuration: unchangedViscosity, steps: 8)
        XCTAssertGreaterThan(MLX.max(MLX.abs(batch.field - viscosityReference.field))
            .item(Float.self), 0.001)
        var unchangedForcing = updated
        unchangedForcing.forcing = initialConfiguration.forcing
        let forcingReference = oldForcing.advance(configuration: unchangedForcing, steps: 8)
        XCTAssertGreaterThan(MLX.max(MLX.abs(batch.field - forcingReference.field))
            .item(Float.self), 0.001)
    }

    private struct FourierMode: Hashable {
        let x: Int
        let y: Int
    }

    // A centrosymmetric cosine field has real Fourier coefficients throughout
    // this inviscid evolution, so the independent reference needs no complex type.
    private func cosineField(_ spectrum: [FourierMode: Double], size: Int) -> [Float] {
        (0..<size * size).map { index in
            let x = 2 * Double.pi * Double(index % size) / Double(size)
            let y = 2 * Double.pi * Double(index / size) / Double(size)
            return Float(spectrum.reduce(0.0) { sum, entry in
                sum + entry.value * cos(Double(entry.key.x) * x + Double(entry.key.y) * y)
            })
        }
    }

    private func fourierAdvection(_ spectrum: [FourierMode: Double], size: Int) -> [FourierMode: Double] {
        var result: [FourierMode: Double] = [:]
        for (p, a) in spectrum {
            let pSquared = p.x * p.x + p.y * p.y
            guard pSquared > 0 else { continue }
            for (q, b) in spectrum {
                let k = FourierMode(x: p.x + q.x, y: p.y + q.y)
                guard abs(k.x) < size / 2, abs(k.y) < size / 2,
                      k.x != 0 || k.y != 0 else { continue }
                let cross = p.y * q.x - p.x * q.y
                guard cross != 0 else { continue }
                result[k, default: 0] += Double(cross) / Double(pSquared) * a * b
            }
        }
        return result
    }

    private func fourierSum(_ initial: [FourierMode: Double],
                            _ terms: [(Double, [FourierMode: Double])]) -> [FourierMode: Double] {
        var result = initial
        for (scale, spectrum) in terms {
            for (mode, amplitude) in spectrum {
                result[mode, default: 0] += scale * amplitude
            }
        }
        return result
    }

    private func fourierHeunStep(_ initial: [FourierMode: Double], size: Int,
                                dt: Double) -> [FourierMode: Double] {
        let a = fourierAdvection(initial, size: size)
        let stage1 = fourierSum(initial, [(dt / 3, a)])
        let b = fourierAdvection(stage1, size: size)
        let stage2 = fourierSum(initial, [(2 * dt / 3, b)])
        let c = fourierAdvection(stage2, size: size)
        return fourierSum(initial, [(dt / 4, a), (3 * dt / 4, c)])
    }
}
