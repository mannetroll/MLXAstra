import AppKit
import Foundation
import Metal
import MLX
import MLXAstraCore

/// Reproducible, headless validation using the exact solver and renderer shipped in the app.
/// Run the app binary with --benchmark or --export <path.png>.
enum AppDiagnostics {
    @MainActor private static var liveCheckStarted = false

    /// Exercises the shipping worker and main-thread controls without requiring
    /// Accessibility permission or changing the user's normal launch behavior.
    @MainActor
    static func beginLiveCheckIfRequested(model: SimulationModel) {
        guard CommandLine.arguments.contains("--live-check"), !liveCheckStarted else { return }
        liveCheckStarted = true
        model.config = diagnosticConfiguration()
        model.stepsPerFrame = batchSize(default: 10)
        Task { @MainActor in
            func waitUntil(_ condition: () -> Bool) async throws {
                let deadline = ProcessInfo.processInfo.systemUptime + 15
                while !condition() {
                    if ProcessInfo.processInfo.systemUptime > deadline {
                        throw NSError(domain: "MLXAstra.LiveCheck", code: 1,
                            userInfo: [NSLocalizedDescriptionKey: "Timed out waiting for the live solver"])
                    }
                    try await Task.sleep(for: .milliseconds(20))
                }
            }
            @MainActor func throughputIsZero() -> Bool {
                model.stepsPerSecond == 0 && model.simulationTimePerSecond == 0
                    && model.initialTurnoversPerSecond == 0
            }
            func nearlyEqual(_ actual: Double, _ expected: Double) -> Bool {
                actual.isFinite && expected.isFinite
                    && abs(actual - expected) <= 1e-12 * max(abs(expected), Double.leastNormalMagnitude)
            }
            do {
                try await waitUntil { model.isReady && model.stats.step > 0 }
                let firstStep = model.stats.step
                let firstSimulationTime = model.stats.time
                let initialTurnoverTime = model.initialTurnoverTime
                let began = ProcessInfo.processInfo.systemUptime
                try await Task.sleep(for: .seconds(3))
                let displayedRate = model.stepsPerSecond
                let displayedSimulationRate = model.simulationTimePerSecond
                let displayedTurnoverRate = model.initialTurnoversPerSecond
                @MainActor func retainsRunningThroughput() -> Bool {
                    model.stepsPerSecond == displayedRate
                        && model.simulationTimePerSecond == displayedSimulationRate
                        && model.initialTurnoversPerSecond == displayedTurnoverRate
                }
                let runningThroughputPositive = displayedRate.isFinite && displayedRate > 0
                    && displayedSimulationRate.isFinite && displayedSimulationRate > 0
                    && displayedTurnoverRate.isFinite && displayedTurnoverRate > 0
                let turnoverRelationMatches = initialTurnoverTime > 0
                    && nearlyEqual(displayedTurnoverRate, displayedSimulationRate / initialTurnoverTime)
                let turnoverFixedWhileAdvancing = model.initialTurnoverTime == initialTurnoverTime
                let milliseconds = model.millisecondsPerStep
                model.togglePause()
                let throughputRetainedOnPause = retainsRunningThroughput()
                try await Task.sleep(for: .milliseconds(150))
                let pausedStep = model.stats.step
                let pausedSimulationTime = model.stats.time
                let wallSeconds = ProcessInfo.processInfo.systemUptime - began
                try await Task.sleep(for: .milliseconds(150))
                let pauseStable = model.stats.step == pausedStep
                let throughputRetainedWhilePaused = retainsRunningThroughput()
                let turnoverFixedWhilePaused = model.initialTurnoverTime == initialTurnoverTime
                model.stepOnce()
                let throughputRetainedOnSingleStep = retainsRunningThroughput()
                try await waitUntil { model.stats.step > pausedStep }
                let singleStepDelta = model.stats.step - pausedStep
                let throughputRetainedAfterSingleStep = retainsRunningThroughput()
                let turnoverFixedAfterSingleStep = model.initialTurnoverTime == initialTurnoverTime
                model.reset()
                let throughputZeroOnReset = throughputIsZero()
                try await waitUntil { model.isReady && model.stats.step == 0 }
                let resetAtZero = model.stats.time == 0
                let throughputZeroAfterReset = throughputIsZero()
                let beforeBrush = model.stats.enstrophy
                let resetTurnoverTime = model.initialTurnoverTime
                let expectedInitialTurnoverTime = 2 * Double.pi / sqrt(2 * Double(beforeBrush))
                let turnoverMatchesInitialEnstrophy = beforeBrush > 0
                    && nearlyEqual(resetTurnoverTime, expectedInitialTurnoverTime)
                model.inject(x: 0.4, y: 0.6, negative: false)
                try await waitUntil { model.stats.enstrophy != beforeBrush }
                let brushWhilePaused = model.stats.step == 0
                let throughputZeroAfterBrush = throughputIsZero()
                let turnoverFixedAfterBrush = model.initialTurnoverTime == resetTurnoverTime
                let measuredSimulationTime = pausedSimulationTime - firstSimulationTime
                let observedSimulationRate = measuredSimulationTime / wallSeconds
                let observedTurnoverRate = observedSimulationRate / initialTurnoverTime
                let passed = displayedRate > 0 && pausedStep > firstStep && pauseStable
                    && singleStepDelta == 1 && resetAtZero && brushWhilePaused && model.stats.isFinite
                    && runningThroughputPositive && turnoverRelationMatches && turnoverFixedWhileAdvancing
                    && throughputRetainedOnPause && throughputRetainedWhilePaused && turnoverFixedWhilePaused
                    && throughputRetainedOnSingleStep && throughputRetainedAfterSingleStep && turnoverFixedAfterSingleStep
                    && throughputZeroOnReset && throughputZeroAfterReset && turnoverMatchesInitialEnstrophy
                    && throughputZeroAfterBrush && turnoverFixedAfterBrush
                    && observedSimulationRate.isFinite && observedSimulationRate > 0
                let report: [String: Any] = [
                    "grid": model.config.gridSize, "preset": model.config.preset.rawValue,
                    "show_every": model.stepsPerFrame,
                    "passed": passed, "displayed_steps_per_second": displayedRate,
                    "displayed_simulation_time_per_second": displayedSimulationRate,
                    "displayed_initial_turnovers_per_second": displayedTurnoverRate,
                    "observed_steps_per_second": Double(pausedStep - firstStep) / wallSeconds,
                    "observed_simulation_time_per_second": observedSimulationRate,
                    "observed_initial_turnovers_per_second": observedTurnoverRate,
                    "measured_simulation_time": measuredSimulationTime,
                    "observation_window": "First published running state to settled pause, including pause settling; separate from the displayed sampling window",
                    "initial_turnover_time": initialTurnoverTime,
                    "reset_initial_turnover_time": resetTurnoverTime,
                    "reset_initial_enstrophy": beforeBrush,
                    "expected_initial_turnover_time": expectedInitialTurnoverTime,
                    "running_throughput_positive": runningThroughputPositive,
                    "turnover_rate_relation_matches": turnoverRelationMatches,
                    "turnover_fixed_while_advancing": turnoverFixedWhileAdvancing,
                    "throughput_retained_on_pause": throughputRetainedOnPause,
                    "throughput_retained_while_paused": throughputRetainedWhilePaused,
                    "turnover_fixed_while_paused": turnoverFixedWhilePaused,
                    "throughput_retained_on_single_step": throughputRetainedOnSingleStep,
                    "throughput_retained_after_single_step": throughputRetainedAfterSingleStep,
                    "turnover_fixed_after_single_step": turnoverFixedAfterSingleStep,
                    "throughput_zero_on_reset": throughputZeroOnReset,
                    "throughput_zero_after_reset": throughputZeroAfterReset,
                    "turnover_matches_initial_enstrophy": turnoverMatchesInitialEnstrophy,
                    "throughput_zero_after_brush": throughputZeroAfterBrush,
                    "turnover_fixed_after_brush": turnoverFixedAfterBrush,
                    "integration_steps": pausedStep - firstStep, "wall_seconds": wallSeconds,
                    "milliseconds_per_step": milliseconds, "pause_stable": pauseStable,
                    "single_step_delta": singleStepDelta, "reset_at_zero": resetAtZero,
                    "brush_while_paused": brushWhilePaused
                ]
                let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                FileHandle.standardOutput.write(data)
                print("")
                model.stop()
                exit(passed ? 0 : 1)
            } catch {
                fputs("Live check failed: \(error.localizedDescription)\n", stderr)
                model.stop()
                exit(1)
            }
        }
    }

    static func runIfRequested() {
        let args = CommandLine.arguments
        guard args.contains("--benchmark") || args.contains("--export") else { return }
        guard let device = MTLCreateSystemDefaultDevice() else {
            fputs("Metal device unavailable\n", stderr)
            exit(1)
        }
        let configuration = diagnosticConfiguration()
        let steps = max(1, min(100_000, argument("--steps").flatMap(Int.init) ?? 240))
        let warmup = max(0, min(1_000, argument("--warmup").flatMap(Int.init) ?? 30))
        let cacheBytes: Int
        if let requestedCache = argument("--cache-mb") {
            guard let cacheMiB = Int(requestedCache), (0...8192).contains(cacheMiB) else {
                fputs("Cache must be 0...8192 MiB\n", stderr); exit(2)
            }
            cacheBytes = cacheMiB * 1024 * 1024
        } else {
            cacheBytes = configuration.recommendedCacheLimit
        }
        let batchSize = batchSize(default: 10)
        Memory.cacheLimit = cacheBytes
        runBenchmark(configuration: configuration, device: device, steps: steps,
                     warmup: warmup, cacheBytes: cacheBytes, batchSize: batchSize)
    }

    private static func argument(_ key: String) -> String? {
        let args = CommandLine.arguments
        guard let index = args.firstIndex(of: key), index + 1 < args.count else { return nil }
        return args[index + 1]
    }

    private static func batchSize(default defaultValue: Int) -> Int {
        guard let value = argument("--batch") else { return defaultValue }
        guard let count = Int(value), (1...50).contains(count) else {
            fputs("Batch must be 1...50 steps\n", stderr); exit(2)
        }
        return count
    }

    private static func diagnosticConfiguration() -> SimulationConfiguration {
        var configuration = SimulationConfiguration()
        if let grid = argument("--grid") {
            guard let n = Int(grid), SimulationConfiguration.gridSizes.contains(n) else {
                let sizes = SimulationConfiguration.gridSizes.map(String.init).joined(separator: ", ")
                fputs("Grid must be one of: \(sizes)\n", stderr); exit(2)
            }
            configuration.gridSize = n
        }
        if let preset = argument("--preset") {
            guard let value = FlowPreset(rawValue: preset) else {
                fputs("Unknown preset\n", stderr); exit(2)
            }
            configuration.preset = value
            if value == .decaying { configuration.forcing = 0 }
        }
        return configuration
    }

    private static func runBenchmark(configuration: SimulationConfiguration, device: any MTLDevice,
                                     steps: Int, warmup: Int, cacheBytes: Int, batchSize: Int) {
        let solver = MLXTurbulenceSolver(configuration: configuration)
        var snapshot = solver.snapshot()
        var warmedSteps = 0
        while warmedSteps < warmup {
            let count = min(batchSize, warmup - warmedSteps)
            snapshot = autoreleasepool { solver.advance(configuration: configuration, steps: count) }
            warmedSteps += count
        }
        var timings: [Double] = []
        timings.reserveCapacity(steps)
        var valid = true
        let firstStep = snapshot.statistics.step
        let firstSimulationTime = snapshot.statistics.time
        var measuredSteps = 0
        let started = ProcessInfo.processInfo.systemUptime
        while measuredSteps < steps {
            let count = min(batchSize, steps - measuredSteps)
            let t = ProcessInfo.processInfo.systemUptime
            autoreleasepool {
                snapshot = solver.advance(configuration: configuration, steps: count)
                // Include the same shared-memory wrapping used by the live view.
                let buffer = snapshot.field.asMTLBuffer(device: device, noCopy: true)
                valid = valid && buffer != nil && snapshot.statistics.isFinite
            }
            let amortizedMilliseconds = (ProcessInfo.processInfo.systemUptime - t) * 1000 / Double(count)
            timings.append(contentsOf: repeatElement(amortizedMilliseconds, count: count))
            measuredSteps = snapshot.statistics.step - firstStep
            if !valid || measuredSteps <= 0 {
                fputs("Non-finite state or GPU buffer failure at step \(measuredSteps)\n", stderr)
                exit(1)
            }
        }
        let elapsed = ProcessInfo.processInfo.systemUptime - started
        timings.sort()
        let p95 = timings[min(timings.count - 1, Int(Double(timings.count) * 0.95))]
        let stats = snapshot.statistics
        let report: [String: Any] = [
            "device": device.name, "grid": configuration.gridSize,
            "padded_grid": configuration.paddedGridSize, "dealiasing": "three-halves-padding",
            "preset": configuration.preset.rawValue, "steps": measuredSteps, "warmup": warmup,
            "batch_steps": batchSize, "cache_limit_mb": Double(cacheBytes) / 1048576,
            "elapsed_seconds": elapsed, "updates_per_second": Double(steps) / elapsed,
            "mean_ms": elapsed * 1000 / Double(steps), "p95_ms": p95,
            "simulation_time": stats.time, "energy": stats.energy, "enstrophy": stats.enstrophy,
            "measured_simulation_time": stats.time - firstSimulationTime,
            "simulation_time_per_second": (stats.time - firstSimulationTime) / elapsed,
            "max_speed": stats.maxSpeed, "finite": valid,
            "mlx_active_mb": Double(Memory.activeMemory) / 1048576,
            "mlx_cache_mb": Double(Memory.cacheMemory) / 1048576,
            "mlx_peak_mb": Double(Memory.peakMemory) / 1048576
        ]
        do {
            if let path = argument("--export") {
                guard let buffer = snapshot.field.asMTLBuffer(device: device, noCopy: true) else {
                    throw NSError(domain: "MLXAstra", code: 1,
                                  userInfo: [NSLocalizedDescriptionKey: "Could not wrap display buffer"])
                }
                let frame = RenderFrame(buffer: buffer, gridSize: snapshot.gridSize, sequence: stats.step,
                                        maxVorticity: stats.maxVorticity, maxSpeed: stats.maxSpeed, owner: snapshot.field)
                let png = try TurbulenceSnapshot.pngData(frame: frame, palette: .ember,
                    display: .vorticity, exposure: 1.2, showFlowLines: false, pixelSize: 1400)
                try png.write(to: URL(fileURLWithPath: path), options: .atomic)
            }
            let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            FileHandle.standardOutput.write(data)
            print("")
            exit(0)
        } catch {
            fputs("Diagnostics failed: \(error.localizedDescription)\n", stderr)
            exit(1)
        }
    }
}
