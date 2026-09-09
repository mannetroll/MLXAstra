import AppKit
import Combine
import Foundation
import Metal
import MLX
import MLXAstraCore
import UniformTypeIdentifiers

/// MLX arrays and their lazy graphs stay on this one serial worker.
private final class SimulationWorker: @unchecked Sendable {
    private var solver: MLXTurbulenceSolver?
    private var generation: UInt64 = .max
    private var sequence = 0
    private var batchSteps = 1
    private var initialTurnoverTime = 0.0
    private var estimatedMillisecondsPerStep = 2.0
    private let batchBudgetMilliseconds = 40.0
    private let device: any MTLDevice

    init(device: any MTLDevice) { self.device = device }

    func compute(configuration: SimulationConfiguration, generation: UInt64,
                 impulses: [VortexImpulse], advance: Bool,
                 continuous: Bool, stepsPerFrame: Int) -> (RenderFrame?, SimulationStatistics, Double) {
        if solver == nil || self.generation != generation {
            Memory.cacheLimit = configuration.recommendedCacheLimit
            if let solver { solver.reset(configuration: configuration) }
            else {
                solver = MLXTurbulenceSolver(configuration: configuration)
            }
            self.generation = generation
            batchSteps = 1
            estimatedMillisecondsPerStep = 2
            // reset() already evaluated and cached the initial snapshot. Capture
            // its turnover scale before the first step or any brush impulses.
            let initialEnstrophy = Double(solver!.snapshot().statistics.enstrophy)
            initialTurnoverTime = initialEnstrophy.isFinite && initialEnstrophy > 0
                ? 2 * .pi / sqrt(2 * initialEnstrophy) : 0
        }
        // Solver batches run independently of the display clock. Keep each batch
        // short enough to pick up input promptly, with one step minimum at large grids.
        // A paused Single Step always bypasses batching and advances exactly once.
        let limit = max(1, min(50, stepsPerFrame))
        let count = advance ? (continuous ? min(batchSteps, limit) : 1) : 0
        let snapshot = solver!.advance(configuration: configuration, impulses: impulses, steps: count)
        if continuous && count > 0 && snapshot.statistics.isFinite {
            let duration = snapshot.statistics.solverMilliseconds
            let sample = max(duration / Double(count), 0.01)
            estimatedMillisecondsPerStep = 0.8 * estimatedMillisecondsPerStep + 0.2 * sample
            let desired = max(1, min(limit, Int(batchBudgetMilliseconds / estimatedMillisecondsPerStep)))
            if duration > batchBudgetMilliseconds * 1.5 {
                // React immediately to a slower grid or competing GPU workload.
                batchSteps = max(1, Int(Double(count) * batchBudgetMilliseconds / duration))
            } else {
                batchSteps = min(desired, count * 2)
            }
        }
        sequence += 1
        // Evaluated float32 planes are contiguous and page-sized at every offered resolution.
        // The frame owns the immutable array until the renderer's command buffer completes.
        let buffer = snapshot.field.asMTLBuffer(device: device, noCopy: true)
        let frame = buffer.map {
            RenderFrame(buffer: $0, gridSize: snapshot.gridSize, sequence: sequence,
                        maxVorticity: snapshot.statistics.maxVorticity,
                        maxSpeed: snapshot.statistics.maxSpeed, owner: snapshot.field)
        }
        return (frame, snapshot.statistics, initialTurnoverTime)
    }
}

@MainActor
final class SimulationModel: ObservableObject {
    @Published var config = SimulationConfiguration() {
        didSet {
            if config.gridSize != oldValue.gridSize || config.preset != oldValue.preset { reset() }
        }
    }
    @Published var palette: ColorPalette = .ember
    @Published var display: FieldDisplay = .vorticity
    @Published var isRunning = true
    @Published var showFlowLines = false
    @Published var stepsPerFrame = 10
    @Published var exposure: Double = 1.2
    @Published var brushRadius: Double = 0.035
    @Published var stats = SimulationStatistics()
    @Published var stepsPerSecond: Double = 0
    @Published var simulationTimePerSecond: Double = 0
    @Published var initialTurnoversPerSecond: Double = 0
    @Published var millisecondsPerStep: Double = 0
    @Published var history: [Double] = []
    @Published var statusMessage: String?
    @Published var isReady = false
    @Published var isFocusMode = false
    @Published var frame: RenderFrame?
    let deviceName: String
    private(set) var initialTurnoverTime = 0.0

    private let queue = DispatchQueue(label: "com.mannetroll.MLXAstra.simulation", qos: .userInitiated)
    private var worker: SimulationWorker?
    private var timer: Timer?
    private var busy = false
    private var active = false
    private var generation: UInt64 = 0
    private var needsFrame = true
    private var pendingSteps = 0
    private var impulses: [VortexImpulse] = []
    private var measurementStart = ProcessInfo.processInfo.systemUptime
    private var completedSteps = 0
    private var completedSimulationTime = 0.0
    private var completedSolverMilliseconds = 0.0
    private var lastDeliveredStep = 0
    private var lastDeliveredTime = 0.0
    private var lastPublication = 0.0
    private var latestStatistics = SimulationStatistics()
    private var pendingFrame: RenderFrame?
    private var lastFramePublication = 0.0
    private let frameInterval = 1.0 / 60.0

    init() {
        if let device = MTLCreateSystemDefaultDevice() {
            deviceName = device.name
            worker = SimulationWorker(device: device)
        } else {
            deviceName = "Metal unavailable"
            statusMessage = "A Metal-capable Apple silicon Mac is required to run this simulation."
            isRunning = false
        }
    }

    func start() {
        guard worker != nil else { return }
        active = true
        guard timer == nil else { return }
        resetThroughputMeasurement(clearDisplayedRates: false)
        let timer = Timer(timeInterval: frameInterval, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.publishPendingFrame()
                self?.tick()
            }
        }
        timer.tolerance = 0.001
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
        tick()
    }

    func stop() {
        active = false
        timer?.invalidate()
        timer = nil
        resetThroughputMeasurement(clearDisplayedRates: false)
    }

    func togglePause() {
        isRunning.toggle()
        resetThroughputMeasurement(clearDisplayedRates: false)
        if !isRunning {
            stats = latestStatistics
        } else {
            tick()
        }
    }

    func reset() {
        generation &+= 1
        needsFrame = true
        impulses.removeAll(keepingCapacity: true)
        history.removeAll(keepingCapacity: true)
        stats = SimulationStatistics()
        latestStatistics = stats
        lastDeliveredStep = 0
        lastDeliveredTime = 0
        initialTurnoverTime = 0
        millisecondsPerStep = 0
        lastPublication = 0
        resetThroughputMeasurement()
        statusMessage = nil
        isReady = false
        pendingSteps = 0
        pendingFrame = nil
        // Old generation results are discarded; no competing reset can touch MLX state.
        if active { tick() }
    }

    func applyPreset(_ preset: FlowPreset) {
        var next = config
        next.preset = preset
        next.forcing = preset == .decaying ? 0 : (preset == .cascade ? 0.8 : 0.25)
        next.viscosity = preset == .shearLayer ? 0.00008 : 0.00015
        if next.preset == config.preset {
            config = next
            reset()
        } else {
            config = next
        }
    }

    func stepOnce() {
        isRunning = false
        resetThroughputMeasurement(clearDisplayedRates: false)
        pendingSteps += 1
        tick()
    }

    private func resetThroughputMeasurement(clearDisplayedRates: Bool = true) {
        measurementStart = ProcessInfo.processInfo.systemUptime
        completedSteps = 0
        completedSimulationTime = 0
        completedSolverMilliseconds = 0
        if clearDisplayedRates {
            stepsPerSecond = 0
            simulationTimePerSecond = 0
            initialTurnoversPerSecond = 0
        }
    }

    func inject(x: Float, y: Float, negative: Bool) {
        guard x.isFinite, y.isFinite, isReady else { return }
        // Bound drag event accumulation if a GPU step is slower than the pointer event stream.
        if impulses.count >= 24 { impulses.removeFirst() }
        impulses.append(VortexImpulse(x: x, y: y, strength: negative ? -12 : 12,
                                      radius: Float(brushRadius)))
        needsFrame = true
        tick()
    }

    private func tick() {
        guard active, !busy, let worker,
              isRunning || needsFrame || pendingSteps > 0 || !impulses.isEmpty else { return }
        busy = true
        let currentGeneration = generation
        let currentConfig = config
        let currentImpulses = impulses
        let continuous = isRunning
        let currentStepsPerFrame = stepsPerFrame
        let advance = continuous || pendingSteps > 0
        impulses.removeAll(keepingCapacity: true)
        needsFrame = false
        if pendingSteps > 0 { pendingSteps -= 1 }
        queue.async { [weak self] in
            let result = autoreleasepool {
                worker.compute(configuration: currentConfig, generation: currentGeneration,
                               impulses: currentImpulses, advance: advance, continuous: continuous,
                               stepsPerFrame: currentStepsPerFrame)
            }
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.busy = false
                guard currentGeneration == self.generation else {
                    self.tick()
                    return
                }
                guard let frame = result.0 else {
                    self.statusMessage = "The GPU could not allocate a display buffer. Try a smaller grid."
                    self.isRunning = false
                    self.resetThroughputMeasurement(clearDisplayedRates: false)
                    return
                }
                guard result.1.isFinite else {
                    self.statusMessage = "The flow became unstable. Reset the field or increase viscosity."
                    self.isRunning = false
                    self.resetThroughputMeasurement(clearDisplayedRates: false)
                    return
                }
                self.pendingFrame = frame
                self.publishPendingFrame()
                self.latestStatistics = result.1
                self.initialTurnoverTime = result.2
                let now = ProcessInfo.processInfo.systemUptime
                let advancedSteps = max(0, result.1.step - self.lastDeliveredStep)
                let advancedTime = max(0, result.1.time - self.lastDeliveredTime)
                self.lastDeliveredStep = result.1.step
                self.lastDeliveredTime = result.1.time
                self.completedSteps += advancedSteps
                self.completedSimulationTime += advancedTime
                if advancedSteps > 0 {
                    self.completedSolverMilliseconds += result.1.solverMilliseconds
                }
                if now - self.lastPublication >= 0.25 || !self.isRunning || result.1.step == 0 {
                    self.stats = result.1
                    let interval = now - self.measurementStart
                    if interval >= 0.25 {
                        // A pause keeps the most recent live throughput visible;
                        // paused work and Single Step do not replace that sample.
                        if self.isRunning && self.active {
                            self.stepsPerSecond = Double(self.completedSteps) / interval
                            self.simulationTimePerSecond = self.completedSimulationTime / interval
                            self.initialTurnoversPerSecond = self.initialTurnoverTime > 0
                                ? self.simulationTimePerSecond / self.initialTurnoverTime : 0
                            if self.completedSteps > 0 {
                                self.millisecondsPerStep = self.completedSolverMilliseconds / Double(self.completedSteps)
                            }
                        }
                        self.measurementStart = now
                        self.completedSteps = 0
                        self.completedSimulationTime = 0
                        self.completedSolverMilliseconds = 0
                    }
                    self.history.append(Double(result.1.energy))
                    if self.history.count > 90 { self.history.removeFirst(self.history.count - 90) }
                    self.lastPublication = now
                }
                // One completion schedules one successor. The main queue remains free
                // for input while the serial worker runs; a slow batch never waits for
                // the next display tick before starting another integration step.
                self.tick()
            }
        }
    }

    private func publishPendingFrame() {
        guard let pendingFrame else { return }
        let now = ProcessInfo.processInfo.systemUptime
        guard now - lastFramePublication >= frameInterval else { return }
        frame = pendingFrame
        if !isReady { isReady = true }
        self.pendingFrame = nil
        lastFramePublication = now
    }

    func saveSnapshot() {
        guard let frame else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "Astra-\(config.preset.rawValue)-\(Int(stats.time)).png"
        panel.title = "Save flow snapshot"
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let palette = palette, display = display, exposure = exposure, flowLines = showFlowLines
        queue.async { [weak self] in
            do {
                let data = try TurbulenceSnapshot.pngData(frame: frame, palette: palette,
                    display: display, exposure: exposure, showFlowLines: flowLines)
                try data.write(to: url, options: .atomic)
            } catch {
                DispatchQueue.main.async { self?.statusMessage = "Could not save snapshot: \(error.localizedDescription)" }
            }
        }
    }
}
