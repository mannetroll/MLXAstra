// CPU-only: compile alongside Sources/MLXAstraCore/SimulationTypes.swift.
import Foundation

@main
struct SimulationRunAverageSmoke {
    static func close(_ actual: Double, _ expected: Double, _ message: String) {
        precondition(abs(actual - expected) < 1e-12, "\(message): \(actual) != \(expected)")
    }

    static func main() {
        var average = SimulationRunAverage()
        close(average.simulationTimePerSecond, 0, "A new run has no rate")

        // Initialization lasted 100 seconds, followed by a two-second first batch.
        average.record(advancedSimulationTime: 1, startedAt: 100, completedAt: 102, segment: 0)
        close(average.simulationTimePerSecond, 0.5, "Include first batch; exclude initialization")

        // Include the one-second scheduling/display gap before the next batch.
        average.record(advancedSimulationTime: 3, startedAt: 103, completedAt: 105, segment: 0)
        close(average.simulationTimePerSecond, 0.8, "Divide total progress by total active time")

        // Pause is requested during this batch: draining work still belongs to the run.
        average.record(advancedSimulationTime: 1, startedAt: 106, completedAt: 108, segment: 0)
        close(average.simulationTimePerSecond, 0.625, "Include draining automatic work")

        // Idle and manual work occur between 108 and 200. Neither is recorded.
        average.record(advancedSimulationTime: 2, startedAt: 200, completedAt: 202, segment: 2)
        close(average.simulationTime, 7, "Exclude manual simulation advance")
        close(average.wallTime, 10, "Exclude idle and manual wall time")
        close(average.simulationTimePerSecond, 0.7, "Resume preserves earlier history")

        average.record(advancedSimulationTime: 3, startedAt: 203, completedAt: 204, segment: 2)
        close(average.simulationTimePerSecond, 10.0 / 12.0, "Resume continues its active segment")

        // Invalid measurements cannot poison a previously valid run average.
        average.record(advancedSimulationTime: .nan, startedAt: 204, completedAt: 205, segment: 2)
        average.record(advancedSimulationTime: 1, startedAt: 205, completedAt: 205, segment: 2)
        close(average.simulationTimePerSecond, 10.0 / 12.0, "Ignore invalid measurements")

        average = SimulationRunAverage()
        close(average.simulationTimePerSecond, 0, "Reset clears history")
        average.record(advancedSimulationTime: 1, startedAt: 500, completedAt: 504, segment: 3)
        close(average.simulationTimePerSecond, 0.25, "New run is independent")
        print("SimulationRunAverageSmoke passed")
    }
}
