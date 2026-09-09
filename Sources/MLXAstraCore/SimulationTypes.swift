import Foundation

public enum FlowPreset: String, CaseIterable, Identifiable, Sendable {
    case cascade, vortexDance, shearLayer, decaying
    public var id: Self { self }
    public var title: String {
        switch self {
        case .cascade: return "Inverse cascade"
        case .vortexDance: return "Vortex dance"
        case .shearLayer: return "Shear instability"
        case .decaying: return "Decaying turbulence"
        }
    }
    public var subtitle: String {
        switch self {
        case .cascade: return "Small eddies. Large structures."
        case .vortexDance: return "A constellation of interacting vortices."
        case .shearLayer: return "Watch a jet roll into billowing eddies."
        case .decaying: return "Let the flow find its own equilibrium."
        }
    }
    public var symbol: String {
        switch self {
        case .cascade: return "hurricane"
        case .vortexDance: return "circle.hexagongrid"
        case .shearLayer: return "water.waves"
        case .decaying: return "sparkles"
        }
    }
}

public enum ColorPalette: String, CaseIterable, Identifiable, Sendable {
    case aurora, ember, glacier
    public var id: Self { self }
    public var title: String { rawValue.capitalized }
    public var shaderIndex: UInt32 {
        switch self { case .aurora: return 0; case .ember: return 1; case .glacier: return 2 }
    }
}

public enum FieldDisplay: String, CaseIterable, Identifiable, Sendable {
    case vorticity, speed
    public var id: Self { self }
    public var title: String { rawValue.capitalized }
    public var shaderIndex: UInt32 { self == .vorticity ? 0 : 1 }
}

public struct SimulationConfiguration: Equatable, Sendable {
    public static let gridSizes = [128, 256, 384, 512, 1024, 2048, 4096]
    public var gridSize: Int = 512
    public var preset: FlowPreset = .cascade
    public var viscosity: Float = 0.00015
    public var forcing: Float = 0.8
    public var timeScale: Float = 1
    public init() {}

    /// Nonlinear products are evaluated here; state and display remain gridSize².
    public var paddedGridSize: Int { 3 * gridSize / 2 }

    /// Reuses freed GPU buffers; this does not limit live solver allocations.
    public var recommendedCacheLimit: Int {
        let mebibyte = 1024 * 1024
        let desired = max(128 * mebibyte, paddedGridSize * paddedGridSize * 512)
        let budget = min(4 * 1024 * mebibyte, Int(ProcessInfo.processInfo.physicalMemory / 8))
        return min(desired, budget)
    }
}

public struct VortexImpulse: Sendable {
    /// Unit-square coordinates, origin at the bottom left; radius as a fraction of the domain width.
    public let x: Float
    public let y: Float
    public let strength: Float
    public let radius: Float
    public init(x: Float, y: Float, strength: Float, radius: Float) {
        self.x = x; self.y = y; self.strength = strength; self.radius = radius
    }
}

public struct SimulationStatistics: Sendable {
    public var time: Double = 0
    public var step: Int = 0
    public var energy: Float = 0
    public var enstrophy: Float = 0
    public var maxSpeed: Float = 0
    public var maxVorticity: Float = 1
    public var solverMilliseconds: Double = 0
    public var isFinite: Bool = true
    public init() {}
}
