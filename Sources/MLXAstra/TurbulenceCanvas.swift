import AppKit
import CoreGraphics
import MetalKit
import MLXAstraCore
import SwiftUI

/// The solver has evaluated a contiguous Float32 [3, N, N] array (omega, u, v).
/// Keeping its owner alive also keeps MLX's zero-copy Metal allocation alive.
final class RenderFrame: @unchecked Sendable {
    let buffer: any MTLBuffer
    let gridSize: Int
    let sequence: Int
    let maxVorticity: Float
    let maxSpeed: Float
    let owner: AnyObject

    init(buffer: any MTLBuffer, gridSize: Int, sequence: Int,
         maxVorticity: Float, maxSpeed: Float, owner: AnyObject) {
        self.buffer = buffer
        self.gridSize = gridSize
        self.sequence = sequence
        self.maxVorticity = maxVorticity
        self.maxSpeed = maxSpeed
        self.owner = owner
    }
}

private enum CanvasError: LocalizedError {
    case unavailable(String)
    var errorDescription: String? {
        switch self { case .unavailable(let message): return message }
    }
}

// Keep this layout in sync with AstraUniforms in Turbulence.metal (64 bytes).
private struct AstraUniforms {
    var gridSize: UInt32 = 1
    var palette: UInt32 = 0
    var display: UInt32 = 0
    var showFlowLines: UInt32 = 0
    var exposure: Float = 1
    var vorticityScale: Float = 1
    var speedScale: Float = 1
    var reserved: Float = 0
    var resolution = SIMD2<Float>(1, 1)
    var cursor = SIMD2<Float>(-1, -1)
    var brushRadius: Float = 0.035
    var cursorVisible: Float = 0
    var cursorNegative: Float = 0
    var padding: Float = 0

    init(frame: RenderFrame, palette: ColorPalette, display: FieldDisplay,
         exposure: Double, showFlowLines: Bool, size: CGSize) {
        gridSize = UInt32(frame.gridSize)
        self.palette = palette.shaderIndex
        self.display = display.shaderIndex
        self.exposure = Float(exposure)
        self.showFlowLines = showFlowLines ? 1 : 0
        vorticityScale = max(frame.maxVorticity, 0.0001)
        speedScale = max(frame.maxSpeed, 0.0001)
        resolution = SIMD2(Float(size.width), Float(size.height))
    }
}

private final class AstraPipeline {
    let device: any MTLDevice
    let queue: any MTLCommandQueue
    let field: any MTLRenderPipelineState
    let flow: any MTLRenderPipelineState
    static let pixelFormat: MTLPixelFormat = .bgra8Unorm

    init(device: any MTLDevice) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw CanvasError.unavailable("Metal could not create a rendering command queue.")
        }
        self.queue = queue
        guard let library = device.makeDefaultLibrary(),
              let fieldVertex = library.makeFunction(name: "astraFieldVertex"),
              let fieldFragment = library.makeFunction(name: "astraFieldFragment"),
              let flowVertex = library.makeFunction(name: "astraFlowVertex"),
              let flowFragment = library.makeFunction(name: "astraFlowFragment") else {
            throw CanvasError.unavailable("The turbulence shaders are missing. Rebuild MLXAstra in Xcode to compile Turbulence.metal.")
        }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "Astra · vorticity field"
        descriptor.vertexFunction = fieldVertex
        descriptor.fragmentFunction = fieldFragment
        descriptor.colorAttachments[0].pixelFormat = Self.pixelFormat
        field = try device.makeRenderPipelineState(descriptor: descriptor)

        descriptor.label = "Astra · velocity streamlines"
        descriptor.vertexFunction = flowVertex
        descriptor.fragmentFunction = flowFragment
        let color = descriptor.colorAttachments[0]!
        color.isBlendingEnabled = true
        color.sourceRGBBlendFactor = .sourceAlpha
        color.destinationRGBBlendFactor = .oneMinusSourceAlpha
        color.sourceAlphaBlendFactor = .one
        color.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        flow = try device.makeRenderPipelineState(descriptor: descriptor)
    }

    func encode(frame: RenderFrame, uniforms: inout AstraUniforms,
                descriptor: MTLRenderPassDescriptor, command: any MTLCommandBuffer) throws {
        guard frame.gridSize > 1, frame.gridSize <= 4096,
              frame.buffer.length >= 3 * frame.gridSize * frame.gridSize * MemoryLayout<Float>.stride else {
            throw CanvasError.unavailable("The solver supplied an invalid field buffer. Reset the simulation to try again.")
        }
        guard frame.buffer.device.registryID == device.registryID else {
            throw CanvasError.unavailable("The solver and renderer must use the same Metal device.")
        }
        guard let encoder = command.makeRenderCommandEncoder(descriptor: descriptor) else {
            throw CanvasError.unavailable("Metal could not begin rendering the turbulence field.")
        }
        encoder.label = "Astra · zero-copy field"
        encoder.setRenderPipelineState(field)
        encoder.setFragmentBuffer(frame.buffer, offset: 0, index: 0)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<AstraUniforms>.stride, index: 1)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        if uniforms.showFlowLines != 0 {
            encoder.setRenderPipelineState(flow)
            encoder.setVertexBuffer(frame.buffer, offset: 0, index: 0)
            encoder.setVertexBytes(&uniforms, length: MemoryLayout<AstraUniforms>.stride, index: 1)
            encoder.drawPrimitives(type: .line, vertexStart: 0, vertexCount: 32 * 32 * 18 * 2)
        }
        encoder.endEncoding()
    }
}

struct TurbulenceCanvas: NSViewRepresentable {
    var frame: RenderFrame?
    var palette: ColorPalette
    var display: FieldDisplay
    var exposure: Double
    var showFlowLines: Bool
    var brushRadius: Double
    var onInteraction: (Float, Float, Bool) -> Void
    var onError: (String) -> Void

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> BrushMetalView {
        let view = BrushMetalView(frame: .zero, device: MTLCreateSystemDefaultDevice())
        view.colorPixelFormat = AstraPipeline.pixelFormat
        view.clearColor = MTLClearColor(red: 0.012, green: 0.019, blue: 0.039, alpha: 1)
        view.colorspace = CGColorSpace(name: CGColorSpace.sRGB)
        view.framebufferOnly = true
        view.isPaused = true
        view.enableSetNeedsDisplay = true
        view.preferredFramesPerSecond = 60
        view.autoResizeDrawable = true
        view.delegate = context.coordinator
        view.onCursorChange = { [weak coordinator = context.coordinator, weak view] in
            coordinator?.needsRender = true
            view?.needsDisplay = true
        }
        context.coordinator.configure(view: view, onError: onError)
        return view
    }

    func updateNSView(_ view: BrushMetalView, context: Context) {
        view.onInteraction = onInteraction
        context.coordinator.update(self)
        view.needsDisplay = true
    }

    static func dismantleNSView(_ view: BrushMetalView, coordinator: Coordinator) {
        view.delegate = nil
        view.onCursorChange = nil
        view.onInteraction = nil
        coordinator.frame = nil
    }

    final class Coordinator: NSObject, MTKViewDelegate {
        private var pipeline: AstraPipeline?
        private let inFlight = DispatchSemaphore(value: 2)
        fileprivate var frame: RenderFrame?
        fileprivate var needsRender = true
        private var palette: ColorPalette = .aurora
        private var display: FieldDisplay = .vorticity
        private var exposure: Double = 1
        private var showFlowLines = false
        private var brushRadius: Double = 0.035
        private var onError: (String) -> Void = { _ in }
        private var lastError: String?

        fileprivate func configure(view: BrushMetalView, onError: @escaping (String) -> Void) {
            self.onError = onError
            do {
                guard let device = view.device else {
                    throw CanvasError.unavailable("Metal is unavailable. MLXAstra needs a Mac with a supported Metal GPU.")
                }
                pipeline = try AstraPipeline(device: device)
            } catch { report(error.localizedDescription) }
        }

        fileprivate func update(_ source: TurbulenceCanvas) {
            if frame !== source.frame || palette != source.palette || display != source.display ||
                exposure != source.exposure || showFlowLines != source.showFlowLines || brushRadius != source.brushRadius {
                needsRender = true
            }
            frame = source.frame
            palette = source.palette
            display = source.display
            exposure = source.exposure
            showFlowLines = source.showFlowLines
            brushRadius = source.brushRadius
            onError = source.onError
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            needsRender = true
            view.needsDisplay = true
        }

        func draw(in view: MTKView) {
            guard needsRender, let pipeline, view.drawableSize.width > 0, view.drawableSize.height > 0 else { return }
            guard inFlight.wait(timeout: .now()) == .success else { return }
            var submitted = false
            defer { if !submitted { inFlight.signal() } }
            guard let descriptor = view.currentRenderPassDescriptor, let drawable = view.currentDrawable,
                  let command = pipeline.queue.makeCommandBuffer() else { return }
            command.label = "Astra · present"
            let retainedFrame = frame
            do {
                if let retainedFrame {
                    var uniforms = AstraUniforms(frame: retainedFrame, palette: palette, display: display,
                                                 exposure: exposure, showFlowLines: showFlowLines, size: view.drawableSize)
                    uniforms.brushRadius = Float(brushRadius)
                    if let brush = view as? BrushMetalView {
                        uniforms.cursor = brush.cursorPosition
                        uniforms.cursorVisible = brush.cursorIsVisible ? 1 : 0
                        uniforms.cursorNegative = brush.cursorIsNegative ? 1 : 0
                    }
                    try pipeline.encode(frame: retainedFrame, uniforms: &uniforms, descriptor: descriptor, command: command)
                } else {
                    guard let encoder = command.makeRenderCommandEncoder(descriptor: descriptor) else { return }
                    encoder.endEncoding()
                }
                command.present(drawable)
                let semaphore = inFlight
                command.addCompletedHandler { [weak self, weak view, retainedFrame] completed in
                    // MTLBuffer alone does not retain the MLX allocation owner.
                    withExtendedLifetime(retainedFrame) {}
                    semaphore.signal()
                    DispatchQueue.main.async {
                        if let error = completed.error { self?.report("Metal rendering failed: \(error.localizedDescription)") }
                        if self?.needsRender == true { view?.needsDisplay = true }
                    }
                }
                needsRender = false
                submitted = true
                command.commit()
            } catch { report(error.localizedDescription) }
        }

        private func report(_ message: String) {
            guard message != lastError else { return }
            lastError = message
            DispatchQueue.main.async { [weak self] in self?.onError(message) }
        }
    }
}

/// AppKit preserves precise mouse coordinates and supports right-button dragging.
final class BrushMetalView: MTKView {
    var onInteraction: ((Float, Float, Bool) -> Void)?
    var onCursorChange: (() -> Void)?
    fileprivate var cursorPosition = SIMD2<Float>(-1, -1)
    fileprivate var cursorIsVisible = false
    fileprivate var cursorIsNegative = false
    private var mouseTracking: NSTrackingArea?
    private var lastInjection: CFTimeInterval = 0

    override var acceptsFirstResponder: Bool { true }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let mouseTracking { removeTrackingArea(mouseTracking) }
        let tracking = NSTrackingArea(rect: .zero,
                                      options: [.mouseEnteredAndExited, .mouseMoved, .activeInKeyWindow, .inVisibleRect, .enabledDuringMouseDrag],
                                      owner: self, userInfo: nil)
        addTrackingArea(tracking)
        mouseTracking = tracking
    }

    override func resetCursorRects() { addCursorRect(bounds, cursor: .crosshair) }
    override func mouseEntered(with event: NSEvent) { updateCursor(event) }
    override func mouseMoved(with event: NSEvent) { updateCursor(event) }
    override func mouseExited(with event: NSEvent) {
        cursorIsVisible = false
        onCursorChange?()
    }
    override func flagsChanged(with event: NSEvent) { updateCursor(event) }
    override func mouseDown(with event: NSEvent) { inject(event, force: true) }
    override func rightMouseDown(with event: NSEvent) { inject(event, force: true) }
    override func mouseDragged(with event: NSEvent) { inject(event, force: false) }
    override func rightMouseDragged(with event: NSEvent) { inject(event, force: false) }

    private func updateCursor(_ event: NSEvent) {
        guard bounds.width > 0, bounds.height > 0 else { return }
        let point = convert(event.locationInWindow, from: nil)
        let rawY = Float(point.y / bounds.height)
        cursorPosition = SIMD2(Float(point.x / bounds.width), isFlipped ? 1 - rawY : rawY)
        cursorIsVisible = bounds.contains(point)
        cursorIsNegative = event.modifierFlags.contains(.option) || event.type == .rightMouseDown || event.type == .rightMouseDragged
        onCursorChange?()
    }

    private func inject(_ event: NSEvent, force: Bool) {
        window?.makeFirstResponder(self)
        updateCursor(event)
        guard cursorIsVisible else { return }
        let now = CACurrentMediaTime()
        guard force || now - lastInjection >= 1.0 / 60.0 else { return }
        lastInjection = now
        onInteraction?(min(max(cursorPosition.x, 0), 1), min(max(cursorPosition.y, 0), 1), cursorIsNegative)
    }
}

enum TurbulenceSnapshot {
    /// A deliberately explicit, single GPU readback; interactive rendering stays zero-copy.
    static func pngData(frame: RenderFrame, palette: ColorPalette, display: FieldDisplay,
                        exposure: Double, showFlowLines: Bool, pixelSize: Int = 1800) throws -> Data {
        guard (64...8192).contains(pixelSize) else {
            throw CanvasError.unavailable("Snapshot size must be between 64 and 8192 pixels.")
        }
        let pipeline = try AstraPipeline(device: frame.buffer.device)
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: AstraPipeline.pixelFormat,
                                                                          width: pixelSize, height: pixelSize, mipmapped: false)
        textureDescriptor.usage = [.renderTarget]
        textureDescriptor.storageMode = .shared
        guard let texture = pipeline.device.makeTexture(descriptor: textureDescriptor),
              let command = pipeline.queue.makeCommandBuffer() else {
            throw CanvasError.unavailable("Metal could not allocate the snapshot. Try a smaller image size.")
        }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = texture
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        var uniforms = AstraUniforms(frame: frame, palette: palette, display: display,
                                     exposure: exposure, showFlowLines: showFlowLines,
                                     size: CGSize(width: pixelSize, height: pixelSize))
        try pipeline.encode(frame: frame, uniforms: &uniforms, descriptor: pass, command: command)
        command.commit()
        command.waitUntilCompleted()
        withExtendedLifetime(frame) {}
        if let error = command.error { throw error }
        let bytesPerRow = pixelSize * 4
        var pixels = Data(count: bytesPerRow * pixelSize)
        pixels.withUnsafeMutableBytes { bytes in
            texture.getBytes(bytes.baseAddress!, bytesPerRow: bytesPerRow,
                             from: MTLRegionMake2D(0, 0, pixelSize, pixelSize), mipmapLevel: 0)
        }
        guard let provider = CGDataProvider(data: pixels as CFData),
              let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let image = CGImage(width: pixelSize, height: pixelSize, bitsPerComponent: 8, bitsPerPixel: 32,
                                  bytesPerRow: bytesPerRow, space: colorSpace,
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue)
                                    .union(.byteOrder32Little),
                                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
              let png = NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:]) else {
            throw CanvasError.unavailable("The snapshot could not be encoded as a PNG image.")
        }
        return png
    }
}
