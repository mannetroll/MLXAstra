import AppKit
import SwiftUI

@main
struct MLXAstraApp: App {
    @StateObject private var model = SimulationModel()

    init() {
        AppDiagnostics.runIfRequested()
    }

    var body: some Scene {
        Window("MLX Astra", id: "astra") {
            AstraView(model: model)
                .frame(minWidth: 1080, minHeight: 760)
                .preferredColorScheme(.dark)
                .onAppear {
                    model.start()
                    AppDiagnostics.beginLiveCheckIfRequested(model: model)
                }
                .onDisappear { model.stop() }
                .onReceive(NotificationCenter.default.publisher(for: NSApplication.didHideNotification)) { _ in model.stop() }
                .onReceive(NotificationCenter.default.publisher(for: NSApplication.didUnhideNotification)) { _ in model.start() }
                .onReceive(NotificationCenter.default.publisher(for: NSWindow.didMiniaturizeNotification)) { _ in model.stop() }
                .onReceive(NotificationCenter.default.publisher(for: NSWindow.didDeminiaturizeNotification)) { _ in model.start() }
        }
        .defaultSize(width: 1280, height: 900)
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentMinSize)
        .commands {
            CommandGroup(replacing: .newItem) {}
            CommandGroup(replacing: .saveItem) {
                Button("Save Snapshot…") { model.saveSnapshot() }
                    .keyboardShortcut("s", modifiers: .command)
                    .disabled(!model.isReady)
            }
            CommandMenu("Simulation") {
                Button(model.isRunning ? "Pause" : "Resume") { model.togglePause() }
                    .keyboardShortcut(.space, modifiers: [])
                Button("Single Step") { model.stepOnce() }
                    .keyboardShortcut(.rightArrow, modifiers: [])
                Button("Reset Flow") { model.reset() }
                    .keyboardShortcut("r", modifiers: [])
                Divider()
                Button(model.isFocusMode ? "Show Controls" : "Focus on Flow") { model.isFocusMode.toggle() }
                    .keyboardShortcut("f", modifiers: [])
            }
            CommandGroup(replacing: .help) {
                Button("Astra Controls & Physics") {
                    let alert = NSAlert()
                    alert.messageText = "A small laboratory for a turbulent world."
                    alert.informativeText = "Drag to add vortices. Option-drag or right-drag reverses the spin. Space pauses; → advances one step; R resets; F focuses; ⌘S saves a PNG.\n\nThe periodic 2π × 2π domain solves the incompressible 2D Navier–Stokes vorticity equation using MLX FFTs, spectral dealiasing, and viscosity. Colours show signed vorticity or speed; the optional flow overlay uses the velocity field.\n\nSimulation speed is limited by an adaptive stability condition. Steps/s measures integration steps per wall-clock second, independently of the display refresh rate. Multiple steps can run between screen refreshes."
                    alert.addButton(withTitle: "Explore")
                    alert.runModal()
                }
            }
        }
    }
}
