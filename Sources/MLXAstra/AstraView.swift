import SwiftUI
import MLXAstraCore

private enum AstraTheme {
    static let background = Color(red: 0.035, green: 0.047, blue: 0.073)
    static let panel = Color(red: 0.061, green: 0.076, blue: 0.106)
    static let secondary = Color(red: 0.48, green: 0.55, blue: 0.65)
    static let accent = Color(red: 0.44, green: 0.91, blue: 0.82)
    static let border = Color.white.opacity(0.075)
}

struct AstraView: View {
    @ObservedObject var model: SimulationModel

    var body: some View {
        VStack(spacing: 0) {
            header
            HStack(alignment: .top, spacing: 18) {
                simulationPanel
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                if !model.isFocusMode {
                    inspector
                        .frame(width: 286)
                        .transition(.move(edge: .trailing).combined(with: .opacity))
                }
            }
            .padding(.horizontal, 22)
            .padding(.bottom, 14)
            footer
        }
        .background(AstraTheme.background)
        .foregroundStyle(Color.white.opacity(0.9))
        .tint(AstraTheme.accent)
        .preferredColorScheme(.dark)
        .animation(.easeInOut(duration: 0.24), value: model.isFocusMode)
    }

    private var header: some View {
        HStack(spacing: 13) {
            ZStack {
                Circle().stroke(AstraTheme.accent.opacity(0.23), lineWidth: 1)
                Image(systemName: "hurricane")
                    .font(.system(size: 23, weight: .light))
                    .foregroundStyle(AstraTheme.accent)
            }
            .frame(width: 40, height: 40)
            VStack(alignment: .leading, spacing: 3) {
                Text("ASTRA")
                    .font(.system(size: 17, weight: .semibold, design: .rounded))
                    .tracking(5)
                Text("TURBULENCE LAB")
                    .font(.system(size: 8, weight: .medium, design: .monospaced))
                    .tracking(2.3)
                    .foregroundStyle(AstraTheme.secondary)
            }
            Rectangle().fill(AstraTheme.border).frame(width: 1, height: 29).padding(.horizontal, 12)
            VStack(alignment: .leading, spacing: 4) {
                Text("Order emerges from motion.")
                    .font(.system(size: 13, weight: .medium))
                Text("An interactive two-dimensional fluid")
                    .font(.system(size: 11))
                    .foregroundStyle(AstraTheme.secondary)
            }
            Spacer(minLength: 12)
            HStack(spacing: 6) {
                Circle().fill(model.isReady ? AstraTheme.accent : AstraTheme.secondary).frame(width: 5, height: 5)
                Text("MLX / GPU")
                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                    .tracking(0.5)
            }
            .foregroundStyle(AstraTheme.accent)
            .padding(.horizontal, 11)
            .padding(.vertical, 8)
            .background(AstraTheme.accent.opacity(0.065), in: Capsule())
            .help(model.deviceName)
            iconButton(model.isFocusMode ? "arrow.down.right.and.arrow.up.left" : "arrow.up.left.and.arrow.down.right", help: "Focus mode · F") {
                model.isFocusMode.toggle()
            }
            iconButton("square.and.arrow.up", help: "Save a PNG snapshot · ⌘S") {
                model.saveSnapshot()
            }
            .disabled(!model.isReady)
        }
        .padding(.horizontal, 24)
        .frame(height: 77)
    }

    private var simulationPanel: some View {
        VStack(spacing: 0) {
            HStack {
                HStack(spacing: 8) {
                    Circle().fill(model.isRunning && model.isReady ? AstraTheme.accent : AstraTheme.secondary)
                        .frame(width: 5, height: 5)
                    Text(model.display == .vorticity ? "VORTICITY FIELD" : "VELOCITY MAGNITUDE")
                        .font(.system(size: 10, weight: .medium, design: .monospaced))
                        .tracking(1.7)
                }
                Spacer()
                Text("\(model.config.gridSize)² · \(model.config.paddedGridSize)² padded")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(AstraTheme.secondary)
                Text("PERIODIC")
                    .font(.system(size: 8, weight: .medium, design: .monospaced))
                    .tracking(1)
                    .foregroundStyle(AstraTheme.secondary)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 4)
                    .overlay(RoundedRectangle(cornerRadius: 4).stroke(AstraTheme.border))
            }
            .padding(.horizontal, 18)
            .frame(height: 43)

            GeometryReader { geometry in
                let side = max(0, min(geometry.size.width, geometry.size.height))
                ZStack {
                    TurbulenceCanvas(
                        frame: model.frame,
                        palette: model.palette,
                        display: model.display,
                        exposure: model.exposure,
                        showFlowLines: model.showFlowLines,
                        brushRadius: model.brushRadius,
                        onInteraction: { x, y, negative in model.inject(x: x, y: y, negative: negative) },
                        onError: { model.statusMessage = $0 }
                    )
                    .accessibilityLabel("Interactive turbulence field")
                    .accessibilityHint("Drag to add vortices. Hold Option or use the right mouse button to reverse their rotation.")

                    if !model.isReady {
                        VStack(spacing: 14) {
                            if model.statusMessage == nil {
                                ProgressView().controlSize(.small)
                            } else {
                                Image(systemName: "exclamationmark.circle")
                                    .font(.system(size: 24, weight: .light))
                                    .foregroundStyle(AstraTheme.secondary)
                            }
                            Text(model.statusMessage == nil ? "Preparing the flow" : "Unable to start the flow")
                                .font(.system(size: 12, weight: .medium))
                            Text(model.statusMessage == nil ? "Compiling MLX kernels on your GPU" : "See the message below to continue")
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundStyle(AstraTheme.secondary)
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .background(AstraTheme.background.opacity(0.85))
                    }

                    VStack {
                        HStack {
                            Text(model.config.preset.title.uppercased())
                                .font(.system(size: 8, weight: .medium, design: .monospaced))
                                .tracking(1.4)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 7)
                                .background(.black.opacity(0.28), in: Capsule())
                            Spacer()
                        }
                        Spacer()
                        if let message = model.statusMessage {
                            HStack(alignment: .top, spacing: 8) {
                                Image(systemName: "info.circle")
                                Text(message).textSelection(.enabled)
                                Spacer(minLength: 0)
                                Button { model.statusMessage = nil } label: {
                                    Image(systemName: "xmark").font(.system(size: 9, weight: .semibold))
                                }
                                .buttonStyle(.plain)
                                .help("Dismiss message")
                            }
                            .font(.system(size: 11))
                            .padding(12)
                            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10))
                            .padding(.bottom, 7)
                        }
                        HStack(alignment: .bottom) {
                            HStack(spacing: 0) {
                                Button { model.togglePause() } label: {
                                    Image(systemName: model.isRunning ? "pause.fill" : "play.fill")
                                        .font(.system(size: 12, weight: .semibold))
                                        .frame(width: 39, height: 35)
                                }
                                .help(model.isRunning ? "Pause · Space" : "Resume · Space")
                                .disabled(!model.isReady)
                                Rectangle().fill(.white.opacity(0.13)).frame(width: 1, height: 15)
                                Button { model.stepOnce() } label: {
                                    Image(systemName: "forward.frame.fill")
                                        .font(.system(size: 12))
                                        .frame(width: 37, height: 35)
                                }
                                .help("Advance one step")
                                .disabled(!model.isReady)
                                Button { model.reset() } label: {
                                    Image(systemName: "arrow.counterclockwise")
                                        .font(.system(size: 12))
                                        .frame(width: 35, height: 35)
                                }
                                .help("Reset simulation · R")
                            }
                            .buttonStyle(.plain)
                            .background(.ultraThinMaterial, in: Capsule())
                            .overlay(Capsule().stroke(.white.opacity(0.12)))
                            Spacer()
                            Text(String(format: "t  %.2f", model.stats.time))
                                .font(.system(size: 10, weight: .medium, design: .monospaced))
                                .monospacedDigit()
                                .padding(.horizontal, 11)
                                .padding(.vertical, 8)
                                .background(.black.opacity(0.3), in: Capsule())
                        }
                    }
                    .padding(15)
                }
                .frame(width: side, height: side)
                .clipShape(RoundedRectangle(cornerRadius: 9))
                .overlay(RoundedRectangle(cornerRadius: 9).stroke(AstraTheme.border))
                .frame(width: geometry.size.width, height: geometry.size.height)
            }
            .padding(.horizontal, 12)

            HStack(spacing: 8) {
                Image(systemName: "cursorarrow.motionlines")
                    .font(.system(size: 10))
                Text("Drag to stir")
                Circle().fill(AstraTheme.secondary.opacity(0.5)).frame(width: 2, height: 2)
                Text("⌥ drag to reverse")
                Spacer()
                colorLegend
            }
            .font(.system(size: 10))
            .foregroundStyle(AstraTheme.secondary)
            .padding(.horizontal, 19)
            .frame(height: 35)

            if !model.isFocusMode {
                Rectangle().fill(AstraTheme.border).frame(height: 1)
                metrics
            }
        }
        .background(AstraTheme.panel.opacity(0.42), in: RoundedRectangle(cornerRadius: 15))
        .overlay(RoundedRectangle(cornerRadius: 15).stroke(AstraTheme.border))
    }

    private var colorLegend: some View {
        HStack(spacing: 6) {
            Text(model.display == .vorticity ? "−ω" : "0")
            RoundedRectangle(cornerRadius: 2)
                .fill(LinearGradient(colors: paletteColors(model.palette), startPoint: .leading, endPoint: .trailing))
                .frame(width: 62, height: 4)
            Text(model.display == .vorticity ? "+ω" : "|u|")
        }
        .font(.system(size: 9, design: .monospaced))
    }

    private var metrics: some View {
        HStack(spacing: 0) {
            metric(title: "SIMULATED TIME", value: metricNumber(model.simulationTimePerSecond), unit: "per wall s")
                .foregroundStyle(AstraTheme.accent)
                .help("Average physical simulation time per active wall second since reset. Excludes initialization, pauses, and manual steps; retains the displayed average while paused.")
            metricDivider
            metric(title: "INITIAL TURNOVERS", value: metricNumber(model.initialTurnoversPerSecond), unit: "per wall s")
                .foregroundStyle(AstraTheme.accent)
                .help("Average initial turnovers per active wall second since reset: the cumulative simulation rate divided by the run's fixed initial turnover time τ₀. Retained while paused.")
            metricDivider
            VStack(alignment: .leading, spacing: 5) {
                metricLabel("KINETIC ENERGY")
                HStack(spacing: 4) {
                    Text(metricNumber(Double(model.stats.energy)))
                        .font(.system(size: 16, weight: .light, design: .monospaced))
                        .monospacedDigit()
                        .lineLimit(1).minimumScaleFactor(0.8)
                    EnergySparkline(samples: model.history)
                        .stroke(AstraTheme.accent.opacity(0.85), style: StrokeStyle(lineWidth: 1, lineCap: .round, lineJoin: .round))
                        .frame(width: 24, height: 16)
                        .accessibilityLabel("Kinetic energy history")
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            metricDivider
            metric(title: "ENSTROPHY", value: metricNumber(Double(model.stats.enstrophy)))
            metricDivider
            metric(title: "SOLVER", value: model.isReady ? String(format: "%.1f", model.millisecondsPerStep) : "—", unit: "ms")
                .help("Mean time per integration step")
            metricDivider
            metric(title: "SIMULATION", value: model.isReady ? String(format: "%.0f", model.stepsPerSecond) : "—", unit: "steps/s")
                .help("Solver integration steps per wall-clock second; display refresh is independent.")
        }
        .padding(.horizontal, 20)
        .frame(height: 62)
    }

    private func metric(title: String, value: String, unit: String? = nil) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            metricLabel(title)
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(value).font(.system(size: 16, weight: .light, design: .monospaced)).monospacedDigit()
                    .lineLimit(1).minimumScaleFactor(0.8)
                if let unit {
                    Text(unit).font(.system(size: 7.5, design: .monospaced)).foregroundStyle(AstraTheme.secondary)
                        .fixedSize()
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func metricLabel(_ title: String) -> some View {
        Text(title)
            .font(.system(size: 7, weight: .medium, design: .monospaced))
            .tracking(0.5)
            .foregroundStyle(AstraTheme.secondary)
            .lineLimit(1)
    }

    private var metricDivider: some View {
        Rectangle().fill(AstraTheme.border).frame(width: 1, height: 28).padding(.horizontal, 7)
    }

    private var inspector: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                VStack(alignment: .leading, spacing: 13) {
                    sectionTitle("01", "INITIAL CONDITIONS")
                    VStack(spacing: 5) {
                        ForEach(FlowPreset.allCases) { preset in
                            Button { model.applyPreset(preset) } label: {
                                HStack(spacing: 10) {
                                    Image(systemName: preset.symbol)
                                        .font(.system(size: 14, weight: .light))
                                        .frame(width: 21)
                                    Text(preset.title).font(.system(size: 12, weight: model.config.preset == preset ? .medium : .regular))
                                    Spacer()
                                    if model.config.preset == preset {
                                        Circle().fill(AstraTheme.accent).frame(width: 5, height: 5)
                                    }
                                }
                                .foregroundStyle(model.config.preset == preset ? AstraTheme.accent : Color.white.opacity(0.64))
                                .padding(.horizontal, 12)
                                .frame(height: 34)
                                .background(model.config.preset == preset ? AstraTheme.accent.opacity(0.085) : .clear, in: RoundedRectangle(cornerRadius: 7))
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                            .accessibilityAddTraits(model.config.preset == preset ? .isSelected : [])
                        }
                    }
                    Text(model.config.preset.subtitle)
                        .font(.system(size: 10))
                        .foregroundStyle(AstraTheme.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.horizontal, 2)
                }
                inspectorDivider
                VStack(alignment: .leading, spacing: 16) {
                    sectionTitle("02", "DYNAMICS")
                    HStack {
                        Text("Resolution").font(.system(size: 12))
                        Spacer()
                        Picker("Resolution", selection: $model.config.gridSize) {
                            ForEach(SimulationConfiguration.gridSizes, id: \.self) { size in
                                Text("\(size) × \(size)").tag(size)
                            }
                        }
                        .labelsHidden()
                        .frame(width: 116)
                        .controlSize(.small)
                        .help("State and display use N × N points; nonlinear terms use a 3N/2 × 3N/2 grid. Changing resolution restarts the simulation.")
                    }
                    HStack {
                        Text("Show every").font(.system(size: 12))
                        Spacer()
                        Picker("Show every", selection: $model.stepsPerFrame) {
                            ForEach([1, 2, 5, 10, 20, 50], id: \.self) { steps in
                                Text("\(steps) \(steps == 1 ? "step" : "steps")").tag(steps)
                            }
                        }
                        .labelsHidden()
                        .frame(width: 116)
                        .controlSize(.small)
                        .help("Maximum integration steps per displayed field. Batches shorten automatically to keep controls responsive.")
                    }
                    sliderControl("Viscosity", value: String(format: "%.5f", model.config.viscosity), binding: Binding(
                        get: { log10(Double(model.config.viscosity)) },
                        set: { model.config.viscosity = Float(pow(10, $0)) }
                    ), range: -5...log10(0.003), low: "Fluid", high: "Viscous")
                    sliderControl("Energy injection", value: String(format: "%.2f", model.config.forcing), binding: Binding(
                        get: { Double(model.config.forcing) }, set: { model.config.forcing = Float($0) }
                    ), range: 0...2, low: "Off", high: "Driven")
                    sliderControl("Time scale", value: String(format: "%.2f×", model.config.timeScale), binding: Binding(
                        get: { Double(model.config.timeScale) }, set: { model.config.timeScale = Float($0) }
                    ), range: 0.25...3)
                }
                inspectorDivider
                VStack(alignment: .leading, spacing: 15) {
                    sectionTitle("03", "APPEARANCE")
                    Picker("Field", selection: $model.display) {
                        ForEach(FieldDisplay.allCases) { field in Text(field.title).tag(field) }
                    }
                    .pickerStyle(.segmented)
                    .controlSize(.small)
                    HStack(spacing: 8) {
                        ForEach(ColorPalette.allCases) { palette in
                            Button { model.palette = palette } label: {
                                VStack(spacing: 7) {
                                    RoundedRectangle(cornerRadius: 4)
                                        .fill(LinearGradient(colors: paletteColors(palette), startPoint: .leading, endPoint: .trailing))
                                        .frame(height: 13)
                                    Text(palette.title)
                                        .font(.system(size: 9, weight: model.palette == palette ? .medium : .regular))
                                        .foregroundStyle(model.palette == palette ? .white : AstraTheme.secondary)
                                }
                                .padding(8)
                                .frame(maxWidth: .infinity)
                                .background(.white.opacity(model.palette == palette ? 0.055 : 0.015), in: RoundedRectangle(cornerRadius: 7))
                                .overlay(RoundedRectangle(cornerRadius: 7).stroke(model.palette == palette ? AstraTheme.accent.opacity(0.6) : AstraTheme.border))
                            }
                            .buttonStyle(.plain)
                            .accessibilityLabel("\(palette.title) palette")
                            .accessibilityAddTraits(model.palette == palette ? .isSelected : [])
                        }
                    }
                    sliderControl("Exposure", value: String(format: "%.1f×", model.exposure), binding: $model.exposure, range: 0.25...3)
                    Toggle(isOn: $model.showFlowLines) {
                        HStack(spacing: 7) {
                            Image(systemName: "wind").foregroundStyle(AstraTheme.secondary)
                            Text("Velocity direction")
                        }
                        .font(.system(size: 12))
                    }
                    .toggleStyle(.switch)
                    .controlSize(.mini)
                }
                inspectorDivider
                VStack(alignment: .leading, spacing: 14) {
                    sectionTitle("04", "INTERACTION")
                    sliderControl("Vortex brush", value: String(format: "%.1f%%", model.brushRadius * 100), binding: $model.brushRadius, range: 0.01...0.12, low: "Fine", high: "Broad")
                    Text("Drag across the field to add rotation. Hold Option or use a right drag to spin the other way.")
                        .font(.system(size: 10))
                        .foregroundStyle(AstraTheme.secondary)
                        .lineSpacing(3)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(18)
        }
        .scrollIndicators(.automatic)
        .background(AstraTheme.panel.opacity(0.68), in: RoundedRectangle(cornerRadius: 15))
        .overlay(RoundedRectangle(cornerRadius: 15).stroke(AstraTheme.border))
    }

    private var inspectorDivider: some View {
        Rectangle().fill(AstraTheme.border).frame(height: 1)
    }

    private func sectionTitle(_ number: String, _ title: String) -> some View {
        HStack(spacing: 8) {
            Text(number).foregroundStyle(AstraTheme.accent.opacity(0.7))
            Text(title).foregroundStyle(AstraTheme.secondary)
        }
        .font(.system(size: 9, weight: .medium, design: .monospaced))
        .tracking(1.2)
    }

    private func sliderControl(_ title: String, value: String, binding: Binding<Double>, range: ClosedRange<Double>, low: String? = nil, high: String? = nil) -> some View {
        VStack(spacing: 6) {
            HStack {
                Text(title).font(.system(size: 12))
                Spacer()
                Text(value)
                    .font(.system(size: 10, design: .monospaced))
                    .monospacedDigit()
                    .foregroundStyle(AstraTheme.accent.opacity(0.9))
            }
            Slider(value: binding, in: range)
                .controlSize(.mini)
                .accessibilityLabel(title)
                .accessibilityValue(value)
            if let low, let high {
                HStack {
                    Text(low)
                    Spacer()
                    Text(high)
                }
                .font(.system(size: 8))
                .foregroundStyle(AstraTheme.secondary.opacity(0.85))
            }
        }
    }

    private var footer: some View {
        HStack(spacing: 7) {
            Image(systemName: "cpu").font(.system(size: 9))
            Text(model.deviceName)
            Text("·").padding(.horizontal, 3)
            Text("PSEUDOSPECTRAL NAVIER–STOKES")
                .tracking(0.8)
            Spacer()
            Text("SPACE  pause")
            Text("R  reset").padding(.leading, 10)
            Text("F  focus").padding(.leading, 10)
        }
        .font(.system(size: 8, design: .monospaced))
        .foregroundStyle(AstraTheme.secondary.opacity(0.8))
        .padding(.horizontal, 25)
        .padding(.bottom, 12)
    }

    private func iconButton(_ symbol: String, help: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.system(size: 13))
                .foregroundStyle(Color.white.opacity(0.63))
                .frame(width: 32, height: 32)
                .background(.white.opacity(0.035), in: RoundedRectangle(cornerRadius: 8))
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(help)
        .accessibilityLabel(help)
    }

    private func metricNumber(_ value: Double) -> String {
        guard model.isReady, value.isFinite else { return "—" }
        if value == 0 { return "0.00" }
        if abs(value) < 0.001 || abs(value) >= 1000 { return String(format: "%.2e", value) }
        return String(format: "%.3f", value)
    }

    private func paletteColors(_ palette: ColorPalette) -> [Color] {
        switch palette {
        case .aurora:
            return [Color(red: 0.22, green: 0.25, blue: 0.82), Color(red: 0.06, green: 0.09, blue: 0.18), Color(red: 0.39, green: 0.94, blue: 0.73)]
        case .ember:
            return [Color(red: 0.42, green: 0.14, blue: 0.57), Color(red: 0.12, green: 0.04, blue: 0.1), Color(red: 1, green: 0.64, blue: 0.27)]
        case .glacier:
            return [Color(red: 0.12, green: 0.29, blue: 0.58), Color(red: 0.03, green: 0.12, blue: 0.19), Color(red: 0.64, green: 0.94, blue: 1)]
        }
    }
}

private struct EnergySparkline: Shape {
    var samples: [Double]

    func path(in rect: CGRect) -> Path {
        let values = Array(samples.filter(\.isFinite).suffix(100))
        guard values.count > 1, let low = values.min(), let high = values.max() else { return Path() }
        let spread = max(high - low, max(abs(high), 1e-12) * 0.02)
        let middle = (high + low) * 0.5
        var path = Path()
        for (index, value) in values.enumerated() {
            let x = rect.minX + CGFloat(index) / CGFloat(values.count - 1) * rect.width
            let y = rect.midY - CGFloat((value - middle) / spread) * rect.height * 0.85
            if index == 0 { path.move(to: CGPoint(x: x, y: y)) }
            else { path.addLine(to: CGPoint(x: x, y: y)) }
        }
        return path
    }
}
