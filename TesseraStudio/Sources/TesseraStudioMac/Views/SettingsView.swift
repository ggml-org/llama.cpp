import SwiftUI
import TesseraCore

/// macOS Settings scene (Settings { SettingsView() }), backed by @AppStorage.
struct SettingsView: View {
    // General
    @AppStorage(TesseraSettingsKey.defaultRuntime) private var defaultRuntime = TesseraRuntime.onDevice.rawValue
    @AppStorage(TesseraSettingsKey.modelDirectory) private var modelDirectory = TesseraSettingsDefault.modelDirectory
    @AppStorage(TesseraSettingsKey.threadCount) private var threadCount = TesseraSettingsDefault.threadCount
    // Agent
    @AppStorage(TesseraSettingsKey.maxIterations) private var maxIterations = TesseraSettingsDefault.maxIterations
    @AppStorage(TesseraSettingsKey.defaultApprovalLevel) private var defaultApprovalLevel = ApprovalLevel.prompt.rawValue
    @AppStorage(TesseraSettingsKey.tokenBudget) private var tokenBudget = TesseraSettingsDefault.tokenBudget
    // Advanced
    @AppStorage(TesseraSettingsKey.telemetryEnabled) private var telemetryEnabled = TesseraSettingsDefault.telemetryEnabled
    @AppStorage(TesseraSettingsKey.logLevel) private var logLevel = TesseraSettingsDefault.logLevel
    @AppStorage(TesseraSettingsKey.cliPath) private var cliPath = TesseraSettingsDefault.cliPath

    var body: some View {
        TabView {
            generalTab
                .tabItem { Label("General", systemImage: "gearshape") }
            agentTab
                .tabItem { Label("Agent", systemImage: "cpu") }
            advancedTab
                .tabItem { Label("Advanced", systemImage: "slider.horizontal.3") }
        }
        .frame(width: 480, height: 300)
    }

    private var generalTab: some View {
        Form {
            Picker("Default runtime", selection: $defaultRuntime) {
                ForEach(TesseraRuntime.allCases, id: \.rawValue) { rt in
                    Text(rt.displayName).tag(rt.rawValue)
                }
            }
            TextField("Model directory", text: $modelDirectory)
            Stepper("Threads: \(threadCount == 0 ? "all cores" : "\(threadCount)")", value: $threadCount, in: 0...64)
        }
        .formStyle(.grouped)
        .padding()
    }

    private var agentTab: some View {
        Form {
            Stepper("Max tool iterations: \(maxIterations)", value: $maxIterations, in: 1...50)
            Picker("Default approval level", selection: $defaultApprovalLevel) {
                ForEach(ApprovalLevel.allCases, id: \.rawValue) { level in
                    Text(level.rawValue.capitalized).tag(level.rawValue)
                }
            }
            TextField("Token budget", value: $tokenBudget, format: .number)
        }
        .formStyle(.grouped)
        .padding()
    }

    private var advancedTab: some View {
        Form {
            Toggle("Enable telemetry", isOn: $telemetryEnabled)
            Picker("Log level", selection: $logLevel) {
                ForEach(TesseraLogLevel.allCases, id: \.rawValue) { level in
                    Text(level.rawValue.uppercased()).tag(level.rawValue)
                }
            }
            TextField("Custom CLI path", text: $cliPath)
        }
        .formStyle(.grouped)
        .padding()
    }
}
