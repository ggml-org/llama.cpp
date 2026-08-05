#if os(iOS)
import SwiftUI
import TesseraCore

/// iOS Settings tab: the same settings as macOS in a Form layout.
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
    @AppStorage(TesseraSettingsKey.tesseraCLIPath) private var tesseraCLIPath = TesseraSettingsDefault.tesseraCLIPath
    @AppStorage(TesseraSettingsKey.tesseraPythonPath) private var tesseraPythonPath = TesseraSettingsDefault.tesseraPythonPath
    // LLM provider
    @AppStorage(TesseraSettingsKey.llmProviderType) private var llmProviderType = TesseraSettingsDefault.llmProviderType
    @AppStorage(TesseraSettingsKey.remoteAPIBaseURL) private var remoteAPIBaseURL = TesseraSettingsDefault.remoteAPIBaseURL
    @AppStorage(TesseraSettingsKey.remoteAPIKey) private var remoteAPIKey = TesseraSettingsDefault.remoteAPIKey
    @AppStorage(TesseraSettingsKey.remoteModelName) private var remoteModelName = TesseraSettingsDefault.remoteModelName
    @AppStorage(TesseraSettingsKey.remoteUseStreaming) private var remoteUseStreaming = TesseraSettingsDefault.remoteUseStreaming
    @AppStorage(TesseraSettingsKey.onDeviceModelPath) private var onDeviceModelPath = TesseraSettingsDefault.onDeviceModelPath
    @AppStorage(TesseraSettingsKey.onDeviceLibraryPath) private var onDeviceLibraryPath = TesseraSettingsDefault.onDeviceLibraryPath
    @AppStorage(TesseraSettingsKey.onDeviceContextLength) private var onDeviceContextLength = TesseraSettingsDefault.onDeviceContextLength
    @AppStorage(TesseraSettingsKey.onDeviceGPULayers) private var onDeviceGPULayers = TesseraSettingsDefault.onDeviceGPULayers

    var body: some View {
        Form {
            Section("General") {
                Picker("Default runtime", selection: $defaultRuntime) {
                    ForEach(TesseraRuntime.allCases, id: \.rawValue) { rt in
                        Text(rt.displayName).tag(rt.rawValue)
                    }
                }
                TextField("Model directory", text: $modelDirectory)
                Stepper("Threads: \(threadCount == 0 ? "all cores" : "\(threadCount)")", value: $threadCount, in: 0...64)
            }

            Section("Agent") {
                Stepper("Max tool iterations: \(maxIterations)", value: $maxIterations, in: 1...50)
                Picker("Default approval level", selection: $defaultApprovalLevel) {
                    ForEach(ApprovalLevel.allCases, id: \.rawValue) { level in
                        Text(level.rawValue.capitalized).tag(level.rawValue)
                    }
                }
                TextField("Token budget", value: $tokenBudget, format: .number)
            }

            Section("Model") {
                Picker("LLM provider", selection: $llmProviderType) {
                    ForEach(TesseraLLMProviderType.allCases, id: \.rawValue) { type in
                        Text(type.displayName).tag(type.rawValue)
                    }
                }
                if llmProviderType == TesseraLLMProviderType.remoteAPI.rawValue {
                    TextField("Base URL", text: $remoteAPIBaseURL)
                    SecureField("API key", text: $remoteAPIKey)
                    TextField("Model name", text: $remoteModelName)
                    Toggle("Stream responses (SSE)", isOn: $remoteUseStreaming)
                }
                if llmProviderType == TesseraLLMProviderType.onDevice.rawValue {
                    TextField("GGUF model path", text: $onDeviceModelPath)
                    TextField("libllama.dylib path (optional)", text: $onDeviceLibraryPath)
                    TextField("Context length", value: $onDeviceContextLength, format: .number)
                    Stepper("GPU layers: \(onDeviceGPULayers < 0 ? "all" : "\(onDeviceGPULayers)")",
                            value: $onDeviceGPULayers, in: -1...200)
                }
            }

            Section("Advanced") {
                Toggle("Enable telemetry", isOn: $telemetryEnabled)
                Picker("Log level", selection: $logLevel) {
                    ForEach(TesseraLogLevel.allCases, id: \.rawValue) { level in
                        Text(level.rawValue.uppercased()).tag(level.rawValue)
                    }
                }
                TextField("Custom CLI path", text: $cliPath)
                TextField("tessera-cli path", text: $tesseraCLIPath)
                TextField("Python interpreter path", text: $tesseraPythonPath)
            }

            Section {
                Text("Plea the Fifth")
                    .font(.headline)
                Text("Architecture reviewed against the design spec. External security audit pending - to be completed before public release.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                // iOS does not have a global hot-key. The covert
                // trigger is the only way to invoke the wipe from a
                // text field on iOS.
                VStack(alignment: .leading, spacing: 4) {
                    Text("Covert trigger phrase (advanced)")
                        .font(.subheadline)
                    TextField("At least 8 characters", text: $covertTriggerDraft)
                    Text("Choose something you can type naturally. Don't choose a famous quote.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    if let err = covertTriggerError {
                        Text(err).font(.caption).foregroundStyle(.red)
                    }
                    HStack {
                        Button("Save") { saveCovertTrigger() }
                            .disabled(covertTriggerDraft.trimmingCharacters(in: .whitespacesAndNewlines).count < PleadTheFifthSettings.minCovertPhraseLength)
                        Button("Test") { testCovertTrigger() }
                            .disabled(!PleadTheFifthSettings.covertTriggerConfigured)
                    }
                    if let note = covertTriggerTestNote {
                        Text(note).font(.caption).foregroundStyle(.secondary)
                    }
                }
                Toggle("Coercion mode", isOn: $coercionMode)
                Text("On iOS, coercion mode hides the in-app 'Plea the Fifth' shortcut. The covert trigger still works.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } header: {
                Text("Plea the Fifth")
            }
        }
        .navigationTitle("Settings")
        .onAppear { covertTriggerDraft = PleadTheFifthSettings.covertTriggerPhrase }
    }

    @AppStorage(PleadTheFifthSettingsKey.coercionMode)
    private var coercionMode: Bool = false

    @State private var covertTriggerDraft: String = ""
    @State private var covertTriggerError: String?
    @State private var covertTriggerTestNote: String?

    private func saveCovertTrigger() {
        do {
            try PleadTheFifthSettings.setCovertTriggerPhrase(covertTriggerDraft)
            covertTriggerError = nil
            covertTriggerTestNote = "Saved."
        } catch {
            covertTriggerError = "\(error.localizedDescription)"
        }
    }

    private func testCovertTrigger() {
        let phrase = PleadTheFifthSettings.covertTriggerPhrase
        let sample = "I said '\(phrase)' yesterday"
        let monitor = CovertTriggerMonitor()
        let wouldFire = monitor.shouldTrigger(in: sample)
        covertTriggerTestNote = wouldFire
            ? "The trigger would have fired on the sample sentence."
            : "The trigger did not fire. The phrase is set but the sample did not match."
    }
}
#endif
