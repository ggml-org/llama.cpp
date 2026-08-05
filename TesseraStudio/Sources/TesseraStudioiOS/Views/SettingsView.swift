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
    // "Plea the Fifth" coercion-resistant destruction. Same
    // shape as the macOS view: phrase in Keychain, mode flag
    // in UserDefaults, local draft + isSet for the UI.
    @AppStorage(TesseraSettingsKey.coercionMode) private var coercionMode = TesseraSettingsDefault.coercionMode
    @State private var covertTriggerPhraseDraft = ""
    @State private var covertTriggerIsSet = false
    @State private var covertTriggerEditing = false
    @State private var covertTriggerTestNote: String?

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

            // "Plea the Fifth" section. Same shape as macOS
            // but rendered as a collapsible Form Section. The
            // coercion mode toggle is here too, so the user
            // can flip it from either platform's Settings.
            // Collapsed by default when coercion mode is on
            // (design section 9.5).
            Section {
                if coercionMode && !pleadTheFifthExpandedIOS {
                    // Collapsed: only the section header
                    // is visible. The user can tap to expand.
                    Text("Plea the Fifth")
                        .foregroundStyle(.secondary)
                        .onTapGesture { pleadTheFifthExpandedIOS = true }
                } else {
                    if covertTriggerIsSet && !covertTriggerEditing {
                        HStack {
                            Text("Phrase is set (\(String(repeating: "\u{2022}", count: 12)))")
                            Spacer()
                            Button("Edit") {
                                Task { await loadCovertTriggerForEditing() }
                                covertTriggerEditing = true
                            }
                            Button("Clear", role: .destructive) {
                                Task { await clearCovertTrigger() }
                            }
                        }
                    } else {
                        SecureField("Covert trigger phrase", text: $covertTriggerPhraseDraft)
                        HStack {
                            Button("Save") { commitCovertTrigger() }
                                .disabled(covertTriggerPhraseDraft.trimmingCharacters(in: .whitespacesAndNewlines).count < CovertTriggerMonitor.minPhraseLength)
                            if covertTriggerIsSet {
                                Button("Cancel") {
                                    covertTriggerPhraseDraft = ""
                                    covertTriggerEditing = false
                                    loadCovertTrigger()
                                }
                            }
                        }
                    }
                    if !covertTriggerPhraseDraft.isEmpty &&
                        covertTriggerPhraseDraft.count < CovertTriggerMonitor.minPhraseLength {
                        Label(
                            "Phrase must be at least \(CovertTriggerMonitor.minPhraseLength) characters.",
                            systemImage: "exclamationmark.triangle"
                        )
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    }
                    Text("Choose something you can type naturally. Don't choose a famous quote. \(CovertTriggerMonitor.minPhraseLength) characters minimum.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    HStack {
                        Button("Test") { testCovertTrigger() }
                            .disabled(!canTestCovertTrigger)
                        if let note = covertTriggerTestNote {
                            Text(note).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    Text("Covert trigger - for use when you can't visibly destroy your data. External security audit pending.")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                    Toggle("Coercion mode", isOn: $coercionMode)
                    if coercionMode {
                        Text("Coercion mode: the visible 'Plea the Fifth' controls are hidden. Make sure you remember the hot-key and the covert trigger phrase.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            } header: {
                HStack {
                    Text("Plea the Fifth")
                    if coercionMode {
                        Circle()
                            .fill(.green)
                            .frame(width: 6, height: 6)
                            .accessibilityLabel("Covert trigger armed")
                    }
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
        }
        .navigationTitle("Settings")
        .onAppear { loadCovertTrigger() }
        .onChange(of: coercionMode) { _, newValue in
            if newValue {
                pleadTheFifthExpandedIOS = false
            } else {
                pleadTheFifthExpandedIOS = true
            }
        }
    }

    /// Whether the "Plea the Fifth" section is expanded. Set
    /// to false when coercion mode flips on (design 9.5).
    @State private var pleadTheFifthExpandedIOS: Bool = false

    /// Whether the Test button is enabled. Same logic as
    /// macOS: a non-empty draft of the minimum length is
    /// testable, otherwise the stored phrase must be set.
    private var canTestCovertTrigger: Bool {
        let draft = covertTriggerPhraseDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        if !draft.isEmpty {
            return draft.count >= CovertTriggerMonitor.minPhraseLength
        }
        return covertTriggerIsSet
    }

    private func loadCovertTrigger() {
        Task { @MainActor in
            let isSet = await CovertTriggerMonitor.shared.isArmed
            self.covertTriggerIsSet = isSet
            if coercionMode {
                self.pleadTheFifthExpandedIOS = false
            } else {
                self.pleadTheFifthExpandedIOS = isSet
            }
        }
    }

    private func loadCovertTriggerForEditing() async {
        let stored = TesseraSecretStore.secret(
            account: CovertTriggerMonitor.keychainAccount
        ) ?? ""
        await MainActor.run {
            self.covertTriggerPhraseDraft = stored
        }
    }

    private func commitCovertTrigger() {
        let draft = covertTriggerPhraseDraft
        Task {
            let ok = await CovertTriggerMonitor.shared.setPhrase(draft)
            await MainActor.run {
                if ok {
                    self.covertTriggerIsSet = !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    self.covertTriggerPhraseDraft = ""
                    self.covertTriggerEditing = false
                }
            }
        }
    }

    private func clearCovertTrigger() async {
        _ = await CovertTriggerMonitor.shared.setPhrase("")
        await MainActor.run {
            self.covertTriggerIsSet = false
            self.covertTriggerEditing = false
            self.covertTriggerPhraseDraft = ""
        }
    }

    private func testCovertTrigger() {
        let draft = covertTriggerPhraseDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        let stored = TesseraSecretStore.secret(
            account: CovertTriggerMonitor.keychainAccount
        ) ?? ""
        let phrase: String = draft.isEmpty ? stored : draft
        guard phrase.count >= CovertTriggerMonitor.minPhraseLength else {
            covertTriggerTestNote = "Phrase too short to test."
            return
        }
        let observed = "Test fire: today the user said '\(phrase)' in chat."
        Task {
            let didFire = await CovertTriggerMonitor.shared.testObserve(
                candidate: phrase, text: observed
            )
            await MainActor.run {
                if didFire {
                    self.covertTriggerTestNote = "The trigger would have fired. Make sure the phrase isn't likely to come up in your normal use."
                } else {
                    self.covertTriggerTestNote = "Trigger did not fire - the phrase didn't match the test text."
                }
            }
        }
    }
}
#endif
