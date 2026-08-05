import SwiftUI
import UniformTypeIdentifiers
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
    @AppStorage(TesseraSettingsKey.tesseraCLIPath) private var tesseraCLIPath = TesseraSettingsDefault.tesseraCLIPath
    // LLM provider
    @AppStorage(TesseraSettingsKey.llmProviderType) private var llmProviderType = TesseraSettingsDefault.llmProviderType
    @AppStorage(TesseraSettingsKey.remoteAPIBaseURL) private var remoteAPIBaseURL = TesseraSettingsDefault.remoteAPIBaseURL
    @AppStorage(TesseraSettingsKey.remoteModelName) private var remoteModelName = TesseraSettingsDefault.remoteModelName
    // The API key is NOT @AppStorage: it lives in the Keychain.
    // `apiKeyDraft` is only the in-flight editing copy; the UI
    // otherwise sees a stored/missing state, never the secret.
    @State private var apiKeyDraft = ""
    @State private var apiKeyState: TesseraSecretState = .missing
    @AppStorage(TesseraSettingsKey.remoteUseStreaming) private var remoteUseStreaming = TesseraSettingsDefault.remoteUseStreaming
    @AppStorage(TesseraSettingsKey.onDeviceModelPath) private var onDeviceModelPath = TesseraSettingsDefault.onDeviceModelPath
    @AppStorage(TesseraSettingsKey.onDeviceLibraryPath) private var onDeviceLibraryPath = TesseraSettingsDefault.onDeviceLibraryPath
    @AppStorage(TesseraSettingsKey.onDeviceContextLength) private var onDeviceContextLength = TesseraSettingsDefault.onDeviceContextLength
    @AppStorage(TesseraSettingsKey.onDeviceGPULayers) private var onDeviceGPULayers = TesseraSettingsDefault.onDeviceGPULayers
    // Learning (drafter training). Read by TesseraLearningServices at launch.
    @AppStorage(TesseraSettingsKey.learningBaseModelPath) private var learningBaseModelPath = TesseraSettingsDefault.learningBaseModelPath
    @AppStorage(TesseraSettingsKey.learningTrainBinary) private var learningTrainBinary = TesseraSettingsDefault.learningTrainBinary
    @AppStorage(TesseraSettingsKey.learningTrainingDryRun) private var learningTrainingDryRun = TesseraSettingsDefault.learningTrainingDryRun
    @AppStorage(TesseraSettingsKey.learningAutoTrain) private var learningAutoTrain = TesseraSettingsDefault.learningAutoTrain
    // Runtime speculative decoding + trace capture (runtime-traces spec 7, 10).
    @AppStorage(TesseraSettingsKey.learningRuntimeDraftModel) private var learningRuntimeDraftModel = TesseraSettingsDefault.learningRuntimeDraftModel
    @AppStorage(TesseraSettingsKey.learningRuntimeCapture) private var learningRuntimeCapture = TesseraSettingsDefault.learningRuntimeCapture
    @AppStorage(TesseraSettingsKey.learningRuntimeCaptureTopk) private var learningRuntimeCaptureTopk = TesseraSettingsDefault.learningRuntimeCaptureTopk
    @AppStorage(TesseraSettingsKey.learningRuntimeDraftMax) private var learningRuntimeDraftMax = TesseraSettingsDefault.learningRuntimeDraftMax
    // "Plea the Fifth" coercion-resistant destruction. The
    // phrase itself lives in the Keychain (see
    // ``CovertTriggerMonitor``); only the coercion-mode flag
    // and the local UI state live in UserDefaults / @State.
    @AppStorage(TesseraSettingsKey.coercionMode) private var coercionMode = TesseraSettingsDefault.coercionMode
    @State private var covertTriggerPhraseDraft = ""
    @State private var covertTriggerIsSet = false
    @State private var covertTriggerEditing = false
    @State private var covertTriggerTestNote: String?

    // Autonomy (autonomy-calibration-design.md 13): snapshots of the learned-
    // permission store, refreshed on appear and after every mutation.
    @State private var autonomyEntries: [TesseraLearnedPermission] = []
    @State private var autonomyRecommendations: [TesseraRecommendation] = []
    @State private var autonomyConfig = TesseraPermissionConfig()
    @State private var autonomySessionID = ""
    @State private var yoloSession: TesseraYoloSession?
    @State private var yoloGoal = ""
    @State private var yoloReason = ""
    @State private var yoloMinutes = 30
    @State private var yoloNote: String?
    @State private var netWarm = false
    @State private var netNote: String?
    @State private var confirmResetGrants = false
    @State private var confirmPurge = false

    var body: some View {
        TabView {
            generalTab
                .tabItem { Label("General", systemImage: "gearshape") }
            agentTab
                .tabItem { Label("Agent", systemImage: "cpu") }
            modelTab
                .tabItem { Label("Model", systemImage: "brain") }
            autonomyTab
                // Was "Autonomy" (HIG 2.14): the tab configures what
                // the agent is ALLOWED to do on its own - permissions
                // is the word users scan for.
                .tabItem { Label("Permissions", systemImage: "hand.raised") }
            // "Plea the Fifth" tab. Collapsed by default when
            // coercion mode is on so the visible state matches
            // the "an adversary wouldn't notice it" threat
            // model (design section 9.5).
            pleadTheFifthTab
                .tabItem { Label("Plea the Fifth", systemImage: "lock.shield") }
            advancedTab
                .tabItem { Label("Advanced", systemImage: "slider.horizontal.3") }
        }
        .frame(width: 520, height: 460)
        .onAppear {
            loadAPIKey()
            loadCovertTrigger()
        }
        .onChange(of: coercionMode) { _, newValue in
            // When coercion mode flips ON, collapse the
            // "Plea the Fifth" section by default so an
            // adversary wouldn't notice it. When it flips
            // OFF, expand it so the user can confirm the
            // trigger is still configured.
            if newValue {
                pleadTheFifthExpanded = false
            } else {
                pleadTheFifthExpanded = true
            }
        }
    }

    private var generalTab: some View {
        Form {
            Picker("Default runtime", selection: $defaultRuntime) {
                ForEach(TesseraRuntime.allCases, id: \.rawValue) { rt in
                    Text(rt.displayName).tag(rt.rawValue)
                }
            }
            PathField("Model directory", text: $modelDirectory, picks: .directory)
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

    private var modelTab: some View {
        Form {
            Picker("LLM provider", selection: $llmProviderType) {
                ForEach(TesseraLLMProviderType.allCases, id: \.rawValue) { type in
                    Text(type.displayName).tag(type.rawValue)
                }
            }
            .pickerStyle(.inline)

            if llmProviderType == TesseraLLMProviderType.remoteAPI.rawValue {
                Section("Remote API") {
                    TextField("Base URL", text: $remoteAPIBaseURL)
                    SecureField("API key", text: $apiKeyDraft)
                        .onSubmit { commitAPIKey() }
                        .onDisappear { commitAPIKey() }
                        .accessibilityHint("Stored in the macOS Keychain, not preferences")
                    apiKeyStateRow
                    TextField("Model name", text: $remoteModelName)
                    Toggle("Stream responses (SSE)", isOn: $remoteUseStreaming)
                }
            }

            if llmProviderType == TesseraLLMProviderType.onDevice.rawValue {
                Section("On-Device (llama.cpp)") {
                    PathField("GGUF model path", text: $onDeviceModelPath,
                              picks: .file(types: [.init(filenameExtension: "gguf")].compactMap { $0 }))
                    PathField("libllama.dylib path (optional)", text: $onDeviceLibraryPath,
                              picks: .file(types: [UTType(filenameExtension: "dylib")].compactMap { $0 }))
                    TextField("Context length", value: $onDeviceContextLength, format: .number)
                    Stepper("GPU layers: \(onDeviceGPULayers < 0 ? "all" : "\(onDeviceGPULayers)")",
                            value: $onDeviceGPULayers, in: -1...200)
                }
            }

            Section("Learning (drafter training)") {
                PathField("Drafter model (GGUF)", text: $learningBaseModelPath,
                          picks: .file(types: [.init(filenameExtension: "gguf")].compactMap { $0 }))
                PathField("Training driver (tessera-train-lk)", text: $learningTrainBinary,
                          picks: .file(types: [.unixExecutable]))
                Toggle("Train automatically when idle", isOn: $learningAutoTrain)
                    .help("Runs a training cycle on the idle schedule when enough traces have accumulated")
                Toggle("Dry run (build the dataset only)", isOn: $learningTrainingDryRun)
                    .help("Idle cycles validate the dataset without training or saving a drafter; the dashboard's Train Drafter button always trains")
                trainBinaryStateRow
                PathField("Runtime drafter (GGUF)", text: $learningRuntimeDraftModel,
                          picks: .file(types: [.init(filenameExtension: "gguf")].compactMap { $0 }))
                runtimeDrafterStateRow
                Toggle("Capture runtime traces", isOn: $learningRuntimeCapture)
                    .help("Records speculative-decoding telemetry while you use the Playground; sessions are curated locally before any training use")
                Stepper("Capture top-k: \(learningRuntimeCaptureTopk)",
                        value: $learningRuntimeCaptureTopk, in: 1...128)
                    .help("Depth of the per-position verifier/drafter distributions captured per spec step; the replay stage deepens promoted sessions offline")
                Stepper("Draft depth: \(learningRuntimeDraftMax)",
                        value: $learningRuntimeDraftMax, in: 1...8)
                    .help("Maximum tokens the runtime drafter proposes per speculative step")
                Text("Auto-train applies immediately. Training paths are read when the app launches; the runtime drafter is read when the Playground provider initializes.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Text("Changes apply the next time the Playground is opened.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .formStyle(.grouped)
        .padding()
    }

    /// The Keychain state line under the API key field. Pairs a
    /// symbol with the text so the state is not color-only.
    @ViewBuilder
    private var apiKeyStateRow: some View {
        switch apiKeyState {
        case .stored:
            Label("Key stored in the Keychain", systemImage: "lock.shield")
                .font(.caption)
                .foregroundStyle(.secondary)
        case .missing:
            Label("No key stored yet", systemImage: "key.slash")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    /// Live resolution status of the training driver. Pairs a symbol
    /// with the text so the state is not color-only.
    @ViewBuilder
    private var trainBinaryStateRow: some View {
        let resolved = TesseraTrainBinaryResolver.resolve(override: learningTrainBinary)
        if FileManager.default.isExecutableFile(atPath: resolved) {
            Label("Driver found at \(resolved)", systemImage: "checkmark.circle")
                .font(.caption)
                .foregroundStyle(.secondary)
        } else {
            Label("Driver not found; expected at \(resolved)", systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    /// Live resolution status of the runtime drafter (runtime-traces spec
    /// section 10): the same found/not-found pattern as the training driver,
    /// showing the auto-derived value when the field is empty. Pairs a
    /// symbol with the text so the state is not color-only.
    @ViewBuilder
    private var runtimeDrafterStateRow: some View {
        let setting = learningRuntimeDraftModel.trimmingCharacters(in: .whitespacesAndNewlines)
        if setting == TesseraRuntimeDrafterResolver.disableSentinel {
            Label("Auto-derive is off; the Playground runs trunk-only", systemImage: "minus.circle")
                .font(.caption)
                .foregroundStyle(.secondary)
        } else if let candidate = TesseraRuntimeDrafterResolver.resolve(
            setting: learningRuntimeDraftModel, trunkPath: learningBaseModelPath) {
            let exists = FileManager.default.fileExists(atPath: candidate)
            if exists {
                Label("Runtime drafter found at \(candidate)", systemImage: "checkmark.circle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if setting.isEmpty {
                Label("No trained drafter yet; will use \(candidate) once training produces it", systemImage: "clock")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Label("Runtime drafter not found at \(candidate); the Playground runs trunk-only", systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        } else {
            Label("Set a drafter model above to derive the runtime drafter from it", systemImage: "minus.circle")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private func loadAPIKey() {
        apiKeyDraft = TesseraSecretStore.secret(
            account: TesseraSecretStore.remoteAPIKeyAccount
        ) ?? ""
        apiKeyState = TesseraSecretStore.state(
            account: TesseraSecretStore.remoteAPIKeyAccount
        )
    }

    /// Write the editing copy to the Keychain. An empty field
    /// deletes the stored secret. Runs on submit and when the
    /// field disappears (tab switch / provider change / window
    /// close), so there is no explicit "Save key" button to
    /// forget.
    private func commitAPIKey() {
        let stored = TesseraSecretStore.setSecret(
            apiKeyDraft.isEmpty ? nil : apiKeyDraft,
            account: TesseraSecretStore.remoteAPIKeyAccount
        )
        if stored {
            apiKeyState = TesseraSecretStore.state(
                account: TesseraSecretStore.remoteAPIKeyAccount
            )
        }
    }

    // MARK: Autonomy tab (autonomy-calibration-design.md 9, 10, 11, 13)

    private var autonomyTab: some View {
        ScrollView {
            Form {
                autonomyDispositionSection
                autonomyYoloSection
                autonomyRecommendationsSection
                autonomyEntriesSection
                autonomyNetworkSection
                autonomyGlobalSection
            }
            .formStyle(.grouped)
        }
        .onAppear { loadAutonomy() }
        .onChange(of: autonomyConfig) { _, newValue in
            TesseraLearningCenter.shared.autonomy.updateConfig { $0 = newValue }
        }
    }

    private var autonomyDispositionSection: some View {
        Section("Disposition (floor and ceiling)") {
            Picker("Floor (minimum requirement)", selection: $autonomyConfig.floor) {
                Text("Restricted (never auto-approve)").tag(TesseraPermissionProfile.restricted)
                Text("Standard").tag(TesseraPermissionProfile.standard)
                Text("Elevated").tag(TesseraPermissionProfile.elevated)
            }
            Picker("Ceiling (maximum learning may reach)", selection: $autonomyConfig.ceiling) {
                Text("Contained low-risk only").tag(AutonomyCeiling.containedLowRiskOnly)
                Text("Any non-irreversible class").tag(AutonomyCeiling.anyNonIrreversible)
            }
            Stepper("Approvals needed to grant: \(autonomyConfig.grantThresholdN)",
                    value: $autonomyConfig.grantThresholdN, in: 1...20)
            Stepper("Distinct sessions needed: \(autonomyConfig.sessionThresholdM)",
                    value: $autonomyConfig.sessionThresholdM, in: 1...10)
            Stepper("Path-glob depth: \(autonomyConfig.pathGlobDepth)",
                    value: $autonomyConfig.pathGlobDepth, in: 1...4)
        }
    }

    private var autonomyYoloSection: some View {
        Section("Autonomous session") {
            if let yolo = yoloSession {
                LabeledContent("goal", value: yolo.goal ?? "-")
                LabeledContent("reason", value: yolo.reason.isEmpty ? "-" : yolo.reason)
                LabeledContent("expires", value: yolo.expiresAt.formatted())
                LabeledContent("actions so far", value: "\(yolo.actionCount)")
                Button("End autonomous session") { endYolo() }
            } else {
                TextField("Goal (optional)", text: $yoloGoal)
                TextField("Reason (for the audit log)", text: $yoloReason)
                Stepper("Minutes: \(yoloMinutes)", value: $yoloMinutes, in: 5...240)
                Button("Start autonomous session") { startYolo() }
                    .disabled(autonomySessionID.isEmpty)
                if autonomySessionID.isEmpty {
                    Text("Run the agent once first; the autonomous session binds to that run.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                if let yoloNote {
                    Text(yoloNote).font(.caption).foregroundStyle(.secondary)
                }
            }
            Text("An autonomous session auto-approves within scope, but irreversible actions always prompt. It expires on time and never persists across sessions.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var autonomyRecommendationsSection: some View {
        Section("Recommendations") {
            if autonomyRecommendations.isEmpty {
                Text("Nothing to recommend yet.").foregroundStyle(.secondary)
            } else {
                ForEach(autonomyRecommendations) { rec in
                    VStack(alignment: .leading, spacing: 6) {
                        Text(rec.message)
                        HStack {
                            Button("Confirm") { confirmRecommendation(rec.actionClass, .confirm) }
                            Button("Not now") { confirmRecommendation(rec.actionClass, .notNow) }
                            Button("Never") { confirmRecommendation(rec.actionClass, .never) }
                        }
                    }
                    .padding(.vertical, 2)
                }
            }
        }
    }

    private var autonomyEntriesSection: some View {
        Section("Learned permissions") {
            if autonomyEntries.isEmpty {
                Text("Tessera starts needy. As you approve actions, classes earn autonomy here.")
                    .foregroundStyle(.secondary)
            } else {
                ForEach(autonomyEntries) { entry in
                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 6) {
                            Text(entry.actionClass).font(.system(.body, design: .monospaced))
                            autonomyBadges(for: entry)
                        }
                        Text("\(entry.totalApprovals) approved / \(entry.totalDenials) denied / \(entry.distinctSessions) session(s)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        HStack {
                            if entry.granted {
                                Button("Revoke") { revokeEntry(entry.actionClass) }
                            }
                            if entry.revoked {
                                Button("Un-revoke") { unrevokeEntry(entry.actionClass) }
                            }
                            if !entry.irreversible {
                                Button("Add to denylist") { denylistEntry(entry.actionClass) }
                            }
                        }
                    }
                    .padding(.vertical, 2)
                }
            }
        }
    }

    @ViewBuilder
    private func autonomyBadges(for entry: TesseraLearnedPermission) -> some View {
        // Each badge pairs its state color with a symbol so the
        // state is not encoded by color alone.
        if entry.irreversible {
            Label("irreversible", systemImage: "exclamationmark.octagon")
                .font(.caption2).padding(.horizontal, 4).background(.red.opacity(0.15))
        } else if entry.granted {
            Label("granted", systemImage: "checkmark.circle")
                .font(.caption2).padding(.horizontal, 4).background(.green.opacity(0.15))
        }
        if entry.revoked {
            Label("revoked", systemImage: "slash.circle")
                .font(.caption2).padding(.horizontal, 4).background(.orange.opacity(0.15))
        }
    }

    private var autonomyNetworkSection: some View {
        Section("Approver network") {
            LabeledContent("status", value: netWarm ? "warm (modulating grants)" : "cold (rule-based only)")
            Button("Train now") { trainApproverNow() }
            if let netNote {
                Text(netNote).font(.caption).foregroundStyle(.secondary)
            }
            Text("A small local network trained on your approval receipts during idle time. It predicts; it never grants. Fails closed.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var autonomyGlobalSection: some View {
        Section {
            Button("Reset all grants") {
                confirmResetGrants = true
            }
            Button("Purge all learning data", role: .destructive) {
                confirmPurge = true
            }
        }
        // HIG 14.1 / 13.5: both actions fire only after an explicit
        // confirmation; the destructive button is never the default.
        .confirmationDialog("Reset all grants?", isPresented: $confirmResetGrants,
                            titleVisibility: .visible) {
            Button("Reset all grants", role: .destructive) {
                TesseraLearningCenter.shared.autonomy.resetAll()
                loadAutonomy()
            }
        } message: {
            Text("All learned permission grants will be removed; Tessera will ask again before running those actions.")
        }
        .confirmationDialog("Purge all learning data?", isPresented: $confirmPurge,
                            titleVisibility: .visible) {
            Button("Purge all learning data", role: .destructive) {
                _ = try? TesseraLearningCenter.shared.purgeAll()
                loadAutonomy()
            }
        } message: {
            Text("All stored learning records will be permanently deleted. This cannot be undone.")
        }
    }

    private func loadAutonomy() {
        let autonomy = TesseraLearningCenter.shared.autonomy
        autonomyEntries = autonomy.entries()
        autonomyRecommendations = autonomy.recommendations()
        autonomyConfig = autonomy.config
        autonomySessionID = autonomy.activeSessionID
        yoloSession = autonomy.activeYolo(for: nil)
        netWarm = autonomy.isNetWarm
        if yoloMinutes == 30 { yoloMinutes = autonomy.config.yoloDefaultMinutes }
    }

    private func startYolo() {
        let autonomy = TesseraLearningCenter.shared.autonomy
        yoloSession = autonomy.startYolo(
            goal: yoloGoal.isEmpty ? nil : yoloGoal,
            sessionID: autonomySessionID,
            reason: yoloReason,
            minutes: yoloMinutes
        )
        yoloNote = nil
    }

    private func endYolo() {
        if let summary = TesseraLearningCenter.shared.autonomy.endYolo() {
            yoloNote = "Last autonomous session ran \(summary.actionCount) action(s) across \(summary.classes.count) class(es); \(summary.denials) denial(s)."
        }
        yoloSession = nil
    }

    private func confirmRecommendation(_ actionClass: String, _ choice: TesseraRecommendationChoice) {
        let sessionID = autonomySessionID.isEmpty ? "settings-ui" : autonomySessionID
        _ = TesseraLearningCenter.shared.autonomy.confirmRecommendation(
            actionClass: actionClass, choice: choice, sessionID: sessionID
        )
        loadAutonomy()
    }

    private func revokeEntry(_ actionClass: String) {
        TesseraLearningCenter.shared.autonomy.revoke(actionClass)
        loadAutonomy()
    }

    private func unrevokeEntry(_ actionClass: String) {
        TesseraLearningCenter.shared.autonomy.unrevoke(actionClass)
        loadAutonomy()
    }

    private func denylistEntry(_ actionClass: String) {
        TesseraLearningCenter.shared.autonomy.denylist(actionClass)
        loadAutonomy()
    }

    private func trainApproverNow() {
        let autonomy = TesseraLearningCenter.shared.autonomy
        let passed = autonomy.trainApprover(denialWeight: 5.0)
        netWarm = autonomy.isNetWarm
        if passed {
            netNote = "Trained; the calibration guard passed."
        } else if netWarm {
            netNote = "Retrained net failed the calibration guard; rolled back to the previous weights."
        } else {
            netNote = "Not trained yet: needs at least \(TesseraAutonomyService.warmupThreshold) approval receipts, or the guard rolled back."
        }
    }

    // MARK: Plea the Fifth tab (docs/tessera-plead-the-fifth-design.md 8.3, 9)

    /// "Plea the Fifth" tab. The covert trigger phrase, the
    /// test button, and the coercion-mode toggle. Collapsed
    /// by default when coercion mode is on (design section 9.5).
    private var pleadTheFifthTab: some View {
        // Coercion mode ON: the section is collapsed (no
        // icon, just a single-line section header) so an
        // adversary who glances at the Settings window
        // wouldn't notice the trigger controls (design 9.5).
        // The user can expand the section manually.
        DisclosureGroup(isExpanded: $pleadTheFifthExpanded) {
            pleadTheFifthSection
        } label: {
            HStack {
                Text("Plea the Fifth")
                if coercionMode {
                    // Subtle indicator - a small "armed" dot,
                    // visible only to the user who knows what
                    // it means.
                    Circle()
                        .fill(.green)
                        .frame(width: 6, height: 6)
                        .accessibilityLabel("Covert trigger armed")
                }
            }
        }
    }

    /// Tracks whether the "Plea the Fifth" section is expanded
    /// in the Settings UI. Defaults to false when coercion mode
    /// is on (design 9.5: "a user who is being watched wouldn't
    /// notice it"), true otherwise. Updated via onChange when
    /// the coercion mode flag flips.
    @State private var pleadTheFifthExpanded: Bool = false

    @ViewBuilder
    private var pleadTheFifthSection: some View {
        Form {
            Section("Covert trigger phrase") {
                covertTriggerField
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
            }
            Section {
                Toggle("Coercion mode", isOn: $coercionMode)
                    .help("Hide the visible 'Plea the Fifth' menu item. The hot-key and covert trigger still work.")
                if coercionMode {
                    Label(
                        "Coercion mode: the visible 'Plea the Fifth' controls are hidden. Make sure you remember the hot-key and the covert trigger phrase.",
                        systemImage: "eye.slash"
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
            }
        }
        .formStyle(.grouped)
    }

    /// The phrase field. SecureField so the user's typing
    /// isn't visible; once set, the field is masked to a row
    /// of bullets with an "Edit" button to reveal it. The
    /// draft is the local editing copy; the stored phrase
    /// lives in the Keychain.
    @ViewBuilder
    private var covertTriggerField: some View {
        if covertTriggerIsSet && !covertTriggerEditing {
            HStack {
                Text("Phrase is set (\(String(repeating: "\u{2022}", count: 12)))")
                Spacer()
                Button("Edit") {
                    // When the user clicks Edit, we read the
                    // stored phrase into the draft so they can
                    // continue editing. After Edit, the field
                    // is a SecureField bound to the draft.
                    Task { await loadCovertTriggerForEditing() }
                    covertTriggerEditing = true
                }
                Button("Clear", role: .destructive) {
                    Task { await clearCovertTrigger() }
                }
            }
        } else {
            SecureField("Covert trigger phrase", text: $covertTriggerPhraseDraft)
                .onSubmit { commitCovertTrigger() }
                .onDisappear { commitCovertTrigger() }
            HStack {
                Button("Save") { commitCovertTrigger() }
                    .disabled(covertTriggerPhraseDraft.trimmingCharacters(in: .whitespacesAndNewlines).count < CovertTriggerMonitor.minPhraseLength)
                if covertTriggerIsSet {
                    Button("Cancel") {
                        // Exit edit mode without saving.
                        covertTriggerPhraseDraft = ""
                        covertTriggerEditing = false
                        loadCovertTrigger()
                    }
                }
            }
        }
    }

    /// Whether the Test button is enabled. We test against the
    /// saved phrase, not the draft, so a half-typed phrase
    /// can't be tested. A draft phrase is also testable if it
    /// meets the length minimum.
    private var canTestCovertTrigger: Bool {
        let draft = covertTriggerPhraseDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        if !draft.isEmpty {
            return draft.count >= CovertTriggerMonitor.minPhraseLength
        }
        return covertTriggerIsSet
    }

    /// Initial load: ask the monitor if a phrase is set (so
    /// the field renders the masked "is set" state). The
    /// actual phrase stays in the Keychain; we only read the
    /// presence flag.
    private func loadCovertTrigger() {
        Task { @MainActor in
            let isSet = await CovertTriggerMonitor.shared.isArmed
            self.covertTriggerIsSet = isSet
            // Collapse the section by default when coercion
            // mode is on (design 9.5). On a fresh load, the
            // user hasn't expanded it yet; flipping the flag
            // later goes through the onChange handler.
            if coercionMode {
                self.pleadTheFifthExpanded = false
            } else {
                self.pleadTheFifthExpanded = isSet
            }
        }
    }

    /// Load the stored phrase into the draft for editing.
    /// This reads the phrase from the Keychain into the
    /// local @State; the phrase is in-memory only while the
    /// edit dialog is open. SecureField keeps the screen
    /// pixels safe.
    private func loadCovertTriggerForEditing() async {
        let stored = TesseraSecretStore.secret(
            account: CovertTriggerMonitor.keychainAccount
        ) ?? ""
        await MainActor.run {
            self.covertTriggerPhraseDraft = stored
        }
    }

    /// Commit the draft to the Keychain via the monitor.
    /// Empty draft clears; short drafts are rejected.
    private func commitCovertTrigger() {
        let draft = covertTriggerPhraseDraft
        Task {
            let ok = await CovertTriggerMonitor.shared.setPhrase(draft)
            await MainActor.run {
                if ok {
                    self.covertTriggerIsSet = !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    self.covertTriggerPhraseDraft = ""
                    self.covertTriggerEditing = false
                    if let note = self.covertTriggerTestNote,
                       note.contains("would have fired") {
                        self.covertTriggerTestNote = nil
                    }
                }
            }
        }
    }

    /// Clear the stored phrase. Disables the monitor.
    private func clearCovertTrigger() async {
        _ = await CovertTriggerMonitor.shared.setPhrase("")
        await MainActor.run {
            self.covertTriggerIsSet = false
            self.covertTriggerEditing = false
            self.covertTriggerPhraseDraft = ""
        }
    }

    /// Simulate a fire. We invoke the monitor's `testObserve`
    /// with a synthetic text containing the phrase (draft or
    /// stored). `testObserve` runs the same matching rules as
    /// `observe` but doesn't fire the callback - the dev
    /// preview leaves the real wipe executor unwired; the
    /// test button just confirms the matching logic works
    /// for the user's phrase.
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

    private var advancedTab: some View {
        Form {
            Toggle("Enable telemetry", isOn: $telemetryEnabled)
            Picker("Log level", selection: $logLevel) {
                ForEach(TesseraLogLevel.allCases, id: \.rawValue) { level in
                    Text(level.rawValue.uppercased()).tag(level.rawValue)
                }
            }
            PathField("Custom CLI path", text: $cliPath, picks: .directory)
            Section("Engine binary (tessera-cli)") {
                PathField("tessera-cli path", text: $tesseraCLIPath, picks: .file(types: [.unixExecutable]))
                tesseraCLIStateRow
                Text("Leave empty to auto-resolve; the known install locations and $PATH are checked in order.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding()
    }

    /// Live resolution status of the tessera-cli binary. Pairs a symbol
    /// with the text so the state is not encoded by color alone. Re-runs
    /// on every `tesseraCLIPath` edit, so the user sees the result of a
    /// paste the moment the field is updated.
    @ViewBuilder
    private var tesseraCLIStateRow: some View {
        switch TesseraCLIBinaryResolver.resolvedPathOrDiagnostic(
            override: tesseraCLIPath,
            settingsKey: TesseraSettingsKey.tesseraCLIPath
        ) {
        case .found(let path):
            Label("Found at \(path)", systemImage: "checkmark.circle")
                .font(.caption)
                .foregroundStyle(.secondary)
        case .notFound(let searched):
            let summary = searched.isEmpty
                ? "not found"
                : "Not found; checked: " + searched.prefix(3).joined(separator: ", ")
            Label(summary, systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}

/// A path setting with a Browse... button. Free typing still works
/// (power users paste paths), but HIG 2.13 asks that path fields
/// also offer a real file / folder picker instead of making the
/// user hand-type an absolute path. The NSOpenPanel resolves
/// security-scoped access for us on selection.
private struct PathField: View {
    enum PickTarget {
        case file(types: [UTType])
        case directory
    }

    let label: String
    @Binding var text: String
    let picks: PickTarget

    init(_ label: String, text: Binding<String>, picks: PickTarget) {
        self.label = label
        self._text = text
        self.picks = picks
    }

    var body: some View {
        LabeledContent(label) {
            HStack(spacing: 6) {
                TextField(label, text: $text)
                Button("Browse…") { browse() }
                    .accessibilityLabel("Browse for \(label)")
            }
        }
    }

    private func browse() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        switch picks {
        case .file(let types):
            panel.canChooseFiles = true
            if !types.isEmpty { panel.allowedContentTypes = types }
        case .directory:
            panel.canChooseDirectories = true
        }
        // Start the panel where the current value points, when it
        // names an existing path - saves re-navigating from $HOME.
        if !text.isEmpty {
            let expanded = (text as NSString).expandingTildeInPath
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: expanded, isDirectory: &isDir) {
                panel.directoryURL = URL(fileURLWithPath: isDir.boolValue
                    ? expanded
                    : (expanded as NSString).deletingLastPathComponent)
            }
        }
        if panel.runModal() == .OK, let url = panel.url {
            text = url.path
        }
    }
}
