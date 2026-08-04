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
            advancedTab
                .tabItem { Label("Advanced", systemImage: "slider.horizontal.3") }
        }
        .frame(width: 520, height: 420)
        .onAppear { loadAPIKey() }
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
        Section("Scoped YOLO") {
            if let yolo = yoloSession {
                LabeledContent("goal", value: yolo.goal ?? "-")
                LabeledContent("reason", value: yolo.reason.isEmpty ? "-" : yolo.reason)
                LabeledContent("expires", value: yolo.expiresAt.formatted())
                LabeledContent("actions so far", value: "\(yolo.actionCount)")
                Button("End YOLO") { endYolo() }
            } else {
                TextField("Goal (optional)", text: $yoloGoal)
                TextField("Reason (for the audit log)", text: $yoloReason)
                Stepper("Minutes: \(yoloMinutes)", value: $yoloMinutes, in: 5...240)
                Button("Start YOLO") { startYolo() }
                    .disabled(autonomySessionID.isEmpty)
                if autonomySessionID.isEmpty {
                    Text("Run the agent once first; YOLO binds to that session.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                if let yoloNote {
                    Text(yoloNote).font(.caption).foregroundStyle(.secondary)
                }
            }
            Text("YOLO auto-approves within scope, but irreversible actions always prompt. It expires on time and never persists across sessions.")
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
            yoloNote = "Last YOLO ran \(summary.actionCount) action(s) across \(summary.classes.count) class(es); \(summary.denials) denial(s)."
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

    private var advancedTab: some View {
        Form {
            Toggle("Enable telemetry", isOn: $telemetryEnabled)
            Picker("Log level", selection: $logLevel) {
                ForEach(TesseraLogLevel.allCases, id: \.rawValue) { level in
                    Text(level.rawValue.uppercased()).tag(level.rawValue)
                }
            }
            PathField("Custom CLI path", text: $cliPath, picks: .directory)
        }
        .formStyle(.grouped)
        .padding()
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
