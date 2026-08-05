import Foundation

/// UserDefaults keys for all settings. Shared by the macOS and iOS settings
/// views and by non-view code that reads settings (e.g. building the agent loop).
public enum TesseraSettingsKey {
    // General
    public static let defaultRuntime = "tessera.settings.defaultRuntime"
    public static let modelDirectory = "tessera.settings.modelDirectory"
    public static let threadCount = "tessera.settings.threadCount"
    // Agent
    public static let maxIterations = "tessera.settings.maxIterations"
    public static let defaultApprovalLevel = "tessera.settings.defaultApprovalLevel"
    public static let tokenBudget = "tessera.settings.tokenBudget"
    // Advanced
    public static let telemetryEnabled = "tessera.settings.telemetryEnabled"
    public static let logLevel = "tessera.settings.logLevel"
    public static let cliPath = "tessera.settings.cliPath"
    /// User-chosen absolute path to the `tessera-cli` binary. Empty (the
    /// default) means auto-resolve via ``TesseraCLIBinaryResolver``; the
    /// known install locations and `$PATH` are walked in order.
    public static let tesseraCLIPath = "tessera.settings.tesseraCLIPath"
    /// Absolute path to the Python interpreter used by
    /// ``PythonEngineBridge`` to invoke tools under `tools/tessera/`. Empty
    /// means auto-resolve via `TESSERA_PYTHON` env var, then `which python3`,
    /// then the well-known Homebrew/system install locations.
    public static let tesseraPythonPath = "tessera.settings.tesseraPythonPath"
    // LLM provider
    public static let llmProviderType = "tessera.settings.llmProviderType"
    public static let remoteAPIBaseURL = "tessera.settings.remoteAPIBaseURL"
    public static let remoteAPIKey = "tessera.settings.remoteAPIKey"
    public static let remoteModelName = "tessera.settings.remoteModelName"
    public static let remoteUseStreaming = "tessera.settings.remoteUseStreaming"
    public static let onDeviceModelPath = "tessera.settings.onDeviceModelPath"
    public static let onDeviceLibraryPath = "tessera.settings.onDeviceLibraryPath"
    public static let onDeviceContextLength = "tessera.settings.onDeviceContextLength"
    public static let onDeviceGPULayers = "tessera.settings.onDeviceGPULayers"
    // First-run
    public static let onboardingComplete = "tessera.settings.onboardingComplete"
    // Audience mode (studio-ux blueprint section 3.1). Simple / Standard / Studio.
    public static let audienceMode = "tessera.settings.audienceMode"
    // Learning (self-improving loop)
    public static let learningEnabled = "tessera.settings.learningEnabled"
    public static let learningEscalationEnabled = "tessera.settings.learningEscalationEnabled"
    public static let learningTeachers = "tessera.settings.learningTeachers"
    public static let learningAnonymizerAggressiveness = "tessera.settings.learningAnonymizerAggressiveness"
    public static let learningAnonymizerBinary = "tessera.settings.learningAnonymizerBinary"
    public static let learningCaptureScopes = "tessera.settings.learningCaptureScopes"
    public static let learningIdleAdaptation = "tessera.settings.learningIdleAdaptation"
    public static let learningOnPowerOnly = "tessera.settings.learningOnPowerOnly"
    public static let learningDataRetentionDays = "tessera.settings.learningDataRetentionDays"
    public static let learningReferenceTTLDays = "tessera.settings.learningReferenceTTLDays"
    public static let learningMaxConcurrentAgents = "tessera.settings.learningMaxConcurrentAgents"
    public static let learningGuardEpsilon = "tessera.settings.learningGuardEpsilon"
    public static let learningAssessmentIntervalHours = "tessera.settings.learningAssessmentIntervalHours"
    public static let learningBaseModelPath = "tessera.settings.learningBaseModelPath"
    public static let learningMinTracesForTraining = "tessera.settings.learningMinTracesForTraining"
    public static let learningTrainingDryRun = "tessera.settings.learningTrainingDryRun"
    public static let learningTrainBinary = "tessera.settings.learningTrainBinary"
    public static let learningAutoTrain = "tessera.settings.learningAutoTrain"
    public static let learningTrainingIntervalHours = "tessera.settings.learningTrainingIntervalHours"
    // Session replay / curation stage (runtime-traces spec section 12)
    public static let learningSessionCuration = "tessera.settings.learningSessionCuration"
    // Runtime speculative decoding + trace capture (runtime-traces spec section 7)
    public static let learningRuntimeDraftModel = "tessera.settings.learningRuntimeDraftModel"
    public static let learningRuntimeCapture = "tessera.settings.learningRuntimeCapture"
    public static let learningRuntimeCaptureTopk = "tessera.settings.learningRuntimeCaptureTopk"
    public static let learningRuntimeDraftMax = "tessera.settings.learningRuntimeDraftMax"
    // Web search (agent research tool)
    public static let searchProvider = "tessera.settings.searchProvider"
    public static let searxngBaseURL = "tessera.settings.searxngBaseURL"
    public static let tavilyAPIKey = "tessera.settings.tavilyAPIKey"
    // "Plea the Fifth" (coercion-resistant destruction). The
    // phrase itself lives in the Keychain (see
    // ``TesseraSecretStore`` account ``covertTriggerPhrase``);
    // only the coercion-mode flag lives in UserDefaults.
    public static let coercionMode = "tessera.settings.coercionMode"
}

/// Factory defaults, registered at app launch.
public enum TesseraSettingsDefault {
    public static let modelDirectory = "~/Models/tessera"
    public static let threadCount = 0          // 0 == all cores
    public static let maxIterations = 10
    public static let tokenBudget = 8192
    public static let telemetryEnabled = false
    public static let logLevel = "info"
    public static let cliPath = "/usr/local/bin"
    /// Empty means auto-resolve via ``TesseraCLIBinaryResolver``. The
    /// legacy `cliPath` directory setting still exists for the older
    /// subprocess layout; `tesseraCLIPath` is the modern full-binary
    /// override and is what the retargeted engine tools read.
    public static let tesseraCLIPath = ""
    /// Empty means auto-resolve via ``PythonEngineBridge.discoverPython()``,
    /// which honors the `TESSERA_PYTHON` env var and walks the standard
    /// install locations. Used by every tool wrapped by the Python engine
    /// bridge.
    public static let tesseraPythonPath = ""
    // LLM provider. Default stays .placeholder for backward compatibility.
    public static let llmProviderType = "placeholder"
    public static let remoteAPIBaseURL = "http://localhost:8080/v1"
    public static let remoteAPIKey = ""
    public static let remoteModelName = "gpt-4"
    public static let remoteUseStreaming = true
    public static let onDeviceModelPath = ""
    public static let onDeviceLibraryPath = ""   // empty -> default dlopen search
    public static let onDeviceContextLength = 4096
    public static let onDeviceGPULayers = -1      // -1 == offload all layers
    // Learning (self-improving loop). Everything is opt-in and egress is
    // off by default; the teacher pool is empty until the user adds keys.
    public static let learningEnabled = false
    public static let learningEscalationEnabled = false
    public static let learningTeachers = ""            // JSON array of TesseraTeacherConfig
    public static let learningAnonymizerAggressiveness = "balanced"  // light | balanced | aggressive
    public static let learningAnonymizerBinary = ""          // empty -> /usr/local/bin/llama-quantize
    public static let learningCaptureScopes = "build,test,git"       // editor,screen off by default
    public static let learningIdleAdaptation = false
    public static let learningOnPowerOnly = true
    public static let learningDataRetentionDays = 90
    public static let learningReferenceTTLDays = 30
    public static let learningMaxConcurrentAgents = 4
    public static let learningGuardEpsilon = 0.02      // collapse-guard regression tolerance
    public static let learningAssessmentIntervalHours = 24
    public static let learningBaseModelPath = ""           // empty -> training disabled
    public static let learningMinTracesForTraining = 1000
    public static let learningTrainingDryRun = true
    public static let learningTrainBinary = ""        // empty -> TesseraTrainBinaryResolver auto-resolves
    // Idle drafter training. On by default (the architect-chosen trigger);
    // the orchestrator's gates (traces, model path, driver binary) and the
    // dry-run default keep it harmless until the flywheel is supplied.
    public static let learningAutoTrain = true
    public static let learningTrainingIntervalHours = 24
    // Idle session curation sweep (analysis + verdicts + replay). On by
    // default like auto-train; the stage's own gates (trunk model, captured
    // sessions) keep it a no-op until the flywheel is supplied.
    public static let learningSessionCuration = true
    // Runtime speculative decoding + trace capture. The drafter path is read
    // when the Playground provider initializes: empty auto-derives
    // "<base>-tessera-trained.gguf" next to the base model, the sentinel "-"
    // disables the auto-derive, and an explicit path always wins.
    public static let learningRuntimeDraftModel = ""
    public static let learningRuntimeCapture = true
    public static let learningRuntimeCaptureTopk = 16
    public static let learningRuntimeDraftMax = 3
    // Web search. Keyless DuckDuckGo is the default; SearXNG (self-hosted) and
    // Tavily (vendor key) are explicit opt-ins that stay off until configured.
    public static let searchProvider = "duckduckgo"
    public static let searxngBaseURL = ""
    public static let tavilyAPIKey = ""
    // "Plea the Fifth" coercion mode. Off by default; the
    // Settings view can flip it; Phase 2's menu bar item also
    // flips it. When on, the "Plea the Fifth" Settings section
    // is collapsed by default.
    public static let coercionMode = false
}

/// Log levels offered in Advanced settings.
public enum TesseraLogLevel: String, CaseIterable, Sendable {
    case debug, info, warning, error
}

/// The audience mode axis (studio-ux blueprint section 3.1). Three depths of
/// the same four-destination shell, switchable from the toolbar. Persisted in
/// UserDefaults so it survives across launches. Simple is the calm default.
public enum AudienceMode: String, CaseIterable, Sendable {
    case simple
    case standard
    case studio

    public var displayName: String {
        switch self {
        case .simple: "Simple"
        case .standard: "Standard"
        case .studio: "Studio"
        }
    }

    /// SF Symbol for the toolbar segment.
    public var icon: String {
        switch self {
        case .simple: "person.fill"
        case .standard: "person.crop.circle.badge gearshape"
        case .studio: "testtube.2"
        }
    }
}

/// Read-only access to current settings for non-view code.
public enum TesseraSettings {
    /// Every registered default, keyed by settings key. Exposed so the
    /// registered surface is auditable (e.g. a test can assert no opt-out
    /// key exists where collection is mandatory).
    public static var registeredDefaults: [String: Any] {
        [
            TesseraSettingsKey.defaultRuntime: TesseraRuntime.onDevice.rawValue,
            TesseraSettingsKey.modelDirectory: TesseraSettingsDefault.modelDirectory,
            TesseraSettingsKey.threadCount: TesseraSettingsDefault.threadCount,
            TesseraSettingsKey.maxIterations: TesseraSettingsDefault.maxIterations,
            TesseraSettingsKey.defaultApprovalLevel: ApprovalLevel.prompt.rawValue,
            TesseraSettingsKey.tokenBudget: TesseraSettingsDefault.tokenBudget,
            TesseraSettingsKey.telemetryEnabled: TesseraSettingsDefault.telemetryEnabled,
            TesseraSettingsKey.logLevel: TesseraSettingsDefault.logLevel,
            TesseraSettingsKey.cliPath: TesseraSettingsDefault.cliPath,
            TesseraSettingsKey.tesseraCLIPath: TesseraSettingsDefault.tesseraCLIPath,
            TesseraSettingsKey.tesseraPythonPath: TesseraSettingsDefault.tesseraPythonPath,
            TesseraSettingsKey.llmProviderType: TesseraSettingsDefault.llmProviderType,
            TesseraSettingsKey.remoteAPIBaseURL: TesseraSettingsDefault.remoteAPIBaseURL,
            TesseraSettingsKey.remoteAPIKey: TesseraSettingsDefault.remoteAPIKey,
            TesseraSettingsKey.remoteModelName: TesseraSettingsDefault.remoteModelName,
            TesseraSettingsKey.remoteUseStreaming: TesseraSettingsDefault.remoteUseStreaming,
            TesseraSettingsKey.onDeviceModelPath: TesseraSettingsDefault.onDeviceModelPath,
            TesseraSettingsKey.onDeviceLibraryPath: TesseraSettingsDefault.onDeviceLibraryPath,
            TesseraSettingsKey.onDeviceContextLength: TesseraSettingsDefault.onDeviceContextLength,
            TesseraSettingsKey.onDeviceGPULayers: TesseraSettingsDefault.onDeviceGPULayers,
            TesseraSettingsKey.onboardingComplete: false,
            TesseraSettingsKey.audienceMode: AudienceMode.simple.rawValue,
            TesseraSettingsKey.learningEnabled: TesseraSettingsDefault.learningEnabled,
            TesseraSettingsKey.learningEscalationEnabled: TesseraSettingsDefault.learningEscalationEnabled,
            TesseraSettingsKey.learningTeachers: TesseraSettingsDefault.learningTeachers,
            TesseraSettingsKey.learningAnonymizerAggressiveness: TesseraSettingsDefault.learningAnonymizerAggressiveness,
            TesseraSettingsKey.learningAnonymizerBinary: TesseraSettingsDefault.learningAnonymizerBinary,
            TesseraSettingsKey.learningCaptureScopes: TesseraSettingsDefault.learningCaptureScopes,
            TesseraSettingsKey.learningIdleAdaptation: TesseraSettingsDefault.learningIdleAdaptation,
            TesseraSettingsKey.learningOnPowerOnly: TesseraSettingsDefault.learningOnPowerOnly,
            TesseraSettingsKey.learningDataRetentionDays: TesseraSettingsDefault.learningDataRetentionDays,
            TesseraSettingsKey.learningReferenceTTLDays: TesseraSettingsDefault.learningReferenceTTLDays,
            TesseraSettingsKey.learningMaxConcurrentAgents: TesseraSettingsDefault.learningMaxConcurrentAgents,
            TesseraSettingsKey.learningGuardEpsilon: TesseraSettingsDefault.learningGuardEpsilon,
            TesseraSettingsKey.learningAssessmentIntervalHours: TesseraSettingsDefault.learningAssessmentIntervalHours,
            TesseraSettingsKey.learningBaseModelPath: TesseraSettingsDefault.learningBaseModelPath,
            TesseraSettingsKey.learningMinTracesForTraining: TesseraSettingsDefault.learningMinTracesForTraining,
            TesseraSettingsKey.learningTrainingDryRun: TesseraSettingsDefault.learningTrainingDryRun,
            TesseraSettingsKey.learningTrainBinary: TesseraSettingsDefault.learningTrainBinary,
            TesseraSettingsKey.learningAutoTrain: TesseraSettingsDefault.learningAutoTrain,
            TesseraSettingsKey.learningTrainingIntervalHours: TesseraSettingsDefault.learningTrainingIntervalHours,
            TesseraSettingsKey.learningSessionCuration: TesseraSettingsDefault.learningSessionCuration,
            TesseraSettingsKey.learningRuntimeDraftModel: TesseraSettingsDefault.learningRuntimeDraftModel,
            TesseraSettingsKey.learningRuntimeCapture: TesseraSettingsDefault.learningRuntimeCapture,
            TesseraSettingsKey.learningRuntimeCaptureTopk: TesseraSettingsDefault.learningRuntimeCaptureTopk,
            TesseraSettingsKey.learningRuntimeDraftMax: TesseraSettingsDefault.learningRuntimeDraftMax,
            TesseraSettingsKey.searchProvider: TesseraSettingsDefault.searchProvider,
            TesseraSettingsKey.searxngBaseURL: TesseraSettingsDefault.searxngBaseURL,
            TesseraSettingsKey.tavilyAPIKey: TesseraSettingsDefault.tavilyAPIKey,
            TesseraSettingsKey.coercionMode: TesseraSettingsDefault.coercionMode,
        ]
    }

    /// Register factory defaults. Call once at app launch.
    public static func registerDefaults() {
        UserDefaults.standard.register(defaults: registeredDefaults)
    }

    public static var defaultRuntime: TesseraRuntime {
        let raw = UserDefaults.standard.string(forKey: TesseraSettingsKey.defaultRuntime) ?? TesseraRuntime.onDevice.rawValue
        return TesseraRuntime(rawValue: raw) ?? .onDevice
    }

    public static var modelDirectory: String {
        UserDefaults.standard.string(forKey: TesseraSettingsKey.modelDirectory) ?? TesseraSettingsDefault.modelDirectory
    }

    public static var threadCount: Int {
        UserDefaults.standard.integer(forKey: TesseraSettingsKey.threadCount)
    }

    public static var maxIterations: Int {
        let v = UserDefaults.standard.integer(forKey: TesseraSettingsKey.maxIterations)
        return v > 0 ? v : TesseraSettingsDefault.maxIterations
    }

    public static var defaultApprovalLevel: ApprovalLevel {
        let raw = UserDefaults.standard.string(forKey: TesseraSettingsKey.defaultApprovalLevel) ?? ApprovalLevel.prompt.rawValue
        return ApprovalLevel(rawValue: raw) ?? .prompt
    }

    public static var tokenBudget: Int {
        let v = UserDefaults.standard.integer(forKey: TesseraSettingsKey.tokenBudget)
        return v > 0 ? v : TesseraSettingsDefault.tokenBudget
    }

    public static var telemetryEnabled: Bool {
        UserDefaults.standard.bool(forKey: TesseraSettingsKey.telemetryEnabled)
    }

    public static var logLevel: TesseraLogLevel {
        let raw = UserDefaults.standard.string(forKey: TesseraSettingsKey.logLevel) ?? TesseraSettingsDefault.logLevel
        return TesseraLogLevel(rawValue: raw) ?? .info
    }

    public static var cliPath: String {
        UserDefaults.standard.string(forKey: TesseraSettingsKey.cliPath) ?? TesseraSettingsDefault.cliPath
    }

    /// User-chosen absolute path to the `tessera-cli` binary. Empty
    /// means auto-resolve (the resolver walks known install locations and
    /// `$PATH`). The engine tools read this on every call so a settings
    /// change is honored without restarting the app.
    public static var tesseraCLIPath: String {
        UserDefaults.standard.string(forKey: TesseraSettingsKey.tesseraCLIPath) ?? TesseraSettingsDefault.tesseraCLIPath
    }

    /// User-chosen absolute path to the Python interpreter used by the
    /// Python engine bridge tools. Empty means auto-resolve
    /// (``PythonEngineBridge.discoverPython()``). Read on every Python
    /// tool invocation, so a settings change is honored without restarting.
    public static var tesseraPythonPath: String {
        UserDefaults.standard.string(forKey: TesseraSettingsKey.tesseraPythonPath) ?? TesseraSettingsDefault.tesseraPythonPath
    }

    // MARK: Audience mode

    /// Current audience mode. Simple by default (blueprint section 3.1).
    public static var audienceMode: AudienceMode {
        let raw = UserDefaults.standard.string(forKey: TesseraSettingsKey.audienceMode) ?? AudienceMode.simple.rawValue
        return AudienceMode(rawValue: raw) ?? .simple
    }

    public static func setAudienceMode(_ mode: AudienceMode) {
        UserDefaults.standard.set(mode.rawValue, forKey: TesseraSettingsKey.audienceMode)
    }

    // MARK: LLM provider

    public static var llmProviderType: TesseraLLMProviderType {        let raw = UserDefaults.standard.string(forKey: TesseraSettingsKey.llmProviderType) ?? TesseraSettingsDefault.llmProviderType
        return TesseraLLMProviderType(rawValue: raw) ?? .placeholder
    }

    public static var remoteAPIBaseURL: String {
        UserDefaults.standard.string(forKey: TesseraSettingsKey.remoteAPIBaseURL) ?? TesseraSettingsDefault.remoteAPIBaseURL
    }

    /// The remote LLM provider API key. Lives in the Keychain,
    /// not UserDefaults (see ``TesseraSecretStore``); the getter
    /// also performs the one-shot migration of any legacy
    /// UserDefaults value left by pre-Keychain builds.
    public static var remoteAPIKey: String {
        TesseraSecretStore.secret(account: TesseraSecretStore.remoteAPIKeyAccount)
            ?? TesseraSettingsDefault.remoteAPIKey
    }

    public static var remoteModelName: String {
        UserDefaults.standard.string(forKey: TesseraSettingsKey.remoteModelName) ?? TesseraSettingsDefault.remoteModelName
    }

    public static var remoteUseStreaming: Bool {
        // register(defaults:) seeds this, so the bool read is meaningful.
        UserDefaults.standard.bool(forKey: TesseraSettingsKey.remoteUseStreaming)
    }

    public static var onDeviceModelPath: String {
        UserDefaults.standard.string(forKey: TesseraSettingsKey.onDeviceModelPath) ?? TesseraSettingsDefault.onDeviceModelPath
    }

    public static var onDeviceLibraryPath: String {
        UserDefaults.standard.string(forKey: TesseraSettingsKey.onDeviceLibraryPath) ?? TesseraSettingsDefault.onDeviceLibraryPath
    }

    public static var onDeviceContextLength: Int {
        let v = UserDefaults.standard.integer(forKey: TesseraSettingsKey.onDeviceContextLength)
        return v > 0 ? v : TesseraSettingsDefault.onDeviceContextLength
    }

    public static var onDeviceGPULayers: Int {
        // 0 is a valid explicit value (CPU only); only fall back when unset.
        if UserDefaults.standard.object(forKey: TesseraSettingsKey.onDeviceGPULayers) == nil {
            return TesseraSettingsDefault.onDeviceGPULayers
        }
        return UserDefaults.standard.integer(forKey: TesseraSettingsKey.onDeviceGPULayers)
    }

    // MARK: Learning (self-improving loop)

    public static var learningEnabled: Bool {
        UserDefaults.standard.bool(forKey: TesseraSettingsKey.learningEnabled)
    }

    public static var learningEscalationEnabled: Bool {
        UserDefaults.standard.bool(forKey: TesseraSettingsKey.learningEscalationEnabled)
    }

    /// The escalation teacher pool: exactly the providers the user has
    /// configured keys for. Escalation fans out to all of them.
    public static var learningTeachers: [TesseraTeacherConfig] {
        let raw = UserDefaults.standard.string(forKey: TesseraSettingsKey.learningTeachers) ?? ""
        guard !raw.isEmpty, let data = raw.data(using: .utf8) else { return [] }
        return (try? JSONDecoder().decode([TesseraTeacherConfig].self, from: data)) ?? []
    }

    public static func setLearningTeachers(_ teachers: [TesseraTeacherConfig]) {
        let data = (try? JSONEncoder().encode(teachers)) ?? Data()
        UserDefaults.standard.set(String(data: data, encoding: .utf8) ?? "[]", forKey: TesseraSettingsKey.learningTeachers)
    }

    public static var learningAnonymizerAggressiveness: String {
        UserDefaults.standard.string(forKey: TesseraSettingsKey.learningAnonymizerAggressiveness) ?? TesseraSettingsDefault.learningAnonymizerAggressiveness
    }

    /// Path to the llama-quantize binary that carries the C++ symbol-level
    /// anonymizer. Empty falls back to the installed default location.
    public static var learningAnonymizerBinary: String {
        UserDefaults.standard.string(forKey: TesseraSettingsKey.learningAnonymizerBinary) ?? TesseraSettingsDefault.learningAnonymizerBinary
    }

    /// Enabled capture scopes (build/test/git on by default; editor/screen off).
    public static var learningCaptureScopes: [String] {
        let raw = UserDefaults.standard.string(forKey: TesseraSettingsKey.learningCaptureScopes) ?? TesseraSettingsDefault.learningCaptureScopes
        return raw.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    }

    public static var learningIdleAdaptation: Bool {
        UserDefaults.standard.bool(forKey: TesseraSettingsKey.learningIdleAdaptation)
    }

    public static var learningOnPowerOnly: Bool {
        // register(defaults:) seeds this, so the bool read is meaningful.
        UserDefaults.standard.bool(forKey: TesseraSettingsKey.learningOnPowerOnly)
    }

    public static var learningDataRetentionDays: Int {
        let v = UserDefaults.standard.integer(forKey: TesseraSettingsKey.learningDataRetentionDays)
        return v > 0 ? v : TesseraSettingsDefault.learningDataRetentionDays
    }

    public static var learningReferenceTTLDays: Int {
        let v = UserDefaults.standard.integer(forKey: TesseraSettingsKey.learningReferenceTTLDays)
        return v > 0 ? v : TesseraSettingsDefault.learningReferenceTTLDays
    }

    public static var learningMaxConcurrentAgents: Int {
        let v = UserDefaults.standard.integer(forKey: TesseraSettingsKey.learningMaxConcurrentAgents)
        return v > 0 ? v : TesseraSettingsDefault.learningMaxConcurrentAgents
    }

    public static var learningGuardEpsilon: Double {
        if UserDefaults.standard.object(forKey: TesseraSettingsKey.learningGuardEpsilon) == nil {
            return TesseraSettingsDefault.learningGuardEpsilon
        }
        return UserDefaults.standard.double(forKey: TesseraSettingsKey.learningGuardEpsilon)
    }

    public static var learningAssessmentIntervalHours: Int {
        let v = UserDefaults.standard.integer(forKey: TesseraSettingsKey.learningAssessmentIntervalHours)
        return v > 0 ? v : TesseraSettingsDefault.learningAssessmentIntervalHours
    }

    /// Base model the drafter trainer fine-tunes. Empty means training is
    /// disabled; the orchestrator reports skippedNoModel rather than guessing.
    public static var learningBaseModelPath: String {
        UserDefaults.standard.string(forKey: TesseraSettingsKey.learningBaseModelPath) ?? TesseraSettingsDefault.learningBaseModelPath
    }

    public static var learningMinTracesForTraining: Int {
        let v = UserDefaults.standard.integer(forKey: TesseraSettingsKey.learningMinTracesForTraining)
        return v > 0 ? v : TesseraSettingsDefault.learningMinTracesForTraining
    }

    public static var learningTrainingDryRun: Bool {
        // register(defaults:) seeds this, so the bool read is meaningful.
        UserDefaults.standard.bool(forKey: TesseraSettingsKey.learningTrainingDryRun)
    }

    /// User-chosen tessera-train-lk path. Empty means auto-resolve over
    /// the known locations (see ``TesseraTrainBinaryResolver``).
    public static var learningTrainBinary: String {
        UserDefaults.standard.string(forKey: TesseraSettingsKey.learningTrainBinary) ?? TesseraSettingsDefault.learningTrainBinary
    }

    /// Whether the idle scheduler runs drafter-training sweeps. The
    /// orchestrator's own gates still apply on every sweep.
    public static var learningAutoTrain: Bool {
        // register(defaults:) seeds this, so the bool read is meaningful.
        UserDefaults.standard.bool(forKey: TesseraSettingsKey.learningAutoTrain)
    }

    public static var learningTrainingIntervalHours: Int {
        let v = UserDefaults.standard.integer(forKey: TesseraSettingsKey.learningTrainingIntervalHours)
        return v > 0 ? v : TesseraSettingsDefault.learningTrainingIntervalHours
    }

    /// Whether the idle scheduler runs session-curation sweeps (analysis,
    /// verdicts, replay). The stage's own gates still apply on every sweep.
    public static var learningSessionCuration: Bool {
        // register(defaults:) seeds this, so the bool read is meaningful.
        UserDefaults.standard.bool(forKey: TesseraSettingsKey.learningSessionCuration)
    }

    /// Runtime speculative drafter GGUF. Empty auto-derives
    /// `<base>-tessera-trained.gguf` next to the base model; the sentinel
    /// "-" disables the auto-derive; an explicit path always wins. Read when
    /// the Playground provider initializes.
    public static var learningRuntimeDraftModel: String {
        UserDefaults.standard.string(forKey: TesseraSettingsKey.learningRuntimeDraftModel) ?? TesseraSettingsDefault.learningRuntimeDraftModel
    }

    /// When spec decoding runs, emit runtime traces (topk > 0). Off means
    /// spec decoding with telemetry_topk 0.
    public static var learningRuntimeCapture: Bool {
        if UserDefaults.standard.object(forKey: TesseraSettingsKey.learningRuntimeCapture) == nil {
            return TesseraSettingsDefault.learningRuntimeCapture
        }
        return UserDefaults.standard.bool(forKey: TesseraSettingsKey.learningRuntimeCapture)
    }

    /// Top-k depth for runtime records. Light records at interactive volume;
    /// the replay stage deepens promoted sessions offline.
    public static var learningRuntimeCaptureTopk: Int {
        let v = UserDefaults.standard.integer(forKey: TesseraSettingsKey.learningRuntimeCaptureTopk)
        return v > 0 ? v : TesseraSettingsDefault.learningRuntimeCaptureTopk
    }

    /// Max drafted tokens per spec step. Matches the fork's
    /// params.speculative.draft.n_max default so runtime acceptance
    /// semantics match what calibration measured.
    public static var learningRuntimeDraftMax: Int {
        let v = UserDefaults.standard.integer(forKey: TesseraSettingsKey.learningRuntimeDraftMax)
        return v > 0 ? v : TesseraSettingsDefault.learningRuntimeDraftMax
    }

    // MARK: Web search

    /// Active web-search backend. Keyless DuckDuckGo unless the user opts into
    /// self-hosted SearXNG or vendor Tavily.
    public static var searchProvider: TesseraSearchProviderKind {
        let raw = UserDefaults.standard.string(forKey: TesseraSettingsKey.searchProvider) ?? TesseraSettingsDefault.searchProvider
        return TesseraSearchProviderKind(rawValue: raw) ?? .duckduckgo
    }

    /// Base URL of a self-hosted SearXNG instance (e.g. http://localhost:8888).
    /// Empty means SearXNG search is not configured.
    public static var searxngBaseURL: String {
        UserDefaults.standard.string(forKey: TesseraSettingsKey.searxngBaseURL) ?? TesseraSettingsDefault.searxngBaseURL
    }

    /// Tavily vendor key. Falls back to the TAVILY_API_KEY environment variable
    /// so a key set outside the app still works.
    public static var tavilyAPIKey: String {
        let explicit = UserDefaults.standard.string(forKey: TesseraSettingsKey.tavilyAPIKey) ?? TesseraSettingsDefault.tavilyAPIKey
        if !explicit.isEmpty { return explicit }
        return ProcessInfo.processInfo.environment["TAVILY_API_KEY"] ?? ""
    }

    // MARK: "Plea the Fifth"

    /// Coercion mode. When true, the visible "Plea the Fifth"
    /// controls (the menu bar item, the Settings section) are
    /// hidden or collapsed. The hot-key and the covert trigger
    /// still work - the user knows them, the adversary doesn't.
    /// The menu bar item is Phase 2; the Settings view here
    /// also reads this flag so the user can flip it from
    /// either surface.
    public static var coercionMode: Bool {
        // register(defaults:) seeds this, so the bool read is meaningful.
        UserDefaults.standard.bool(forKey: TesseraSettingsKey.coercionMode)
    }
}
