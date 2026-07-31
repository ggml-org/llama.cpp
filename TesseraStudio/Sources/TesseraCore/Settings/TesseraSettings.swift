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
}

/// Log levels offered in Advanced settings.
public enum TesseraLogLevel: String, CaseIterable, Sendable {
    case debug, info, warning, error
}

/// Read-only access to current settings for non-view code.
public enum TesseraSettings {
    /// Register factory defaults. Call once at app launch.
    public static func registerDefaults() {
        UserDefaults.standard.register(defaults: [
            TesseraSettingsKey.defaultRuntime: TesseraRuntime.onDevice.rawValue,
            TesseraSettingsKey.modelDirectory: TesseraSettingsDefault.modelDirectory,
            TesseraSettingsKey.threadCount: TesseraSettingsDefault.threadCount,
            TesseraSettingsKey.maxIterations: TesseraSettingsDefault.maxIterations,
            TesseraSettingsKey.defaultApprovalLevel: ApprovalLevel.prompt.rawValue,
            TesseraSettingsKey.tokenBudget: TesseraSettingsDefault.tokenBudget,
            TesseraSettingsKey.telemetryEnabled: TesseraSettingsDefault.telemetryEnabled,
            TesseraSettingsKey.logLevel: TesseraSettingsDefault.logLevel,
            TesseraSettingsKey.cliPath: TesseraSettingsDefault.cliPath,
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
        ])
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

    // MARK: LLM provider

    public static var llmProviderType: TesseraLLMProviderType {
        let raw = UserDefaults.standard.string(forKey: TesseraSettingsKey.llmProviderType) ?? TesseraSettingsDefault.llmProviderType
        return TesseraLLMProviderType(rawValue: raw) ?? .placeholder
    }

    public static var remoteAPIBaseURL: String {
        UserDefaults.standard.string(forKey: TesseraSettingsKey.remoteAPIBaseURL) ?? TesseraSettingsDefault.remoteAPIBaseURL
    }

    public static var remoteAPIKey: String {
        UserDefaults.standard.string(forKey: TesseraSettingsKey.remoteAPIKey) ?? TesseraSettingsDefault.remoteAPIKey
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
}
