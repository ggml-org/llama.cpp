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
    // First-run
    public static let onboardingComplete = "tessera.settings.onboardingComplete"
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
            TesseraSettingsKey.onboardingComplete: false,
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
}
