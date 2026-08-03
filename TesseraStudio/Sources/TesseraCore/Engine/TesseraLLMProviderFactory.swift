import Foundation

/// Which LLM backend the agent loop drives.
public enum TesseraLLMProviderType: String, CaseIterable, Sendable {
    /// Echo/keyword placeholder - needs no configuration or network.
    case placeholder
    /// Remote OpenAI-compatible /v1/chat/completions endpoint.
    case remoteAPI
    /// On-device inference through the libllama (llama.cpp) bridge.
    case onDevice

    public var displayName: String {
        switch self {
        case .placeholder: "Placeholder (built-in)"
        case .remoteAPI: "Remote API (OpenAI-compatible)"
        case .onDevice: "On-Device (llama.cpp)"
        }
    }
}

/// Everything needed to construct a concrete provider. Kept separate from
/// UserDefaults so providers are testable and can be built with explicit
/// values (e.g. from the Settings view or a fixture).
public struct TesseraLLMProviderConfig: Sendable {
    // Remote API
    public var remoteBaseURL: String
    public var remoteAPIKey: String
    public var remoteModelName: String
    public var remoteUseStreaming: Bool
    // On-device
    public var onDeviceModelPath: String
    public var onDeviceLibraryPath: String
    public var onDeviceContextLength: Int
    public var onDeviceGPULayers: Int
    public var onDeviceThreadCount: Int

    public init(
        remoteBaseURL: String = TesseraSettingsDefault.remoteAPIBaseURL,
        remoteAPIKey: String = TesseraSettingsDefault.remoteAPIKey,
        remoteModelName: String = TesseraSettingsDefault.remoteModelName,
        remoteUseStreaming: Bool = TesseraSettingsDefault.remoteUseStreaming,
        onDeviceModelPath: String = TesseraSettingsDefault.onDeviceModelPath,
        onDeviceLibraryPath: String = TesseraSettingsDefault.onDeviceLibraryPath,
        onDeviceContextLength: Int = TesseraSettingsDefault.onDeviceContextLength,
        onDeviceGPULayers: Int = TesseraSettingsDefault.onDeviceGPULayers,
        onDeviceThreadCount: Int = TesseraSettingsDefault.threadCount
    ) {
        self.remoteBaseURL = remoteBaseURL
        self.remoteAPIKey = remoteAPIKey
        self.remoteModelName = remoteModelName
        self.remoteUseStreaming = remoteUseStreaming
        self.onDeviceModelPath = onDeviceModelPath
        self.onDeviceLibraryPath = onDeviceLibraryPath
        self.onDeviceContextLength = onDeviceContextLength
        self.onDeviceGPULayers = onDeviceGPULayers
        self.onDeviceThreadCount = onDeviceThreadCount
    }

    /// Build a config from the current UserDefaults-backed settings.
    public static func fromSettings() -> TesseraLLMProviderConfig {
        TesseraLLMProviderConfig(
            remoteBaseURL: TesseraSettings.remoteAPIBaseURL,
            remoteAPIKey: TesseraSettings.remoteAPIKey,
            remoteModelName: TesseraSettings.remoteModelName,
            remoteUseStreaming: TesseraSettings.remoteUseStreaming,
            onDeviceModelPath: TesseraSettings.onDeviceModelPath,
            onDeviceLibraryPath: TesseraSettings.onDeviceLibraryPath,
            onDeviceContextLength: TesseraSettings.onDeviceContextLength,
            onDeviceGPULayers: TesseraSettings.onDeviceGPULayers,
            onDeviceThreadCount: TesseraSettings.threadCount
        )
    }
}

/// Builds the concrete `LLMProvider` for a requested backend.
///
/// Resolution policy (studio-ux blueprint section 8 Phase 1): the on-device
/// path is the real default. A brand-new user has `llmProviderType == .placeholder`
/// in UserDefaults (the registered default). `makeFromSettings` upgrades that
/// to the on-device provider as soon as a GGUF is discoverable on disk, so
/// onboarding's "pick a starter model" step leaves the user with a working
/// chat. The PlaceholderLLMProvider is kept only as the last-resort fallback
/// for the genuinely empty library, so the app never hard-crashes on a
/// missing model.
public enum TesseraLLMProviderFactory {
    public static func make(
        type: TesseraLLMProviderType,
        config: TesseraLLMProviderConfig
    ) -> any LLMProvider {
        switch type {
        case .placeholder:
            return PlaceholderLLMProvider()
        case .remoteAPI:
            return RemoteLLMProvider(
                baseURL: config.remoteBaseURL,
                apiKey: config.remoteAPIKey,
                modelName: config.remoteModelName,
                useStreaming: config.remoteUseStreaming
            )
        case .onDevice:
            // If the configured model path is missing, fall back to a
            // discoverable GGUF; if none exists either, keep the app
            // responsive with the placeholder rather than crashing on load.
            let resolved = resolvedOnDeviceModelPath(config: config)
            if resolved.isEmpty {
                return PlaceholderLLMProvider()
            }
            var effective = config
            effective.onDeviceModelPath = resolved
            return LlamaLLMProvider(
                modelPath: effective.onDeviceModelPath,
                libraryPath: effective.onDeviceLibraryPath,
                contextLength: effective.onDeviceContextLength,
                gpuLayers: effective.onDeviceGPULayers,
                threadCount: effective.onDeviceThreadCount
            )
        }
    }

    /// Convenience: build from the current settings. Upgrades the legacy
    /// placeholder default to the on-device path when a model is available.
    public static func makeFromSettings() -> any LLMProvider {
        let type = TesseraSettings.llmProviderType
        let config = TesseraLLMProviderConfig.fromSettings()
        if type == .placeholder {
            // Auto-upgrade: if a model is discoverable, use it on-device.
            if !resolvedOnDeviceModelPath(config: config).isEmpty {
                return make(type: .onDevice, config: config)
            }
            return PlaceholderLLMProvider()
        }
        return make(type: type, config: config)
    }

    /// Resolve the on-device model path: the configured one if it exists on
    /// disk, else the first GGUF/.mlmodelc found in the standard scan
    /// directories, else empty string (caller falls back to placeholder).
    public static func resolvedOnDeviceModelPath(config: TesseraLLMProviderConfig) -> String {
        let fm = FileManager.default
        let configured = config.onDeviceModelPath.isEmpty
            ? ""
            : NSString(string: config.onDeviceModelPath).expandingTildeInPath
        if !configured.isEmpty && fm.fileExists(atPath: configured) {
            return configured
        }
        // Discover a starter model in the standard scan directories.
        let dirs = [
            NSString(string: TesseraSettings.modelDirectory).expandingTildeInPath,
            NSString(string: "~/Models").expandingTildeInPath
        ]
        for dir in dirs {
            guard let contents = try? fm.contentsOfDirectory(atPath: dir) else { continue }
            // Prefer a Tessera-quantized .mlmodelc (ANE-ready), then any
            // .mlmodelc, then any .gguf. Sorted for determinism.
            let sortedContents = contents.sorted()
            if let mlc = sortedContents.first(where: { $0.hasSuffix(".mlmodelc") }) {
                return (dir as NSString).appendingPathComponent(mlc)
            }
            if let gguf = sortedContents.first(where: { $0.hasSuffix(".gguf") }) {
                return (dir as NSString).appendingPathComponent(gguf)
            }
        }
        return ""
    }
}
