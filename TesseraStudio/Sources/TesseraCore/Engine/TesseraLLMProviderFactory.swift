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

/// Builds the concrete `LLMProvider` for a requested backend. Defaults to the
/// placeholder so an unconfigured app keeps working exactly as before.
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
            return LlamaLLMProvider(
                modelPath: config.onDeviceModelPath,
                libraryPath: config.onDeviceLibraryPath,
                contextLength: config.onDeviceContextLength,
                gpuLayers: config.onDeviceGPULayers,
                threadCount: config.onDeviceThreadCount
            )
        }
    }

    /// Convenience: build from the current settings.
    public static func makeFromSettings() -> any LLMProvider {
        make(type: TesseraSettings.llmProviderType, config: .fromSettings())
    }
}
