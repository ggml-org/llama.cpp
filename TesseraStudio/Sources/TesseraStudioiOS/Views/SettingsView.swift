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
            }
        }
        .navigationTitle("Settings")
    }
}
#endif
