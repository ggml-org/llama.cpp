import SwiftUI
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
    @AppStorage(TesseraSettingsKey.remoteAPIKey) private var remoteAPIKey = TesseraSettingsDefault.remoteAPIKey
    @AppStorage(TesseraSettingsKey.remoteModelName) private var remoteModelName = TesseraSettingsDefault.remoteModelName
    @AppStorage(TesseraSettingsKey.remoteUseStreaming) private var remoteUseStreaming = TesseraSettingsDefault.remoteUseStreaming
    @AppStorage(TesseraSettingsKey.onDeviceModelPath) private var onDeviceModelPath = TesseraSettingsDefault.onDeviceModelPath
    @AppStorage(TesseraSettingsKey.onDeviceLibraryPath) private var onDeviceLibraryPath = TesseraSettingsDefault.onDeviceLibraryPath
    @AppStorage(TesseraSettingsKey.onDeviceContextLength) private var onDeviceContextLength = TesseraSettingsDefault.onDeviceContextLength
    @AppStorage(TesseraSettingsKey.onDeviceGPULayers) private var onDeviceGPULayers = TesseraSettingsDefault.onDeviceGPULayers

    var body: some View {
        TabView {
            generalTab
                .tabItem { Label("General", systemImage: "gearshape") }
            agentTab
                .tabItem { Label("Agent", systemImage: "cpu") }
            modelTab
                .tabItem { Label("Model", systemImage: "brain") }
            advancedTab
                .tabItem { Label("Advanced", systemImage: "slider.horizontal.3") }
        }
        .frame(width: 520, height: 420)
    }

    private var generalTab: some View {
        Form {
            Picker("Default runtime", selection: $defaultRuntime) {
                ForEach(TesseraRuntime.allCases, id: \.rawValue) { rt in
                    Text(rt.displayName).tag(rt.rawValue)
                }
            }
            TextField("Model directory", text: $modelDirectory)
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
                    SecureField("API key", text: $remoteAPIKey)
                    TextField("Model name", text: $remoteModelName)
                    Toggle("Stream responses (SSE)", isOn: $remoteUseStreaming)
                }
            }

            if llmProviderType == TesseraLLMProviderType.onDevice.rawValue {
                Section("On-Device (llama.cpp)") {
                    TextField("GGUF model path", text: $onDeviceModelPath)
                    TextField("libllama.dylib path (optional)", text: $onDeviceLibraryPath)
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

    private var advancedTab: some View {
        Form {
            Toggle("Enable telemetry", isOn: $telemetryEnabled)
            Picker("Log level", selection: $logLevel) {
                ForEach(TesseraLogLevel.allCases, id: \.rawValue) { level in
                    Text(level.rawValue.uppercased()).tag(level.rawValue)
                }
            }
            TextField("Custom CLI path", text: $cliPath)
        }
        .formStyle(.grouped)
        .padding()
    }
}
