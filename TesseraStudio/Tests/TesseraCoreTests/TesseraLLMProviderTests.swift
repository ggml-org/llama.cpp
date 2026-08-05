import XCTest
@testable import TesseraCore

final class TesseraLLMProviderFactoryTests: XCTestCase {
    func testMakePlaceholder() {
        let provider = TesseraLLMProviderFactory.make(type: .placeholder, config: .init())
        XCTAssertTrue(provider is PlaceholderLLMProvider)
    }

    func testMakeRemote() {
        let provider = TesseraLLMProviderFactory.make(type: .remoteAPI, config: .init())
        XCTAssertTrue(provider is RemoteLLMProvider)
    }

    func testMakeOnDevice() {
        // The factory only requires a resolvable model path; GGUF validity
        // is checked later, at model load. Point at a temp file so the test
        // does not depend on the model library of the machine running it
        // (with an empty config the factory scans ~/Models and falls back
        // to PlaceholderLLMProvider when nothing is found).
        let modelURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("tessera-ondevice-factory-\(UUID().uuidString).gguf")
        FileManager.default.createFile(atPath: modelURL.path, contents: Data())
        defer { try? FileManager.default.removeItem(at: modelURL) }

        let provider = TesseraLLMProviderFactory.make(
            type: .onDevice,
            config: TesseraLLMProviderConfig(onDeviceModelPath: modelURL.path)
        )
        XCTAssertTrue(provider is LlamaLLMProvider)
    }

    func testProviderTypeRawValues() {
        XCTAssertEqual(TesseraLLMProviderType.placeholder.rawValue, "placeholder")
        XCTAssertEqual(TesseraLLMProviderType.remoteAPI.rawValue, "remoteAPI")
        XCTAssertEqual(TesseraLLMProviderType.onDevice.rawValue, "onDevice")
        XCTAssertEqual(TesseraLLMProviderType.allCases.count, 3)
    }

    func testConfigFromDefaults() {
        let config = TesseraLLMProviderConfig()
        XCTAssertEqual(config.remoteBaseURL, TesseraSettingsDefault.remoteAPIBaseURL)
        XCTAssertEqual(config.remoteModelName, TesseraSettingsDefault.remoteModelName)
        XCTAssertEqual(config.onDeviceGPULayers, TesseraSettingsDefault.onDeviceGPULayers)
    }
}

final class RemoteLLMProviderTests: XCTestCase {
    func testStripsTrailingSlash() {
        let provider = RemoteLLMProvider(baseURL: "http://localhost:8080/v1/", apiKey: "", modelName: "m")
        XCTAssertEqual(provider.baseURL, "http://localhost:8080/v1")
    }

    func testKeepsCleanBaseURL() {
        let provider = RemoteLLMProvider(baseURL: "https://api.openai.com/v1", apiKey: "k", modelName: "gpt-4")
        XCTAssertEqual(provider.baseURL, "https://api.openai.com/v1")
        XCTAssertEqual(provider.modelName, "gpt-4")
        XCTAssertTrue(provider.useStreaming)
    }
}

final class LlamaPromptTests: XCTestCase {
    private let tools = [
        ToolDescriptor(name: "list_models", description: "List models", parameters: JSONSchema()),
    ]

    func testBuildPromptInjectsToolSchemas() {
        let prompt = LlamaLLMProvider.buildPrompt(
            system: "You are helpful.",
            messages: [LLMMessage(role: "user", content: "hi")],
            tools: tools
        )
        XCTAssertTrue(prompt.contains("### System"))
        XCTAssertTrue(prompt.contains("list_models"))
        XCTAssertTrue(prompt.contains("### User"))
        XCTAssertTrue(prompt.contains("hi"))
        XCTAssertTrue(prompt.hasSuffix("### Assistant\n"))
    }

    func testBuildPromptOmitsToolBlockWhenNoTools() {
        let prompt = LlamaLLMProvider.buildPrompt(
            system: "You are helpful.",
            messages: [LLMMessage(role: "user", content: "hi")],
            tools: []
        )
        XCTAssertFalse(prompt.contains("Available tools"))
    }

    func testParseExtractsToolCall() {
        let output = """
        Let me check.
        ```tool
        {"name": "quantize", "arguments": {"model_path": "/m.gguf"}}
        ```
        """
        let parsed = LlamaLLMProvider.parse(output)
        XCTAssertEqual(parsed.toolCalls.count, 1)
        XCTAssertEqual(parsed.toolCalls.first?.name, "quantize")
        XCTAssertEqual(parsed.toolCalls.first?.arguments["model_path"]?.stringValue, "/m.gguf")
        XCTAssertEqual(parsed.content, "Let me check.")
    }

    func testParsePlainTextHasNoToolCalls() {
        let parsed = LlamaLLMProvider.parse("Just a plain answer.")
        XCTAssertTrue(parsed.toolCalls.isEmpty)
        XCTAssertEqual(parsed.content, "Just a plain answer.")
    }

    func testParseIgnoresMalformedToolBlock() {
        let parsed = LlamaLLMProvider.parse("```tool\nnot json\n```")
        XCTAssertTrue(parsed.toolCalls.isEmpty)
    }
}
