import Foundation
import CLlama

/// Errors surfaced by `LlamaLLMProvider`.
public enum LlamaLLMError: Error, LocalizedError {
    case libraryUnavailable(String)
    case modelLoadFailed(String)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .libraryUnavailable(let detail): "libllama is not available: \(detail)"
        case .modelLoadFailed(let detail): "Failed to load the on-device model: \(detail)"
        case .generationFailed(let detail): "On-device generation failed: \(detail)"
        }
    }
}

/// An `LLMProvider` that runs a GGUF model locally through the libllama
/// (llama.cpp) C bridge. It is an actor so model access is serialized and
/// inference stays off the main thread.
///
/// llama.cpp has no native function-calling, so tool definitions are injected
/// into the system prompt and the model is asked to emit calls as a fenced
/// JSON block, which `parse(_:)` extracts back into `LLMToolCall`s.
public actor LlamaLLMProvider: LLMProvider {
    private let modelPath: String
    private let libraryPath: String
    private let contextLength: Int
    private let gpuLayers: Int
    private let threadCount: Int
    private let maxTokens: Int

    private var engine: OpaquePointer?
    private var libraryLoaded = false

    public init(
        modelPath: String,
        libraryPath: String = "",
        contextLength: Int = 4096,
        gpuLayers: Int = -1,
        threadCount: Int = 0,
        maxTokens: Int = 1024
    ) {
        self.modelPath = NSString(string: modelPath).expandingTildeInPath
        self.libraryPath = libraryPath
        self.contextLength = contextLength
        self.gpuLayers = gpuLayers
        self.threadCount = threadCount
        self.maxTokens = maxTokens
    }

    deinit {
        if let engine {
            cllama_engine_free(engine)
        }
    }

    /// Whether the native library can be loaded on this machine. Cheap probe
    /// the Settings UI can use to show availability without loading a model.
    public nonisolated static func probeLibrary(libraryPath: String = "") -> Bool {
        cllama_load_library(libraryPath) != 0
    }

    public func complete(
        system: String,
        messages: [LLMMessage],
        tools: [ToolDescriptor]
    ) async throws -> LLMResponse {
        try ensureReady()
        guard let engine else {
            throw LlamaLLMError.modelLoadFailed(lastError())
        }

        let prompt = Self.buildPrompt(system: system, messages: messages, tools: tools)

        // The shim invokes the callback once per decoded token. A reference
        // box is passed through the C user-data pointer so the non-escaping C
        // closure can accumulate without capturing Swift state directly.
        let box = TokenBox()
        let boxPtr = Unmanaged.passUnretained(box).toOpaque()

        let generated = cllama_engine_generate(engine, prompt, Int32(maxTokens), { piece, _, ctx in
            guard let ctx, let piece else { return }
            let box = Unmanaged<TokenBox>.fromOpaque(ctx).takeUnretainedValue()
            box.text += String(cString: piece)
        }, boxPtr)

        if generated < 0 {
            throw LlamaLLMError.generationFailed(lastError())
        }

        let parsed = Self.parse(box.text)
        return LLMResponse(
            content: parsed.content,
            toolCalls: parsed.toolCalls,
            tokenCount: Int(generated)
        )
    }

    /// Free the model/context immediately instead of waiting for deinit.
    public func unload() {
        if let engine = self.engine {
            cllama_engine_free(engine)
            self.engine = nil
        }
    }

    // MARK: - Setup

    private func ensureReady() throws {
        if !libraryLoaded {
            guard cllama_load_library(libraryPath) != 0 else {
                throw LlamaLLMError.libraryUnavailable(lastError())
            }
            libraryLoaded = true
        }
        if engine == nil {
            guard FileManager.default.fileExists(atPath: modelPath) else {
                throw LlamaLLMError.modelLoadFailed("model not found at \(modelPath)")
            }
            let handle = cllama_engine_load(
                modelPath,
                UInt32(contextLength),
                Int32(threadCount),
                Int32(gpuLayers)
            )
            guard let handle else {
                throw LlamaLLMError.modelLoadFailed(lastError())
            }
            engine = handle
        }
    }

    private func lastError() -> String {
        guard let c = cllama_last_error() else { return "unknown error" }
        let s = String(cString: c)
        return s.isEmpty ? "unknown error" : s
    }

    // MARK: - Prompt construction

    static func buildPrompt(system: String, messages: [LLMMessage], tools: [ToolDescriptor]) -> String {
        var lines: [String] = []
        var systemBlock = system
        if !tools.isEmpty {
            systemBlock += "\n\n" + toolsInstruction(tools)
        }
        lines.append("### System\n\(systemBlock)")
        for m in messages {
            let role: String
            switch m.role {
            case "user": role = "User"
            case "assistant": role = "Assistant"
            default: role = "Tool"
            }
            lines.append("### \(role)\n\(m.content)")
        }
        lines.append("### Assistant\n")
        return lines.joined(separator: "\n\n")
    }

    static func toolsInstruction(_ tools: [ToolDescriptor]) -> String {
        var schemas: [String] = []
        for tool in tools {
            schemas.append("""
            - \(tool.name): \(tool.description)
              parameters: \(tool.parameters.toJSON())
            """)
        }
        return """
        You can call the following tools. To call a tool, respond with a JSON
        code fence of the exact form:
        ```tool
        {"name": "<tool_name>", "arguments": { ... }}
        ```
        Emit at most one tool call, and nothing else, when a tool is needed.
        Otherwise answer the user directly in plain text.

        Available tools:
        \(schemas.joined(separator: "\n"))
        """
    }

    // MARK: - Output parsing

    static func parse(_ output: String) -> (content: String, toolCalls: [LLMToolCall]) {
        guard let range = output.range(of: "```tool") else {
            return (output.trimmingCharacters(in: .whitespacesAndNewlines), [])
        }
        let content = output[output.startIndex..<range.lowerBound]
            .trimmingCharacters(in: .whitespacesAndNewlines)

        // Isolate the fenced JSON block after the ```tool marker.
        let afterMarker = output[range.upperBound...]
        guard let endFence = afterMarker.range(of: "```") else {
            return (content, [])
        }
        let jsonText = afterMarker[afterMarker.startIndex..<endFence.lowerBound]
            .trimmingCharacters(in: .whitespacesAndNewlines)

        guard let call = decodeToolCall(jsonText) else {
            return (content, [])
        }
        return (content, [call])
    }

    private static func decodeToolCall(_ json: String) -> LLMToolCall? {
        guard let data = json.data(using: .utf8) else { return nil }
        guard let obj = try? JSONDecoder().decode([String: JSONValue].self, from: data) else {
            return nil
        }
        guard let name = obj["name"]?.stringValue, !name.isEmpty else { return nil }
        var arguments: [String: JSONValue] = [:]
        if case .object(let args)? = obj["arguments"] {
            arguments = args
        }
        return LLMToolCall(name: name, arguments: arguments)
    }
}

/// Reference box used to accumulate streamed token pieces from the C callback.
private final class TokenBox: @unchecked Sendable {
    var text = ""
}
