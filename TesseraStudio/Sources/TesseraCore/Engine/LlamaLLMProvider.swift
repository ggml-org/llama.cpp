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

    // Runtime spec decoding (runtime-traces spec section 7). The drafter
    // path is resolved once at init; nil means trunk-only.
    private let drafterPath: String?
    private let runtimeCapture: Bool
    private let runtimeCaptureTopk: Int
    private let runtimeDraftMax: Int

    private var engine: OpaquePointer?
    private var specEngine: OpaquePointer?
    private var libraryLoaded = false
    private var specLibraryLoaded = false

    // Per-session runtime trace records (llama.tessera.spec.v1 JSONL lines,
    // provenance "runtime"). Flushed to the trace store per completed
    // generation.
    private(set) var sessionTraceBuffer: [String] = []
    private let traceStore: TesseraTraceStore

    public init(
        modelPath: String,
        libraryPath: String = "",
        contextLength: Int = 4096,
        gpuLayers: Int = -1,
        threadCount: Int = 0,
        maxTokens: Int = 1024,
        runtimeDraftModelSetting: String? = nil,
        runtimeCapture: Bool? = nil,
        runtimeCaptureTopk: Int? = nil,
        runtimeDraftMax: Int? = nil,
        traceStore: TesseraTraceStore? = nil
    ) {
        let expanded = NSString(string: modelPath).expandingTildeInPath
        self.modelPath = expanded
        self.libraryPath = libraryPath
        self.contextLength = contextLength
        self.gpuLayers = gpuLayers
        self.threadCount = threadCount
        self.maxTokens = maxTokens

        // v1 reads the drafter path at provider init (spec section 7);
        // hot-swapping after a training cycle is a follow-up.
        let setting = runtimeDraftModelSetting ?? TesseraSettings.learningRuntimeDraftModel
        self.drafterPath = TesseraRuntimeDrafterResolver.resolvedDrafter(
            setting: setting, trunkPath: expanded)
        self.runtimeCapture = runtimeCapture ?? TesseraSettings.learningRuntimeCapture
        self.runtimeCaptureTopk = runtimeCaptureTopk ?? TesseraSettings.learningRuntimeCaptureTopk
        self.runtimeDraftMax = runtimeDraftMax ?? TesseraSettings.learningRuntimeDraftMax
        // Same default directory as the training orchestrator's store, so
        // runtime records join the combined training gate count.
        self.traceStore = traceStore ?? TesseraTraceStore()
    }

    deinit {
        if let engine {
            cllama_engine_free(engine)
        }
        if let specEngine {
            cllama_engine_free_spec(specEngine)
        }
    }

    /// Whether the native library can be loaded on this machine. Cheap probe
    /// the Settings UI can use to show availability without loading a model.
    public nonisolated static func probeLibrary(libraryPath: String = "") -> Bool {
        cllama_load_library(libraryPath) != 0
    }

    /// Whether the spec library (libllama-common.dylib, tessera_rt_*) can be
    /// loaded. Degrades open: a false here just keeps the single-model path.
    public nonisolated static func probeSpecLibrary(libraryPath: String = "") -> Bool {
        cllama_load_spec_library(libraryPath) != 0
    }

    /// Pure routing decision: spec mode needs BOTH a resolved drafter and a
    /// usable spec library. Either missing keeps today's single-model path.
    static func usesSpecEngine(drafterPath: String?, specLibraryAvailable: Bool) -> Bool {
        drafterPath != nil && specLibraryAvailable
    }

    /// The drafter this provider resolved at init (nil = trunk-only).
    /// Exposed for the Settings UI status row and tests.
    public var resolvedRuntimeDrafter: String? { drafterPath }

    public func complete(
        system: String,
        messages: [LLMMessage],
        tools: [ToolDescriptor]
    ) async throws -> LLMResponse {
        try ensureReady()

        let prompt = Self.buildPrompt(system: system, messages: messages, tools: tools)

        // The shim invokes the callback once per decoded token. A reference
        // box is passed through the C user-data pointer so the non-escaping C
        // closure can accumulate without capturing Swift state directly.
        let box = TokenBox()
        let boxPtr = Unmanaged.passUnretained(box).toOpaque()

        if let specEngine {
            // Spec mode: capture on -> topk > 0, one trace record per spec
            // step; capture off -> telemetry_topk 0 (the genuinely-cheap
            // path, no trace callbacks). Both callbacks share the single
            // user-data pointer, so one box carries text and trace lines.
            let topk = runtimeCapture ? Int32(runtimeCaptureTopk) : 0
            let specBox = SpecGenerationBox()
            let specPtr = Unmanaged.passUnretained(specBox).toOpaque()

            let generated = cllama_engine_generate_spec(
                specEngine, prompt, Int32(maxTokens), topk,
                { piece, _, ctx in
                    guard let ctx, let piece else { return }
                    let box = Unmanaged<SpecGenerationBox>.fromOpaque(ctx).takeUnretainedValue()
                    box.text += String(cString: piece)
                },
                topk > 0 ? { line, ctx in
                    guard let ctx, let line else { return }
                    let box = Unmanaged<SpecGenerationBox>.fromOpaque(ctx).takeUnretainedValue()
                    box.traceLines.append(String(cString: line))
                } : nil,
                specPtr
            )

            if generated < 0 {
                throw LlamaLLMError.generationFailed(lastError())
            }

            // Session buffer: flushed to the trace store per completed
            // generation (section 8).
            if !specBox.traceLines.isEmpty {
                sessionTraceBuffer.append(contentsOf: specBox.traceLines)
            }
            flushSessionTraces()

            let parsed = Self.parse(specBox.text)
            return LLMResponse(
                content: parsed.content,
                toolCalls: parsed.toolCalls,
                tokenCount: Int(generated)
            )
        }

        guard let engine else {
            throw LlamaLLMError.modelLoadFailed(lastError())
        }

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

    /// Drain the session buffer into the trace store (section 8). One flush
    /// per completed generation keeps each runtime file to a single sid. On
    /// a store failure the buffer is kept so the next generation retries -
    /// capture plumbing must never break generation itself.
    private func flushSessionTraces() {
        guard !sessionTraceBuffer.isEmpty else { return }
        do {
            // Quarantined sessions are exempt from automatic retention
            // entirely (spec section 12.4); the ledger under the same
            // learning root names them.
            let ledger = TesseraCurationLedger.forStore(traceStore)
            try traceStore.appendRuntime(
                records: sessionTraceBuffer, exemptSids: ledger.quarantinedSids())
            sessionTraceBuffer.removeAll()
        } catch {
            print("[tessera.runtime] trace flush failed, keeping \(sessionTraceBuffer.count) record(s) buffered: \(error.localizedDescription)")
        }
    }

    /// Free the model/context immediately instead of waiting for deinit.
    public func unload() {
        if let engine = self.engine {
            cllama_engine_free(engine)
            self.engine = nil
        }
        if let specEngine = self.specEngine {
            cllama_engine_free_spec(specEngine)
            self.specEngine = nil
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
        if engine == nil && specEngine == nil {
            guard FileManager.default.fileExists(atPath: modelPath) else {
                throw LlamaLLMError.modelLoadFailed("model not found at \(modelPath)")
            }

            // Spec mode when a drafter resolved AND the spec library loads;
            // otherwise today's single-model path, unchanged.
            if !specLibraryLoaded {
                specLibraryLoaded = cllama_load_spec_library(libraryPath) != 0
            }
            if Self.usesSpecEngine(drafterPath: drafterPath,
                                   specLibraryAvailable: specLibraryLoaded),
               let drafterPath {
                let handle = cllama_engine_load_spec(
                    modelPath,
                    drafterPath,
                    UInt32(contextLength),
                    Int32(threadCount),
                    Int32(gpuLayers),
                    Int32(runtimeDraftMax)
                )
                guard let handle else {
                    throw LlamaLLMError.modelLoadFailed(lastError())
                }
                specEngine = handle
                return
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

/// Reference box for spec-mode generation: streamed pieces plus one
/// llama.tessera.spec.v1 JSONL line per spec step. Both callbacks share the
/// single C user-data pointer.
private final class SpecGenerationBox: @unchecked Sendable {
    var text = ""
    var traceLines: [String] = []
}
