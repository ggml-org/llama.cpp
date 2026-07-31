import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Errors surfaced by `RemoteLLMProvider`.
public enum RemoteLLMError: Error, LocalizedError {
    case invalidURL(String)
    case invalidResponse
    case httpError(statusCode: Int, body: String)
    case rateLimited(body: String)
    case decoding(String)
    case transport(String)

    public var errorDescription: String? {
        switch self {
        case .invalidURL(let url): "Invalid API URL: \(url)"
        case .invalidResponse: "The API returned an unexpected response."
        case .httpError(let code, let body): "API error (\(code)): \(body)"
        case .rateLimited(let body): "Rate limited by the API: \(body)"
        case .decoding(let detail): "Failed to decode API response: \(detail)"
        case .transport(let detail): "Network error: \(detail)"
        }
    }
}

/// An `LLMProvider` backed by any OpenAI-compatible `/v1/chat/completions`
/// endpoint (OpenAI, a local llama-server, Anthropic-behind-proxy, etc.).
/// Supports both blocking and SSE-streaming completions and translates
/// OpenAI function-calling tool calls into the agent loop's `LLMToolCall`.
public struct RemoteLLMProvider: LLMProvider {
    public let baseURL: String
    public let apiKey: String
    public let modelName: String
    public let useStreaming: Bool

    private let session: URLSession
    private let decoder: JSONDecoder

    public init(
        baseURL: String,
        apiKey: String,
        modelName: String,
        useStreaming: Bool = true,
        session: URLSession = .shared
    ) {
        self.baseURL = baseURL.hasSuffix("/") ? String(baseURL.dropLast()) : baseURL
        self.apiKey = apiKey
        self.modelName = modelName
        self.useStreaming = useStreaming
        self.session = session
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        self.decoder = decoder
    }

    public func complete(
        system: String,
        messages: [LLMMessage],
        tools: [ToolDescriptor]
    ) async throws -> LLMResponse {
        guard let url = URL(string: baseURL + "/chat/completions") else {
            throw RemoteLLMError.invalidURL(baseURL + "/chat/completions")
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if !apiKey.isEmpty {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }

        let body = RequestBody(
            model: modelName,
            messages: buildMessages(system: system, messages: messages),
            tools: buildTools(tools),
            stream: useStreaming
        )
        do {
            request.httpBody = try JSONEncoder().encode(body)
        } catch {
            throw RemoteLLMError.decoding(error.localizedDescription)
        }

        return useStreaming
            ? try await completeStreaming(request)
            : try await completeBlocking(request)
    }

    // MARK: - Blocking

    private func completeBlocking(_ request: URLRequest) async throws -> LLMResponse {
        let data = try await perform(request)
        let body: ResponseBody
        do {
            body = try decoder.decode(ResponseBody.self, from: data)
        } catch {
            throw RemoteLLMError.decoding(error.localizedDescription)
        }
        guard let choice = body.choices.first else {
            throw RemoteLLMError.invalidResponse
        }
        let content = choice.message.content ?? ""
        let toolCalls = (choice.message.toolCalls ?? []).map(Self.makeToolCall)
        let tokens = body.usage?.totalTokens ?? Self.estimateTokens(content)
        return LLMResponse(content: content, toolCalls: toolCalls, tokenCount: tokens)
    }

    // MARK: - Streaming (SSE)

    private func completeStreaming(_ request: URLRequest) async throws -> LLMResponse {
        let (bytes, response) = try await session.bytes(for: request)
        try Self.validateHTTP(response, bodyHint: "streaming response")

        var content = ""
        var partial: [Int: StreamToolCallAccumulator] = [:]
        var usageTokens: Int?

        for try await line in bytes.lines {
            // SSE payloads are "data: <json>"; ignore event:/id:/keep-alives.
            guard line.hasPrefix("data:") else { continue }
            let payload = line.dropFirst("data:".count).trimmingCharacters(in: .whitespaces)
            if payload == "[DONE]" { break }
            guard let data = payload.data(using: .utf8) else { continue }

            let chunk: StreamChunk
            do {
                chunk = try decoder.decode(StreamChunk.self, from: data)
            } catch {
                // Tolerate a malformed keep-alive chunk rather than aborting.
                continue
            }
            if let total = chunk.usage?.totalTokens {
                usageTokens = total
            }
            guard let choice = chunk.choices.first else { continue }
            if let delta = choice.delta.content {
                content += delta
            }
            for tc in choice.delta.toolCalls ?? [] {
                let index = tc.index ?? 0
                var acc = partial[index] ?? StreamToolCallAccumulator()
                if let id = tc.id { acc.id = id }
                if let name = tc.function?.name { acc.name += name }
                if let args = tc.function?.arguments { acc.arguments += args }
                partial[index] = acc
            }
        }

        let toolCalls = partial
            .sorted { $0.key < $1.key }
            .map { Self.makeToolCall(from: $0.value) }
        let tokens = usageTokens ?? Self.estimateTokens(content)
        return LLMResponse(content: content, toolCalls: toolCalls, tokenCount: tokens)
    }

    // MARK: - Request building

    private func buildMessages(system: String, messages: [LLMMessage]) -> [RequestBody.Message] {
        var out: [RequestBody.Message] = []
        if !system.isEmpty {
            out.append(RequestBody.Message(role: "system", content: system))
        }
        for m in messages {
            out.append(RequestBody.Message(role: m.role, content: m.content))
        }
        return out
    }

    private func buildTools(_ tools: [ToolDescriptor]) -> [RequestBody.Tool]? {
        guard !tools.isEmpty else { return nil }
        return tools.map { tool in
            RequestBody.Tool(function: .init(
                name: tool.name,
                description: tool.description,
                parameters: tool.parameters
            ))
        }
    }

    // MARK: - Transport helpers

    private func perform(_ request: URLRequest) async throws -> Data {
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            throw RemoteLLMError.transport(error.localizedDescription)
        }
        try Self.validateHTTP(response, bodyHint: String(data: data, encoding: .utf8) ?? "")
        return data
    }

    private static func validateHTTP(_ response: URLResponse, bodyHint: String) throws {
        guard let http = response as? HTTPURLResponse else {
            throw RemoteLLMError.invalidResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            if http.statusCode == 429 {
                throw RemoteLLMError.rateLimited(body: bodyHint)
            }
            throw RemoteLLMError.httpError(statusCode: http.statusCode, body: bodyHint)
        }
    }

    // MARK: - Tool call parsing

    private static func makeToolCall(_ tc: ResponseBody.ToolCall) -> LLMToolCall {
        LLMToolCall(
            name: tc.function?.name ?? "",
            arguments: parseArguments(tc.function?.arguments)
        )
    }

    private static func makeToolCall(from acc: StreamToolCallAccumulator) -> LLMToolCall {
        LLMToolCall(name: acc.name, arguments: parseArguments(acc.arguments))
    }

    /// OpenAI sends tool arguments as a JSON *string*. Decode it into the
    /// agent loop's typed map; fall back to empty on partial/invalid JSON.
    private static func parseArguments(_ raw: String?) -> [String: JSONValue] {
        guard let raw, !raw.isEmpty, let data = raw.data(using: .utf8) else { return [:] }
        return (try? JSONDecoder().decode([String: JSONValue].self, from: data)) ?? [:]
    }

    private static func estimateTokens(_ text: String) -> Int {
        max(1, text.count / 4)
    }
}

// MARK: - Wire format

private struct RequestBody: Encodable {
    struct Message: Encodable {
        let role: String
        let content: String
    }
    struct Tool: Encodable {
        let type: String = "function"
        let function: Function
        struct Function: Encodable {
            let name: String
            let description: String
            let parameters: JSONSchema
        }
    }
    let model: String
    let messages: [Message]
    let tools: [Tool]?
    let stream: Bool
}

private struct ResponseBody: Decodable {
    struct Message: Decodable {
        let role: String?
        let content: String?
        let toolCalls: [ToolCall]?
    }
    struct Choice: Decodable {
        let message: Message
        let finishReason: String?
    }
    struct ToolCall: Decodable {
        let id: String?
        let index: Int?
        let type: String?
        // Optional: streaming deltas may carry only id/index in some chunks.
        let function: Function?
        struct Function: Decodable {
            let name: String?
            let arguments: String?
        }
    }
    struct Usage: Decodable {
        let totalTokens: Int?
        let completionTokens: Int?
    }
    let choices: [Choice]
    let usage: Usage?
}

private struct StreamChunk: Decodable {
    struct Delta: Decodable {
        let content: String?
        let toolCalls: [ResponseBody.ToolCall]?
    }
    struct Choice: Decodable {
        let delta: Delta
        let finishReason: String?
    }
    let choices: [Choice]
    let usage: ResponseBody.Usage?
}

/// Accumulates a single streamed tool call across many SSE deltas. OpenAI
/// splits the function name and the arguments JSON across chunks, keyed by
/// the tool call's `index`.
private struct StreamToolCallAccumulator {
    var id: String?
    var name = ""
    var arguments = ""
}
