import Foundation
import Observation

/// Events emitted by the agent loop during execution.
public enum AgentEvent: Sendable {
    case thinking(String)
    case text(String)
    case toolCall(name: String, arguments: [String: JSONValue])
    case toolResult(name: String, result: ToolResult)
    case error(String)
    case done
}

/// The streaming agent loop. Takes a user message, calls the LLM,
/// parses tool calls, executes them via the registry with approval
/// gating, and streams events back to the UI.
@Observable
@MainActor
public final class TesseraAgentLoop {
    private let registry: TesseraToolRegistry
    public let approvalEngine: TesseraApprovalEngine
    private let llmProvider: any LLMProvider
    private let maxIterations: Int

    public private(set) var isRunning = false
    public private(set) var currentTask: Task<Void, Never>?
    public private(set) var tokenBudget: TokenBudget

    public init(
        registry: TesseraToolRegistry,
        approvalEngine: TesseraApprovalEngine,
        llmProvider: (any LLMProvider)? = nil,
        maxIterations: Int = 10,
        tokenLimit: Int = 8192
    ) {
        self.registry = registry
        self.approvalEngine = approvalEngine
        self.llmProvider = llmProvider ?? PlaceholderLLMProvider()
        self.maxIterations = max(1, maxIterations)
        self.tokenBudget = TokenBudget(used: 0, limit: tokenLimit)
    }

    /// Run the agent loop on a user message, returning a stream of events.
    public func run(
        userMessage: String,
        history: [ChatMessage]
    ) -> AsyncStream<AgentEvent> {
        AsyncStream { continuation in
            let task = Task { @MainActor in
                self.isRunning = true
                defer {
                    self.isRunning = false
                    continuation.finish()
                }

                do {
                    try await self.executeLoop(
                        userMessage: userMessage,
                        history: history,
                        continuation: continuation
                    )
                } catch is CancellationError {
                    continuation.yield(.error("Cancelled"))
                } catch {
                    continuation.yield(.error(error.localizedDescription))
                }

                continuation.yield(.done)
            }
            self.currentTask = task

            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }

    public func cancel() {
        currentTask?.cancel()
    }

    // MARK: - Private

    private func executeLoop(
        userMessage: String,
        history: [ChatMessage],
        continuation: AsyncStream<AgentEvent>.Continuation
    ) async throws {
        let systemPrompt = buildSystemPrompt()
        var messages = history.map { LLMMessage(role: $0.role.rawValue, content: $0.content) }
        messages.append(LLMMessage(role: "user", content: userMessage))

        var iterations = 0

        while iterations < maxIterations {
            iterations += 1

            guard !Task.isCancelled else { throw CancellationError() }

            continuation.yield(.thinking("Calling model..."))

            let response = try await llmProvider.complete(
                system: systemPrompt,
                messages: messages,
                tools: registry.allTools.map { ToolDescriptor(name: $0.name, description: $0.description, parameters: $0.parameters) }
            )

            tokenBudget.used += response.tokenCount

            // Emit any text content
            if !response.content.isEmpty {
                continuation.yield(.text(response.content))
            }

            // No tool calls -> done
            guard !response.toolCalls.isEmpty else { break }

            // Execute each tool call
            for call in response.toolCalls {
                guard !Task.isCancelled else { throw CancellationError() }

                continuation.yield(.toolCall(name: call.name, arguments: call.arguments))

                // Approval gate
                let approved = await approvalEngine.requestApproval(
                    toolName: call.name,
                    arguments: call.arguments
                )
                guard approved else {
                    let denied = ToolResult.fail("Denied by user")
                    continuation.yield(.toolResult(name: call.name, result: denied))
                    messages.append(LLMMessage(role: "tool", content: "Tool '\(call.name)' was denied by the user."))
                    continue
                }

                // Execute
                let result: ToolResult
                if let tool = registry.tool(named: call.name) {
                    do {
                        result = try await tool.execute(arguments: call.arguments)
                    } catch {
                        result = .fail(error.localizedDescription)
                    }
                } else {
                    result = .fail("Unknown tool: \(call.name)")
                }

                continuation.yield(.toolResult(name: call.name, result: result))
                messages.append(LLMMessage(role: "tool", content: result.output))
            }
        }
    }

    private func buildSystemPrompt() -> String {
        """
        You are Tessera Studio Agent, an assistant for quantizing, calibrating,
        and deploying LLMs with the Tessera engine. You help users manage models,
        run calibration, evolve quantization policies, and evaluate results.

        \(registry.systemPromptToolsBlock())
        """
    }
}

// MARK: - Token Budget

public struct TokenBudget: Sendable {
    public var used: Int
    public let limit: Int

    public init(used: Int, limit: Int) {
        self.used = used
        self.limit = limit
    }

    public var fraction: Double {
        guard limit > 0 else { return 0 }
        return min(Double(used) / Double(limit), 1.0)
    }

    public var remaining: Int { max(limit - used, 0) }
}

// MARK: - LLM Provider Protocol

public struct LLMMessage: Sendable {
    public let role: String
    public let content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

public struct ToolDescriptor: Sendable {
    public let name: String
    public let description: String
    public let parameters: JSONSchema

    public init(name: String, description: String, parameters: JSONSchema) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }
}

public struct LLMToolCall: Sendable {
    public let name: String
    public let arguments: [String: JSONValue]

    public init(name: String, arguments: [String: JSONValue]) {
        self.name = name
        self.arguments = arguments
    }
}

public struct LLMResponse: Sendable {
    public let content: String
    public let toolCalls: [LLMToolCall]
    public let tokenCount: Int

    public init(content: String, toolCalls: [LLMToolCall], tokenCount: Int) {
        self.content = content
        self.toolCalls = toolCalls
        self.tokenCount = tokenCount
    }
}

public protocol LLMProvider: Sendable {
    func complete(
        system: String,
        messages: [LLMMessage],
        tools: [ToolDescriptor]
    ) async throws -> LLMResponse
}

/// Placeholder LLM that echoes the user message. Replace with a real
/// provider (local llama.cpp server, MLX, or cloud API) in production.
public struct PlaceholderLLMProvider: LLMProvider {
    public init() {}

    public func complete(
        system: String,
        messages: [LLMMessage],
        tools: [ToolDescriptor]
    ) async throws -> LLMResponse {
        guard let last = messages.last else {
            return LLMResponse(content: "No input.", toolCalls: [], tokenCount: 0)
        }

        // Simple keyword-based tool dispatch for demonstration
        let lower = last.content.lowercased()
        if lower.contains("list") && lower.contains("model") {
            return LLMResponse(
                content: "",
                toolCalls: [LLMToolCall(name: "list_models", arguments: [:])],
                tokenCount: 25
            )
        }
        if lower.contains("inspect") || lower.contains("sidecar") {
            let path = extractPath(from: last.content) ?? ""
            return LLMResponse(
                content: "",
                toolCalls: [LLMToolCall(name: "inspect_sidecar", arguments: ["path": .string(path)])],
                tokenCount: 30
            )
        }
        if lower.contains("quantize") {
            return LLMResponse(
                content: "",
                toolCalls: [LLMToolCall(name: "quantize", arguments: [
                    "model_path": .string(extractPath(from: last.content) ?? ""),
                    "output_path": .string(""),
                    "policy_path": .string(""),
                ])],
                tokenCount: 35
            )
        }

        let reply = "I'm the Tessera Studio agent (placeholder). You said: \"\(last.content)\". " +
            "In production, this connects to a local or remote LLM. " +
            "Try asking me to list models, inspect a sidecar, or quantize a model."
        return LLMResponse(content: reply, toolCalls: [], tokenCount: reply.count / 4)
    }

    private func extractPath(from text: String) -> String? {
        let words = text.split(separator: " ")
        return words.first { $0.contains("/") || $0.hasSuffix(".gguf") || $0.hasSuffix(".json") }
            .map(String.init)
    }
}
