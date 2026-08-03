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
    private let skillLoader: TesseraSkillLoader
    private let permissionProfile: TesseraPermissionProfile
    private let sandboxEnforceable: Bool

    public private(set) var isRunning = false
    public private(set) var currentTask: Task<Void, Never>?
    public private(set) var tokenBudget: TokenBudget
    /// Stable id for this loop run; tags approval receipts (autonomy spec 14).
    public let sessionID = UUID().uuidString

    public init(
        registry: TesseraToolRegistry,
        approvalEngine: TesseraApprovalEngine,
        llmProvider: (any LLMProvider)? = nil,
        maxIterations: Int = 10,
        tokenLimit: Int = 8192,
        skillLoader: TesseraSkillLoader? = nil,
        permissionProfile: TesseraPermissionProfile = .standard,
        sandboxEnforceable: Bool? = nil
    ) {
        self.registry = registry
        self.approvalEngine = approvalEngine
        // Fallback only when a caller passes nil. ContentView passes a real
        // provider from TesseraLLMProviderFactory.makeFromSettings().
        self.llmProvider = llmProvider ?? PlaceholderLLMProvider()
        self.maxIterations = max(1, maxIterations)
        self.tokenBudget = TokenBudget(used: 0, limit: tokenLimit)
        self.skillLoader = skillLoader ?? TesseraSkillLoader()
        self.permissionProfile = permissionProfile
        // nil -> platform default: sandboxed on iOS (App Store), not on macOS
        // (Developer ID). Only matters once the safety spine's auto-approve path
        // is wired; the reject path it uses today is sandbox-independent.
        #if os(iOS)
        self.sandboxEnforceable = sandboxEnforceable ?? true
        #else
        self.sandboxEnforceable = sandboxEnforceable ?? false
        #endif
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
        // Publish the live session id so a YOLO session started from the UI
        // can bind to this loop (autonomy-calibration-design.md 10).
        approvalEngine.autonomy.setActiveSession(sessionID)
        let systemPrompt = buildSystemPrompt(userMessage: userMessage)
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

                // Autonomy-aware gate (autonomy-calibration-design.md 7):
                // the base safety spine (S2/S3/S4) plus the learned-permission
                // ratchet (steps 4, 6). Three outcomes:
                //   reject     -> never run; recorded in the breaker.
                //   askUser    -> force a prompt; a user denial feeds the
                //                 breaker and revokes the class (ratchet).
                //   autoApprove-> run without prompting (base or learned).
                // Every outcome is receipt-logged (section 14).
                let action = PendingAction(toolName: call.name, arguments: call.arguments)
                let gate = approvalEngine.gateCheck(
                    for: action,
                    permissionProfile: permissionProfile,
                    sandboxEnforceable: sandboxEnforceable,
                    sessionID: sessionID
                )
                switch gate.check {
                case .reject:
                    approvalEngine.recordOutcome(
                        action: action,
                        risk: (try? TesseraActionVerifier.ruleBasedRisk(for: action)) ?? .medium,
                        sandboxed: sandboxEnforceable,
                        decision: .reject,
                        userChoice: .none,
                        source: gate.source,
                        sessionID: sessionID
                    )
                    let blocked = ToolResult.fail("Blocked by the safety policy before execution.")
                    continuation.yield(.toolResult(name: call.name, result: blocked))
                    messages.append(LLMMessage(role: "tool", content: "Tool '\(call.name)' was blocked by the safety policy."))
                    if approvalEngine.circuitBreaker.isTripped {
                        continuation.yield(.error("Interrupted: the denial circuit-breaker tripped (too many blocked actions)."))
                        return
                    }
                    continue
                case .askUser:
                    let approved = await approvalEngine.requestApprovalForced(
                        toolName: call.name,
                        arguments: call.arguments
                    )
                    let risk = (try? TesseraActionVerifier.ruleBasedRisk(for: action)) ?? .medium
                    approvalEngine.recordOutcome(
                        action: action,
                        risk: risk,
                        sandboxed: sandboxEnforceable,
                        decision: .askUser,
                        userChoice: approved ? .approved : .denied,
                        source: gate.source,
                        sessionID: sessionID
                    )
                    guard approved else {
                        approvalEngine.circuitBreaker.record(denied: true)
                        let denied = ToolResult.fail("Denied by user")
                        continuation.yield(.toolResult(name: call.name, result: denied))
                        messages.append(LLMMessage(role: "tool", content: "Tool '\(call.name)' was denied by the user."))
                        if approvalEngine.circuitBreaker.isTripped {
                            continuation.yield(.error("Interrupted: the denial circuit-breaker tripped (too many denied actions)."))
                            return
                        }
                        continue
                    }
                case .autoApprove:
                    approvalEngine.recordOutcome(
                        action: action,
                        risk: (try? TesseraActionVerifier.ruleBasedRisk(for: action)) ?? .medium,
                        sandboxed: sandboxEnforceable,
                        decision: .autoApprove,
                        userChoice: .none,
                        source: gate.source,
                        sessionID: sessionID
                    )
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

    private func buildSystemPrompt(userMessage: String) -> String {
        var prompt = """
        You are Tessera Studio Agent, an assistant for quantizing, calibrating,
        and deploying LLMs with the Tessera engine. You help users manage models,
        run calibration, evolve quantization policies, and evaluate results.

        \(registry.systemPromptToolsBlock())
        """
        // On-demand skills (absorption I1): inject any skill whose manifest
        // matches the user's message. Empty until the user authors skills.
        let skills = skillLoader.systemPromptFragment(for: userMessage)
        if !skills.isEmpty {
            prompt += "\n\n# Relevant skills\n\n" + skills
        }
        return prompt
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

/// Last-resort LLM that echoes the user message and recognizes a few tool
/// keywords. The factory (TesseraLLMProviderFactory.makeFromSettings) upgrades
/// to the on-device provider as soon as a model is available, so this is only
/// reached when the library is genuinely empty - it keeps the app responsive
/// instead of crashing on a missing model.
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
