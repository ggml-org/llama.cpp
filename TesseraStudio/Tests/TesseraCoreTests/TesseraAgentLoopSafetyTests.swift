import XCTest
@testable import TesseraCore

@MainActor
final class TesseraAgentLoopSafetyTests: XCTestCase {

    /// A provider that always emits one tool call, so the loop keeps cycling
    /// until the denial circuit-breaker interrupts it (rather than the model
    /// deciding to stop).
    private struct RepeatToolProvider: LLMProvider {
        let toolName: String
        func complete(system: String, messages: [LLMMessage], tools: [ToolDescriptor]) async throws -> LLMResponse {
            LLMResponse(content: "", toolCalls: [LLMToolCall(name: toolName, arguments: [:])], tokenCount: 1)
        }
    }

    /// A deterministic tool that always succeeds, so tests can observe whether
    /// the loop reached execution without depending on the FFI/model directory.
    /// The name starts with "list" so the rule-based verifier rates it low risk.
    private struct StubTool: TesseraTool {
        let name: String
        let defaultApprovalLevel: ApprovalLevel
        let description = "stub tool"
        let parameters = JSONSchema()
        func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
            .ok("executed \(name)")
        }
    }

    func testDeniedToolIsBlockedAndBreakerInterruptsLoop() async {
        let approval = TesseraApprovalEngine()
        approval.setOverride(.denied, for: "list_models")
        defer { approval.clearOverride(for: "list_models") }

        let loop = TesseraAgentLoop(
            registry: .default,
            approvalEngine: approval,
            llmProvider: RepeatToolProvider(toolName: "list_models"),
            maxIterations: 10
        )

        var blockedCount = 0
        var sawBreakerError = false
        for await event in loop.run(userMessage: "list models", history: []) {
            switch event {
            case .toolResult(_, let result):
                if !result.success { blockedCount += 1 }
            case .error(let message):
                if message.contains("circuit-breaker") { sawBreakerError = true }
            default:
                break
            }
        }

        // The safety spine rejects the disabled tool before execution, and the
        // breaker trips on the third consecutive denial - so exactly three
        // blocked results land before the loop is interrupted (not ten).
        XCTAssertEqual(blockedCount, 3)
        XCTAssertTrue(sawBreakerError)
    }

    /// A contained, low-risk action under an auto policy clears the spine as
    /// `autoApprove` and runs every iteration without ever presenting a sheet.
    func testAutoApproveRunsWithoutPrompting() async {
        let registry = TesseraToolRegistry(tools: [StubTool(name: "list_items", defaultApprovalLevel: .auto)])
        let approval = TesseraApprovalEngine()
        approval.setOverride(.auto, for: "list_items")
        defer { approval.clearOverride(for: "list_items") }

        let loop = TesseraAgentLoop(
            registry: registry,
            approvalEngine: approval,
            llmProvider: RepeatToolProvider(toolName: "list_items"),
            maxIterations: 3,
            sandboxEnforceable: true
        )

        var executed = 0
        var blocked = 0
        for await event in loop.run(userMessage: "go", history: []) {
            switch event {
            case .toolResult(_, let result):
                if result.success { executed += 1 } else { blocked += 1 }
            default:
                break
            }
        }

        XCTAssertEqual(executed, 3)
        XCTAssertEqual(blocked, 0)
        XCTAssertNil(approval.pendingRequest)
    }

    /// The discriminating case for the three-outcome wiring: the tool is
    /// generally auto-approved, but this action is not sandbox-contained, so
    /// the spine returns `askUser`. The loop must force a REAL prompt rather
    /// than fall through to the auto policy. Denying every prompt feeds the
    /// breaker, which trips on the third denial and interrupts the loop.
    func testAskUserForcesPromptAndUserDenialTripsBreaker() async {
        let registry = TesseraToolRegistry(tools: [StubTool(name: "list_items", defaultApprovalLevel: .auto)])
        let approval = TesseraApprovalEngine()
        approval.setOverride(.auto, for: "list_items")
        defer { approval.clearOverride(for: "list_items") }

        // Resolve every forced prompt with a denial, off the loop's own path.
        let resolver = Task { @MainActor in
            while !Task.isCancelled {
                if approval.pendingRequest != nil {
                    approval.resolvePending(approved: false)
                }
                try? await Task.sleep(nanoseconds: 1_000_000)
            }
        }
        defer { resolver.cancel() }

        let loop = TesseraAgentLoop(
            registry: registry,
            approvalEngine: approval,
            llmProvider: RepeatToolProvider(toolName: "list_items"),
            maxIterations: 10,
            sandboxEnforceable: false
        )

        var executed = 0
        var denied = 0
        var sawBreakerError = false
        for await event in loop.run(userMessage: "go", history: []) {
            switch event {
            case .toolResult(_, let result):
                if result.success { executed += 1 } else { denied += 1 }
            case .error(let message):
                if message.contains("circuit-breaker") { sawBreakerError = true }
            default:
                break
            }
        }

        // The forced prompt is honored (nothing executes) and three consecutive
        // user denials trip the breaker - so exactly three denied results land
        // before the loop is interrupted (not ten).
        XCTAssertEqual(executed, 0)
        XCTAssertEqual(denied, 3)
        XCTAssertTrue(sawBreakerError)
    }
}
