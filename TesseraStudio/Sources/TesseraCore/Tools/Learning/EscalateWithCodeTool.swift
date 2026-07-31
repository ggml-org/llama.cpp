import Foundation

/// Tier-2 code escalation: send an anonymized worktree to a teacher. Highest
/// sensitivity. v1 keeps the tool surface and approval real but does not
/// perform egress - it fails honestly once the egress guards pass.
public struct EscalateWithCodeTool: TesseraTool {
    public let name = "escalate_with_code"
    public let description = "Tier 2: send an anonymized worktree to a teacher. Highest sensitivity; requires the anonymizer pipeline."
    public let defaultApprovalLevel = ApprovalLevel.prompt

    public let parameters = JSONSchema(
        type: "object",
        properties: [
            "problem_class": SchemaProperty(
                type: "string",
                description: "The problem class being escalated."
            ),
            "summary": SchemaProperty(
                type: "string",
                description: "Natural-language problem frame."
            ),
        ],
        required: ["problem_class", "summary"]
    )

    public init() {}

    public func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        if !TesseraSettings.learningEnabled || !TesseraSettings.learningEscalationEnabled {
            return .fail("Escalation egress is disabled (enable learning + escalation in settings).")
        }
        if TesseraLearningCenter.shared.escalation.availableTeachers().isEmpty {
            return .fail("No escalation teachers configured (set learning.teachers).")
        }
        // Guards passed. Tier-2 code egress is not implemented in v1.
        return .fail("Tier-2 code escalation (anonymizer pipeline) is not implemented yet; use escalate_reasoning (tier 1).")
    }
}
