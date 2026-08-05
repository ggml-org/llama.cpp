import Foundation

/// Tier-1 escalation: send a natural-language frame plus a structured
/// diagnostic envelope to every configured teacher and collect their
/// reasoning. No source code crosses the boundary at tier 1.
public struct EscalateReasoningTool: TesseraTool {
    public let name = "escalate_reasoning"
    public let description = "Tier 1: send a problem frame and diagnostic envelope to the teacher ensemble and collect reasoning. No source code is sent."
    public let defaultApprovalLevel = ApprovalLevel.prompt

    public let parameters = JSONSchema(
        type: "object",
        properties: [
            "problem_class": SchemaProperty(
                type: "string",
                description: "The problem class, e.g. \"failing-test-resolution\"."
            ),
            "summary": SchemaProperty(
                type: "string",
                description: "Natural-language problem frame."
            ),
            "observed_vs_expected": SchemaProperty(
                type: "string",
                description: "What happened versus what was expected."
            ),
            "failing_tests": SchemaProperty(
                type: "array",
                description: "Names of failing tests."
            ),
            "redacted_errors": SchemaProperty(
                type: "array",
                description: "Error text already scrubbed of secrets."
            ),
            "stack_shape": SchemaProperty(
                type: "string",
                description: "Type / stack shape, no source."
            ),
        ],
        required: ["problem_class", "summary"]
    )

    public init() {}

    public func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        if !TesseraSettings.learningEnabled || !TesseraSettings.learningEscalationEnabled {
            return .fail("Escalation egress is disabled (enable learning + escalation in settings).")
        }
        let center = TesseraLearningCenter.shared
        if center.escalation.availableTeachers().isEmpty {
            return .fail("No escalation teachers configured (set learning.teachers).")
        }

        guard let problemClass = arguments["problem_class"]?.stringValue, !problemClass.isEmpty else {
            return .fail("problem_class is required")
        }
        guard let summary = arguments["summary"]?.stringValue, !summary.isEmpty else {
            return .fail("summary is required")
        }

        let frame = TesseraEscalationFrame(
            problemClass: problemClass,
            summary: summary,
            observedVsExpected: arguments["observed_vs_expected"]?.stringValue ?? "",
            failingTests: Self.stringArray(arguments["failing_tests"]),
            redactedErrors: Self.stringArray(arguments["redacted_errors"]),
            stackShape: arguments["stack_shape"]?.stringValue ?? ""
        )

        do {
            let result = try await center.escalation.escalate(frame: frame)
            var responded: [String] = []
            for proposal in result.proposals where !responded.contains(proposal.teacherId) {
                responded.append(proposal.teacherId)
            }
            let teachers = responded.joined(separator: ", ")
            // Surface each proposal's id (paired with its teacher) so a later
            // record_outcome can pass it as proposal_id and the trial lands on
            // the teacher that produced it instead of the "unknown" bucket.
            let ids = result.proposals.map { "\($0.id) (\($0.teacherId))" }.joined(separator: ", ")
            let output = "Escalated to \(result.fannedOutTo.count) teacher(s); received \(result.proposals.count) proposal(s) from: \(teachers.isEmpty ? "none" : teachers). Proposal ids: \(ids.isEmpty ? "none" : ids). Pass a proposal id to record_outcome to attribute the verifying outcome to its teacher."
            return .ok(output, data: [
                "proposals": .number(Double(result.proposals.count)),
                "teachers": .string(teachers),
                "proposals_detail": .array(result.proposals.map { proposal in
                    .object(["id": .string(proposal.id), "teacher_id": .string(proposal.teacherId)])
                }),
            ])
        } catch {
            return .fail(error.localizedDescription)
        }
    }

    private static func stringArray(_ value: JSONValue?) -> [String] {
        guard let value, case .array(let items) = value else { return [] }
        return items.compactMap(\.stringValue)
    }
}
