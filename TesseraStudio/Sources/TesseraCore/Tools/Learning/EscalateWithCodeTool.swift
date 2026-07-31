import Foundation

/// Tier-2 code escalation: scrub a code payload and fan it out to the
/// assessment-weighted teacher ensemble. Highest sensitivity. The payload is
/// scrubbed with curation.scrub (the best anonymization available now); the
/// full symbol-level anonymizer (the C++ tessera-anonymizer, design Phase 5)
/// upgrades payload quality in a later wave.
public struct EscalateWithCodeTool: TesseraTool {
    public let name = "escalate_with_code"
    public let description = "Tier 2: scrub a code payload and send it to the teacher ensemble. Highest sensitivity; the payload is secret-scrubbed before egress."
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
            "code": SchemaProperty(
                type: "string",
                description: "The code or worktree context to escalate. It is scrubbed before egress."
            ),
        ],
        required: ["problem_class", "summary", "code"]
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
        // escalateWithCode is an addition on the concrete service, not part of
        // the TesseraEscalating spine protocol, so resolve the concrete type.
        guard let service = center.escalation as? TesseraEscalationService else {
            return .fail("Tier-2 escalation requires the concrete TesseraEscalationService.")
        }

        guard let problemClass = arguments["problem_class"]?.stringValue, !problemClass.isEmpty else {
            return .fail("problem_class is required")
        }
        guard let summary = arguments["summary"]?.stringValue, !summary.isEmpty else {
            return .fail("summary is required")
        }
        guard let code = arguments["code"]?.stringValue, !code.isEmpty else {
            return .fail("code is required (the code/context to escalate at tier 2)")
        }

        let frame = TesseraEscalationFrame(problemClass: problemClass, summary: summary)

        // Scrub the payload with the best anonymization available right now.
        // NOTE: curation.scrub is a secret-scrubber, not the full symbol-level
        // anonymizer. The type-preserving symbol anonymizer (a later wave / the
        // C++ tessera-anonymizer, design Phase 5) upgrades payload quality;
        // until then curation.scrub is the scrubber.
        let anonymizedPayload = center.curation.scrub(code)

        do {
            let result = try await service.escalateWithCode(frame: frame, anonymizedPayload: anonymizedPayload)
            var responded: [String] = []
            for proposal in result.proposals where !responded.contains(proposal.teacherId) {
                responded.append(proposal.teacherId)
            }
            let teachers = responded.joined(separator: ", ")
            let output = "Tier-2 escalation to \(result.fannedOutTo.count) teacher(s); received \(result.proposals.count) proposal(s) from: \(teachers.isEmpty ? "none" : teachers). Payload scrubbed with curation.scrub (\(code.count) -> \(anonymizedPayload.count) chars)."
            return .ok(output, data: [
                "proposals": .number(Double(result.proposals.count)),
                "teachers": .string(teachers),
                "original_chars": .number(Double(code.count)),
                "scrubbed_chars": .number(Double(anonymizedPayload.count)),
            ])
        } catch {
            return .fail(error.localizedDescription)
        }
    }
}
