import Foundation

/// Tier-2 code escalation: anonymize a code payload and fan it out to the
/// assessment-weighted teacher ensemble. Highest sensitivity. The payload is
/// run through the symbol-level C++ anonymizer (TesseraAnonymizerService,
/// design Phase 5) and the local de-anonymization map is persisted under an
/// escalation id so a teacher's answer can be de-anonymized later. When the
/// anonymizer binary is unavailable the service degrades honestly to
/// curation.scrub and reports the fallback.
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

        // Symbol-level anonymization (design Phase 5). The service degrades to
        // curation.scrub and flags it when the binary is missing or fails - it
        // never reports a symbol-level pass it did not actually run.
        let anonymizer = TesseraAnonymizerService()
        let anonymized = await anonymizer.anonymize(code)

        // Persist the de-anonymization map under an escalation id so a teacher's
        // answer can be reversed locally later. The map is the local-only key;
        // it is empty (and skipped) on a scrub fallback.
        let escalationId = UUID().uuidString
        if !anonymized.map.isEmpty {
            try? anonymizer.persistMap(anonymized.map, forEscalation: escalationId)
        }

        do {
            let result = try await service.escalateWithCode(frame: frame, anonymizedPayload: anonymized.text)
            var responded: [String] = []
            for proposal in result.proposals where !responded.contains(proposal.teacherId) {
                responded.append(proposal.teacherId)
            }
            let teachers = responded.joined(separator: ", ")

            let method: String
            if anonymized.usedFallback {
                method = "curation-scrub fallback (\(anonymized.note)); no de-anonymization map"
            } else {
                method = "symbol-level anonymizer (level: \(anonymized.level)); \(anonymized.map.count) symbol(s) mapped, escalation id \(escalationId)"
            }
            // Surface each proposal's id (paired with its teacher) so a later
            // record_outcome can pass it as proposal_id and the trial lands on
            // the teacher that produced it instead of the "unknown" bucket.
            let ids = result.proposals.map { "\($0.id) (\($0.teacherId))" }.joined(separator: ", ")
            let output = "Tier-2 escalation to \(result.fannedOutTo.count) teacher(s); received \(result.proposals.count) proposal(s) from: \(teachers.isEmpty ? "none" : teachers). Payload prepared with \(method). (\(code.count) -> \(anonymized.text.count) chars). Proposal ids: \(ids.isEmpty ? "none" : ids). Pass a proposal id to record_outcome to attribute the verifying outcome to its teacher."
            return .ok(output, data: [
                "proposals": .number(Double(result.proposals.count)),
                "teachers": .string(teachers),
                "proposals_detail": .array(result.proposals.map { proposal in
                    .object(["id": .string(proposal.id), "teacher_id": .string(proposal.teacherId)])
                }),
                "anonymizer": .string(anonymized.anonymizer),
                "level": .string(anonymized.level),
                "used_fallback": .bool(anonymized.usedFallback),
                "escalation_id": .string(escalationId),
                "mapped_symbols": .number(Double(anonymized.map.count)),
                "original_chars": .number(Double(code.count)),
                "anonymized_chars": .number(Double(anonymized.text.count)),
            ])
        } catch {
            return .fail(error.localizedDescription)
        }
    }
}
