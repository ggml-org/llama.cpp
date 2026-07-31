import Foundation

/// Records a verifiable world outcome (build/test/commit/revert), feeds it to
/// curation, and - when it verifies a teacher proposal - records the trial
/// against the world gate.
public struct RecordOutcomeTool: TesseraTool {
    public let name = "record_outcome"
    public let description = "Record a verifiable world outcome (build, test, commit, or revert) as learning signal."
    public let defaultApprovalLevel = ApprovalLevel.notify

    public let parameters = JSONSchema(
        type: "object",
        properties: [
            "kind": SchemaProperty(
                type: "string",
                description: "The kind of world signal.",
                enumValues: TesseraWorldOutcomeKind.allCases.map(\.rawValue)
            ),
            "success": SchemaProperty(
                type: "boolean",
                description: "Whether the outcome succeeded."
            ),
            "detail": SchemaProperty(
                type: "string",
                description: "Optional human-readable detail."
            ),
            "proposal_id": SchemaProperty(
                type: "string",
                description: "Optional teacher proposal id this outcome verifies."
            ),
        ],
        required: ["kind", "success"]
    )

    public init() {}

    public func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        guard let kindRaw = arguments["kind"]?.stringValue,
              let kind = TesseraWorldOutcomeKind(rawValue: kindRaw) else {
            return .fail("kind is required and must be one of: build, test, commit, revert")
        }
        guard let success = Self.parseBool(arguments["success"]) else {
            return .fail("success is required (true or false)")
        }

        let detail = arguments["detail"]?.stringValue ?? ""
        let proposalId = arguments["proposal_id"]?.stringValue
        let outcome = TesseraWorldOutcome(kind: kind, success: success, detail: detail, proposalId: proposalId)

        let center = TesseraLearningCenter.shared
        do {
            let receipt = try await center.worldSignals.record(outcome)
            // Best-effort curation enrichment; a curation failure does not
            // fail the outcome record.
            _ = try? await center.curation.ingest(outcome: outcome)

            // If this outcome verifies a teacher proposal, resolve the teacher
            // that produced it via the proposal registry and record the trial
            // against the world gate. Proposals the registry does not know
            // (e.g. recorded before the registry existed) fall back to the
            // "unknown" bucket.
            var attributedTeacher = ""
            if let proposalId, !proposalId.isEmpty {
                let teacherId = TesseraProposalRegistry.shared.teacherId(forProposalId: proposalId) ?? "unknown"
                attributedTeacher = teacherId
                let attributed = TesseraTeacherProposal(id: proposalId, teacherId: teacherId, reasoning: "")
                try? center.assessor.recordTrial(proposal: attributed, passedWorldGate: success)
            }

            return .ok(receipt.summary, data: [
                "outcome_id": .string(outcome.id),
                "kind": .string(kind.rawValue),
                "success": .bool(success),
                "attributed_teacher": .string(attributedTeacher),
            ])
        } catch {
            return .fail(error.localizedDescription)
        }
    }

    private static func parseBool(_ value: JSONValue?) -> Bool? {
        guard let value else { return nil }
        if case .bool(let b) = value { return b }
        if let s = value.stringValue { return s == "true" || s == "1" }
        if let n = value.numberValue { return n != 0 }
        return nil
    }
}
