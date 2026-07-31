import Foundation

/// Tier-2 preview: scrub secrets from text and show what WOULD be sent to a
/// teacher. Performs no egress. The aggressiveness dial is surfaced and
/// reported; tuning the scrubber per level lands with the full tier-2
/// anonymizer pipeline in Phase 5.
public struct AnonymizeWorktreeTool: TesseraTool {
    public let name = "anonymize_worktree"
    public let description = "Tier 2 preview: scrub secrets from text and show what would be sent to a teacher. Performs no egress."
    public let defaultApprovalLevel = ApprovalLevel.prompt

    public let parameters = JSONSchema(
        type: "object",
        properties: [
            "text": SchemaProperty(
                type: "string",
                description: "The content to scrub before a hypothetical tier-2 escalation."
            ),
            "aggressiveness": SchemaProperty(
                type: "string",
                description: "Scrub aggressiveness.",
                enumValues: ["light", "balanced", "aggressive"],
                defaultValue: TesseraSettings.learningAnonymizerAggressiveness
            ),
        ],
        required: ["text"]
    )

    public init() {}

    public func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        guard let text = arguments["text"]?.stringValue, !text.isEmpty else {
            return .fail("text is required")
        }
        let aggressiveness = arguments["aggressiveness"]?.stringValue ?? TesseraSettings.learningAnonymizerAggressiveness
        let scrubbed = TesseraLearningCenter.shared.curation.scrub(text)

        let output = """
            Anonymizer preview (aggressiveness: \(aggressiveness)). No egress performed.
            Original length: \(text.count) chars; scrubbed length: \(scrubbed.count) chars.

            --- scrubbed preview (what WOULD be sent) ---
            \(scrubbed)
            """
        return .ok(output, data: [
            "aggressiveness": .string(aggressiveness),
            "original_chars": .number(Double(text.count)),
            "scrubbed_chars": .number(Double(scrubbed.count)),
        ])
    }
}
