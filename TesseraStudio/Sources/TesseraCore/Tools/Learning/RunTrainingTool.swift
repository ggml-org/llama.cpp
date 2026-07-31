import Foundation

/// Runs one drafter training cycle: prepare a dataset from accumulated
/// traces, fine-tune a LoRA adapter, and export the merged model. Heavy and
/// mutating, so it defaults to a dry run and gates behind explicit approval.
public struct RunTrainingTool: TesseraTool {
    public let name = "run_training"
    public let description = "Run one drafter training cycle: prepare dataset from accumulated traces, fine-tune a LoRA adapter, and export the merged model. Requires learning.baseModelPath to be set."
    public let defaultApprovalLevel = ApprovalLevel.prompt

    public let parameters = JSONSchema(
        type: "object",
        properties: [
            "dry_run": SchemaProperty(
                type: "boolean",
                description: "If true (default), record what would run without training.",
                defaultValue: "true"
            ),
        ],
        required: []
    )

    public init() {}

    public func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        guard let orchestrator = TesseraLearningCenter.shared.training else {
            return .fail("Training orchestrator is not installed.")
        }
        // Absent param falls back to the configured default, so a manual call
        // without arguments honors learning.trainingDryRun.
        let dryRun = Self.parseBool(arguments["dry_run"]) ?? TesseraSettings.learningTrainingDryRun
        let record = await orchestrator.run(overrideDryRun: dryRun)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let json = (try? encoder.encode(record)).flatMap { String(data: $0, encoding: .utf8) } ?? "{}"

        return .ok(json, data: [
            "outcome": .string(record.outcome.rawValue),
            "trace_count": .number(Double(record.traceCount)),
            "dry_run": .bool(dryRun),
        ])
    }

    private static func parseBool(_ value: JSONValue?) -> Bool? {
        guard let value else { return nil }
        if case .bool(let b) = value { return b }
        if let s = value.stringValue { return s == "true" || s == "1" }
        if let n = value.numberValue { return n != 0 }
        return nil
    }
}
