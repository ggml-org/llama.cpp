import Foundation

/// Triggers a background adaptation step now. Defaults to a dry run that
/// reports what would fire without performing any training.
public struct RunAdaptationTool: TesseraTool {
    public let name = "run_adaptation"
    public let description = "Trigger a background adaptation step now. Defaults to a dry run that performs no training."
    public let defaultApprovalLevel = ApprovalLevel.notify

    public let parameters = JSONSchema(
        type: "object",
        properties: [
            "dry_run": SchemaProperty(
                type: "boolean",
                description: "If true (default), report what would run without training.",
                defaultValue: "true"
            ),
        ],
        required: []
    )

    public init() {}

    public func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        let dryRun = Self.parseBool(arguments["dry_run"]) ?? true
        do {
            let receipt = try await TesseraLearningCenter.shared.scheduler.runAdaptation(dryRun: dryRun)
            return .ok(receipt.summary, data: ["dry_run": .bool(dryRun)])
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
