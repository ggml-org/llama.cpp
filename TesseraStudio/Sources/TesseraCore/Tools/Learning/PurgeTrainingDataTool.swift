import Foundation

/// Deletes harvested learning data across every purgeable store. Destructive;
/// gated behind explicit approval.
public struct PurgeTrainingDataTool: TesseraTool {
    public let name = "purge_training_data"
    public let description = "Delete harvested learning data across all stores (outcomes, playbook, reference, assessments). Destructive."
    public let defaultApprovalLevel = ApprovalLevel.prompt

    public let parameters = JSONSchema()

    public init() {}

    public func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        do {
            let receipt = try TesseraLearningCenter.shared.purgeAll()
            return .ok(receipt.summary, data: receipt.payload)
        } catch {
            return .fail(error.localizedDescription)
        }
    }
}
