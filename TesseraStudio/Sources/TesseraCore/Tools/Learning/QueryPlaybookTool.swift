import Foundation

/// Retrieves reasoning strategies recorded for a problem class.
public struct QueryPlaybookTool: TesseraTool {
    public let name = "query_playbook"
    public let description = "Retrieve reasoning strategies recorded for a problem class from the reasoning playbook."
    public let defaultApprovalLevel = ApprovalLevel.auto

    public let parameters = JSONSchema(
        type: "object",
        properties: [
            "problem_class": SchemaProperty(
                type: "string",
                description: "The problem class to look up, e.g. \"failing-test-resolution\"."
            ),
        ],
        required: ["problem_class"]
    )

    public init() {}

    public func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        guard let problemClass = arguments["problem_class"]?.stringValue, !problemClass.isEmpty else {
            return .fail("problem_class is required")
        }

        let strategies = TesseraLearningCenter.shared.playbook.strategies(forProblemClass: problemClass)
        if strategies.isEmpty {
            return .ok("No playbook strategies for \"\(problemClass)\" yet.", data: ["strategies": .number(0)])
        }

        let body = strategies.map { "- \($0)" }.joined(separator: "\n")
        return .ok("Playbook strategies for \"\(problemClass)\":\n\(body)", data: [
            "strategies": .number(Double(strategies.count)),
        ])
    }
}
