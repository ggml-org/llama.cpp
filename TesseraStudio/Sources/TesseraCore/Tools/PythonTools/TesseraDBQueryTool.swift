import Foundation

/// Tessera `tessera_db_query.py` wrapped as a `TesseraTool`. The
/// Library view calls this with `{"query": "list_models"}` to populate
/// the model grid; downstream analytics tools use `list_tensor_stats`,
/// `table_names`, and `query` for read-only inspection.
public struct TesseraDBQueryTool: TesseraTool, Sendable {
    public let tool: PythonTool

    public init() {
        // swiftlint:disable:next force_try
        self.tool = try! PythonTool(scriptName: "tessera_db_query")
    }

    public var name: String { tool.name }
    public var description: String { tool.description }
    public var parameters: JSONSchema { tool.parameters }
    public var defaultApprovalLevel: ApprovalLevel { tool.defaultApprovalLevel }

    public func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        try await tool.execute(arguments: arguments)
    }
}
