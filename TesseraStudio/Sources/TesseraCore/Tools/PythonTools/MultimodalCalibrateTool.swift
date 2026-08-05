import Foundation

/// Tessera `multimodal_calibrate.py` wrapped as a `TesseraTool`.
public struct MultimodalCalibrateTool: TesseraTool, Sendable {
    public let tool: PythonTool

    public init() {
        // The schema sidecar at tools/tessera/multimodal_calibrate.schema.json
        // is the source of truth for the parameter list.
        // swiftlint:disable:next force_try
        self.tool = try! PythonTool(scriptName: "multimodal_calibrate")
    }

    public var name: String { tool.name }
    public var description: String { tool.description }
    public var parameters: JSONSchema { tool.parameters }
    public var defaultApprovalLevel: ApprovalLevel { tool.defaultApprovalLevel }

    public func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        try await tool.execute(arguments: arguments)
    }
}
