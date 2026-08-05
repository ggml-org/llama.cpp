import Foundation

/// Tessera `shadow-calibrate.py` wrapped as a `TesseraTool`.
public struct ShadowCalibrateTool: TesseraTool, Sendable {
    public let tool: PythonTool

    public init() {
        // swiftlint:disable:next force_try
        self.tool = try! PythonTool(scriptName: "shadow-calibrate")
    }

    public var name: String { tool.name }
    public var description: String { tool.description }
    public var parameters: JSONSchema { tool.parameters }
    public var defaultApprovalLevel: ApprovalLevel { tool.defaultApprovalLevel }

    public func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        try await tool.execute(arguments: arguments)
    }
}
