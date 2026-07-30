import Foundation

/// Registry of available tools. Provides lookup by name and
/// enumerates all registered tools for the LLM system prompt.
final class TesseraToolRegistry: Sendable {
    private let tools: [String: any TesseraTool]

    init(tools: [any TesseraTool]) {
        var map: [String: any TesseraTool] = [:]
        for tool in tools {
            map[tool.name] = tool
        }
        self.tools = map
    }

    func tool(named name: String) -> (any TesseraTool)? {
        tools[name]
    }

    var allTools: [any TesseraTool] {
        Array(tools.values).sorted { $0.name < $1.name }
    }

    /// Builds the tool descriptions block for the LLM system prompt.
    func systemPromptToolsBlock() -> String {
        var lines: [String] = ["You have access to the following tools:", ""]
        for tool in allTools {
            lines.append("## \(tool.name)")
            lines.append(tool.description)
            lines.append("Parameters: \(tool.parameters.toJSON())")
            lines.append("")
        }
        return lines.joined(separator: "\n")
    }

    /// The default registry with all 8 v1 tools.
    static let `default` = TesseraToolRegistry(tools: [
        ListModelsTool(),
        LoadModelTool(),
        InspectSidecarTool(),
        CalibrateTool(),
        EvolveTool(),
        QuantizeTool(),
        ConvertTool(),
        EvaluateTool(),
    ])
}
