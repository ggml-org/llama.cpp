import Foundation

/// Registry of available tools. Provides lookup by name and
/// enumerates all registered tools for the LLM system prompt.
public final class TesseraToolRegistry: Sendable {
    private let tools: [String: any TesseraTool]

    public init(tools: [any TesseraTool]) {
        var map: [String: any TesseraTool] = [:]
        for tool in tools {
            map[tool.name] = tool
        }
        self.tools = map
    }

    public func tool(named name: String) -> (any TesseraTool)? {
        tools[name]
    }

    public var allTools: [any TesseraTool] {
        Array(tools.values).sorted { $0.name < $1.name }
    }

    /// Builds the tool descriptions block for the LLM system prompt.
    public func systemPromptToolsBlock() -> String {
        var lines: [String] = ["You have access to the following tools:", ""]
        for tool in allTools {
            lines.append("## \(tool.name)")
            lines.append(tool.description)
            lines.append("Parameters: \(tool.parameters.toJSON())")
            lines.append("")
        }
        return lines.joined(separator: "\n")
    }

    /// The default registry: the 8 v1 tools plus the 9 learning tools
    /// plus the 9 Python-tool wrappers under tools/tessera/.
    public static let `default` = TesseraToolRegistry(tools: [
        ListModelsTool(),
        LoadModelTool(),
        InspectSidecarTool(),
        CalibrateTool(),
        EvolveTool(),
        QuantizeTool(),
        ConvertTool(),
        EvaluateTool(),
        // General-agent harness: cited web research. Keyless DuckDuckGo search
        // by default, SearXNG/Tavily opt-in (docs/tessera-studio-design.md 5.4).
        // Egresses the query to a search engine, so it runs at approval .prompt.
        TesseraResearchTool(),
        // Self-improving learning loop (docs/self-improving-loop-design.md)
        LookupDocsTool(),
        QueryPlaybookTool(),
        RecordOutcomeTool(),
        EscalateReasoningTool(),
        AnonymizeWorktreeTool(),
        EscalateWithCodeTool(),
        RunAdaptationTool(),
        CollectTrainingTracesTool(),
        RunTrainingTool(),
        InspectLearningTool(),
        PurgeTrainingDataTool(),
        // Python tooling surface (docs/tessera-studio-design.md 2.3): the
        // calibration / DB / evidence Python scripts are first-class tools.
        // The Library view consumes tessera_db_query for its model grid.
        AWQEvolveTool(),
        BackfillTool(),
        EvidenceStoreSummarizeTool(),
        L3HessianTraceTool(),
        MultimodalCalibrateTool(),
        PerTensorCalibrateTool(),
        ShadowCalibrateTool(),
        TesseraDBQueryTool(),
        UnifiedCalibrateTool(),
    ])
}
