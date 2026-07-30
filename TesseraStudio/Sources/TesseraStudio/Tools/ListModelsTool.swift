import Foundation

/// Lists available models in the Tessera model directory.
struct ListModelsTool: TesseraTool {
    let name = "list_models"
    let description = "List all available GGUF and .mlmodelc models in the Tessera model directory."
    let defaultApprovalLevel = ApprovalLevel.auto

    let parameters = JSONSchema(
        type: "object",
        properties: [
            "directory": SchemaProperty(
                type: "string",
                description: "Optional directory to scan. Defaults to ~/Models/tessera.",
                defaultValue: "~/Models/tessera"
            ),
            "filter": SchemaProperty(
                type: "string",
                description: "Optional substring filter on model name."
            ),
        ],
        required: []
    )

    func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        let dir = arguments["directory"]?.stringValue ?? "~/Models/tessera"
        let filter = arguments["filter"]?.stringValue
        let expandedDir = NSString(string: dir).expandingTildeInPath

        let fm = FileManager.default
        guard fm.fileExists(atPath: expandedDir) else {
            return .fail("Directory not found: \(expandedDir)")
        }

        let contents = try fm.contentsOfDirectory(atPath: expandedDir)
        var models = contents.filter {
            $0.hasSuffix(".gguf") || $0.hasSuffix(".mlmodelc")
        }
        if let filter, !filter.isEmpty {
            models = models.filter { $0.localizedCaseInsensitiveContains(filter) }
        }

        guard !models.isEmpty else {
            return .ok("No models found in \(expandedDir)")
        }

        var lines = ["Found \(models.count) model(s) in \(expandedDir):", ""]
        for m in models.sorted() {
            let path = (expandedDir as NSString).appendingPathComponent(m)
            let attrs = try? fm.attributesOfItem(atPath: path)
            let size = (attrs?[.size] as? Int64) ?? 0
            let sizeStr = ByteCountFormatter.string(fromByteCount: size, countStyle: .file)
            let kind = m.hasSuffix(".mlmodelc") ? "CoreML" : "GGUF"
            lines.append("  \(m)  [\(kind), \(sizeStr)]")
        }

        return .ok(lines.joined(separator: "\n"), data: [
            "count": .number(Double(models.count)),
            "directory": .string(expandedDir),
        ])
    }
}
