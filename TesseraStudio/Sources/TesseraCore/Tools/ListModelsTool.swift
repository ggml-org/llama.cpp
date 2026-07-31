import Foundation

/// Lists available models in the Tessera model directory.
public struct ListModelsTool: TesseraTool {
    public let name = "list_models"
    public let description = "List all available GGUF and .mlmodelc models in the Tessera model directory."
    public let defaultApprovalLevel = ApprovalLevel.auto

    public let parameters = JSONSchema(
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

    public init() {}

    public func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        let dir = arguments["directory"]?.stringValue ?? "~/Models/tessera"
        let filter = arguments["filter"]?.stringValue
        let expandedDir = NSString(string: dir).expandingTildeInPath

        if TesseraFFIBridge.isAvailable {
            do {
                var models = try TesseraFFIBridge.listModels(directory: expandedDir)
                if let filter, !filter.isEmpty {
                    models = models.filter { $0.localizedCaseInsensitiveContains(filter) }
                }
                return .ok(models.isEmpty
                    ? "No models found in \(expandedDir)"
                    : "Found \(models.count) model(s) in \(expandedDir):\n\n" + models.sorted().map { "  \($0)" }.joined(separator: "\n"),
                    data: [
                        "count": .number(Double(models.count)),
                        "directory": .string(expandedDir),
                        "backend": .string("ffi"),
                    ])
            } catch {
                // fall through to CLI / direct scan
            }
        }

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
            "backend": .string("cli"),
        ])
    }
}
