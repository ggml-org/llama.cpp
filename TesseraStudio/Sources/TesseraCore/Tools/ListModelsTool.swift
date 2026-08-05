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

    private let shell: TesseraProcessShell

    public init() {
        self.shell = ProcessRunner()
    }

    /// Test seam.
    init(shell: TesseraProcessShell) {
        self.shell = shell
    }

    public func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        let dir = arguments["directory"]?.stringValue ?? "~/Models/tessera"
        let filter = arguments["filter"]?.stringValue
        let expandedDir = NSString(string: dir).expandingTildeInPath

        let fm = FileManager.default
        guard fm.fileExists(atPath: expandedDir) else {
            return .fail("Directory not found: \(expandedDir)")
        }

        // Prefer the linked FFI lister; the stub returns .fallbackToCLI so
        // the SwiftPM build path goes through the local directory walk.
        if TesseraFFIBridge.isAvailable {
            switch TesseraFFIBridge.listModels(dir: expandedDir) {
            case let .success(output):
                if let arr = EngineToolSupport.parseJSONArray(stdout: output) {
                    return formatList(arr: arr, dir: expandedDir, filter: filter, backend: "ffi")
                }
                return .ok("List complete.\n\(output)", data: [
                    "directory": .string(expandedDir),
                    "backend": .string("ffi"),
                ])
            case .fallbackToCLI:
                break
            case .error:
                break
            }
        }

        // CLI fallback: tessera-cli list-models <dir>
        if let cli = TesseraCLIBinaryResolver.resolve(
            override: TesseraSettings.tesseraCLIPath,
            settingsKey: TesseraSettingsKey.tesseraCLIPath
        ) {
            let result = try await shell.run(
                executable: cli,
                arguments: ["list-models", expandedDir],
                environment: nil,
                workingDirectory: nil
            )
            if result.exitCode == 0,
               let arr = EngineToolSupport.parseJSONArray(stdout: result.stdout) {
                return formatList(arr: arr, dir: expandedDir, filter: filter, backend: "cli")
            }
        }

        // Last-resort local directory walk: the same contract the old code
        // shipped, used when neither the FFI nor a tessera-cli is on disk.
        return localDirectoryList(dir: expandedDir, filter: filter)
    }

    private func formatList(arr: [Any], dir: String, filter: String?, backend: String) -> ToolResult {
        var models: [String] = []
        for entry in arr {
            // Accept either a bare string array or objects with a "name" key.
            if let s = entry as? String {
                models.append(s)
            } else if let d = entry as? [String: Any] {
                if let n = d["name"] as? String { models.append(n) }
                else if let p = d["path"] as? String { models.append((p as NSString).lastPathComponent) }
            }
        }
        if let f = filter, !f.isEmpty {
            models = models.filter { $0.localizedCaseInsensitiveContains(f) }
        }
        guard !models.isEmpty else {
            return .ok("No models found in \(dir)", data: [
                "count": .number(0),
                "directory": .string(dir),
                "backend": .string(backend),
            ])
        }
        let lines = ["Found \(models.count) model(s) in \(dir):", ""]
            + models.sorted().map { "  \($0)" }
        return .ok(lines.joined(separator: "\n"), data: [
            "count": .number(Double(models.count)),
            "directory": .string(dir),
            "backend": .string(backend),
        ])
    }

    private func localDirectoryList(dir: String, filter: String?) -> ToolResult {
        let fm = FileManager.default
        guard let contents = try? fm.contentsOfDirectory(atPath: dir) else {
            return .fail("Cannot read directory: \(dir)")
        }
        var models = contents.filter {
            $0.hasSuffix(".gguf") || $0.hasSuffix(".mlmodelc")
        }
        if let f = filter, !f.isEmpty {
            models = models.filter { $0.localizedCaseInsensitiveContains(f) }
        }
        guard !models.isEmpty else {
            return .ok("No models found in \(dir)")
        }
        var lines = ["Found \(models.count) model(s) in \(dir):", ""]
        for m in models.sorted() {
            let path = (dir as NSString).appendingPathComponent(m)
            let attrs = try? fm.attributesOfItem(atPath: path)
            let size = (attrs?[.size] as? Int64) ?? 0
            let sizeStr = ByteCountFormatter.string(fromByteCount: size, countStyle: .file)
            let kind = m.hasSuffix(".mlmodelc") ? "CoreML" : "GGUF"
            lines.append("  \(m)  [\(kind), \(sizeStr)]")
        }
        return .ok(lines.joined(separator: "\n"), data: [
            "count": .number(Double(models.count)),
            "directory": .string(dir),
            "backend": .string("local"),
        ])
    }
}
