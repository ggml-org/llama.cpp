import Foundation

/// Inspects a Tessera sidecar file and reports its contents.
public struct InspectSidecarTool: TesseraTool {
    public let name = "inspect_sidecar"
    public let description = "Read and display the contents of a Tessera calibration-policy sidecar JSON file."
    public let defaultApprovalLevel = ApprovalLevel.auto

    public let parameters = JSONSchema(
        type: "object",
        properties: [
            "path": SchemaProperty(
                type: "string",
                description: "Path to the sidecar JSON file."
            ),
        ],
        required: ["path"]
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
        guard let path = arguments["path"]?.stringValue, !path.isEmpty else {
            return .fail("path is required")
        }

        let expanded = NSString(string: path).expandingTildeInPath

        guard FileManager.default.fileExists(atPath: expanded) else {
            return .fail("Sidecar file not found: \(expanded)")
        }

        // Prefer the linked FFI inspector when the xcframework is linked; the
        // stub returns .fallbackToCLI so this gate is safe in SwiftPM builds.
        if TesseraFFIBridge.isAvailable {
            switch TesseraFFIBridge.inspectSidecar(path: expanded) {
            case let .success(output):
                return formatSidecarOutput(jsonString: output, expanded: expanded, backend: "ffi")
            case .fallbackToCLI:
                break
            case let .error(code, message):
                return .fail("Inspect failed via FFI (code \(code)): \(message)")
            }
        }

        // CLI fallback: tessera-cli inspect-sidecar <path>
        guard let cli = TesseraCLIBinaryResolver.resolve(
            override: TesseraSettings.tesseraCLIPath,
            settingsKey: TesseraSettingsKey.tesseraCLIPath
        ) else {
            return .fail(TesseraCLIBinaryResolver.diagnosticMessage())
        }

        let result = try await shell.run(
            executable: cli,
            arguments: ["inspect-sidecar", expanded],
            environment: nil,
            workingDirectory: nil
        )
        if result.exitCode != 0 {
            return .fail("Inspect failed (exit \(result.exitCode)):\n\(result.stderr)")
        }
        return formatSidecarOutput(jsonString: result.stdout, expanded: expanded, backend: "cli")
    }

    /// Render the sidecar's JSON as a short multi-line report. Both the FFI
    /// happy path and the CLI fallback produce JSON; this is the single
    /// presentation layer for either source.
    private func formatSidecarOutput(jsonString: String, expanded: String, backend: String) -> ToolResult {
        let json: [String: Any]?
        if let parsed = EngineToolSupport.parseJSONObject(stdout: jsonString) {
            json = parsed
        } else if let data = try? Data(contentsOf: URL(fileURLWithPath: expanded)),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            // Defensive fallback: the CLI returned something that did not
            // parse, so re-read the file directly. Keeps the tool honest
            // when the on-disk file is still the source of truth.
            json = obj
        } else {
            return .fail("Invalid JSON in sidecar file: \(expanded)")
        }
        guard let json else {
            return .fail("Invalid JSON in sidecar file: \(expanded)")
        }
        let schemaVersion = json["schema_version"] as? Int ?? 0
        let profile = json["tessera_profile"] as? String ?? "unknown"
        let effectiveBits = json["effective_bits"] as? Double ?? 0
        let kernelVersion = json["kernel_version"] as? String ?? "unknown"

        var lines = [
            "Sidecar: \(expanded)",
            "Schema version: \(schemaVersion)",
            "Tessera profile: \(profile)",
            "Effective bits: \(effectiveBits)",
            "Kernel version: \(kernelVersion)",
        ]

        if let modalityScales = json["modality_scales"] as? [[String: Any]] {
            lines.append("Modality scales (\(modalityScales.count)):")
            for ms in modalityScales {
                let modality = ms["modality"] as? String ?? "?"
                let alpha = ms["awq_alpha"] as? Double ?? 0
                lines.append("  \(modality): alpha=\(alpha)")
            }
        }

        if let corpus = json["calibration_corpus"] as? String {
            lines.append("Calibration corpus: \(corpus)")
        }

        return .ok(lines.joined(separator: "\n"), data: [
            "schema_version": .number(Double(schemaVersion)),
            "tessera_profile": .string(profile),
            "effective_bits": .number(effectiveBits),
            "backend": .string(backend),
        ])
    }
}
