import Foundation

/// Inspects a Tessera sidecar file and reports its contents.
struct InspectSidecarTool: TesseraTool {
    let name = "inspect_sidecar"
    let description = "Read and display the contents of a Tessera calibration-policy sidecar JSON file."
    let defaultApprovalLevel = ApprovalLevel.auto

    let parameters = JSONSchema(
        type: "object",
        properties: [
            "path": SchemaProperty(
                type: "string",
                description: "Path to the sidecar JSON file."
            ),
        ],
        required: ["path"]
    )

    func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        guard let path = arguments["path"]?.stringValue, !path.isEmpty else {
            return .fail("path is required")
        }

        let expanded = NSString(string: path).expandingTildeInPath
        guard FileManager.default.fileExists(atPath: expanded) else {
            return .fail("Sidecar file not found: \(expanded)")
        }

        let data = try Data(contentsOf: URL(fileURLWithPath: expanded))
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return .fail("Invalid JSON in sidecar file")
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
        ])
    }
}
