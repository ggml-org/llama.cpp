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

    public init() {}

    public func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        guard let path = arguments["path"]?.stringValue, !path.isEmpty else {
            return .fail("path is required")
        }

        let expanded = NSString(string: path).expandingTildeInPath

        if TesseraFFIBridge.isAvailable {
            do {
                let info = try TesseraFFIBridge.inspectSidecar(path: expanded)
                return .ok(Self.format(info), data: [
                    "schema_version": .number(Double(info.schemaVersion)),
                    "tessera_profile": .string(info.tesseraProfile),
                    "effective_bits": .number(info.effectiveBits),
                    "backend": .string("ffi"),
                ])
            } catch {
                // fall through to CLI / direct read
            }
        }

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
            "backend": .string("cli"),
        ])
    }

    private static func format(_ info: SidecarInfo) -> String {
        var lines = [
            "Sidecar: \(info.modelPath)",
            "Schema version: \(info.schemaVersion)",
            "Tessera profile: \(info.tesseraProfile)",
            "Effective bits: \(info.effectiveBits)",
            "Kernel version: \(info.kernelVersion)",
        ]
        if !info.modalityScales.isEmpty {
            lines.append("Modality scales (\(info.modalityScales.count)):")
            for ms in info.modalityScales {
                lines.append("  \(ms.modality): alpha=\(ms.awqAlpha)")
            }
        }
        if !info.calibrationCorpus.isEmpty {
            lines.append("Calibration corpus: \(info.calibrationCorpus)")
        }
        return lines.joined(separator: "\n")
    }
}
