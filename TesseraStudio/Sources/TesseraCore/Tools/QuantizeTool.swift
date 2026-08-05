import Foundation

/// Quantizes a model using a Tessera policy.
public struct QuantizeTool: TesseraTool {
    public let name = "quantize"
    public let description = "Quantize a GGUF model using a Tessera calibration policy. Produces a Tessera-quantized GGUF with per-tensor policy embedded in metadata."
    public let defaultApprovalLevel = ApprovalLevel.prompt

    public let parameters = JSONSchema(
        type: "object",
        properties: [
            "model_path": SchemaProperty(
                type: "string",
                description: "Path to the source BF16/FP16 GGUF model."
            ),
            "output_path": SchemaProperty(
                type: "string",
                description: "Path for the quantized output GGUF."
            ),
            "policy_path": SchemaProperty(
                type: "string",
                description: "Path to the calibration-policy JSON (from evolve)."
            ),
            "imatrix_path": SchemaProperty(
                type: "string",
                description: "Optional path to the imatrix v2 file."
            ),
            "n_threads": SchemaProperty(
                type: "integer",
                description: "Number of threads for quantization. Default: all cores.",
                defaultValue: "0",
                minimum: 0
            ),
        ],
        required: ["model_path", "output_path", "policy_path"]
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
        guard let modelPath = arguments["model_path"]?.stringValue, !modelPath.isEmpty else {
            return .fail("model_path is required")
        }
        guard let outputPath = arguments["output_path"]?.stringValue, !outputPath.isEmpty else {
            return .fail("output_path is required")
        }
        guard let policyPath = arguments["policy_path"]?.stringValue, !policyPath.isEmpty else {
            return .fail("policy_path is required")
        }

        let expandedModel = NSString(string: modelPath).expandingTildeInPath
        let expandedOutput = NSString(string: outputPath).expandingTildeInPath

        // Prefer the linked xcframework (in-process, no subprocess) when
        // available; otherwise shell out to tessera-cli.
        if TesseraFFIBridge.isAvailable {
            var config: [String: Any] = [
                "policy_path": NSString(string: policyPath).expandingTildeInPath,
            ]
            if let imatrixPath = arguments["imatrix_path"]?.stringValue, !imatrixPath.isEmpty {
                config["imatrix_path"] = NSString(string: imatrixPath).expandingTildeInPath
            }
            if let nThreads = arguments["n_threads"]?.numberValue.map({ Int($0) }), nThreads > 0 {
                config["nthreads"] = nThreads
            }
            switch TesseraFFIBridge.quantize(
                model: expandedModel, output: expandedOutput, config: config
            ) {
            case let .success(output):
                return .ok(output, data: [
                    "output_path": .string(outputPath),
                    "policy_path": .string(policyPath),
                    "backend": .string("ffi"),
                ])
            case .fallbackToCLI:
                break   // fall through to the CLI subprocess below
            case let .error(code, message):
                return .fail("Quantization failed via FFI (code \(code)): \(message)")
            }
        }

        // CLI fallback: tessera-cli quantize <model> <output> --config <json>
        guard let cli = TesseraCLIBinaryResolver.resolve(
            override: TesseraSettings.tesseraCLIPath,
            settingsKey: TesseraSettingsKey.tesseraCLIPath
        ) else {
            return .fail(TesseraCLIBinaryResolver.diagnosticMessage())
        }

        var config: [String: Any] = [
            "policy_path": NSString(string: policyPath).expandingTildeInPath,
        ]
        if let imatrixPath = arguments["imatrix_path"]?.stringValue, !imatrixPath.isEmpty {
            config["imatrix_path"] = NSString(string: imatrixPath).expandingTildeInPath
        }
        if let nThreads = arguments["n_threads"]?.numberValue.map({ Int($0) }), nThreads > 0 {
            config["nthreads"] = nThreads
        }

        let configPath: String
        do {
            configPath = try EngineToolSupport.writeConfigFile(config: config)
        } catch {
            return .fail("Failed to write quantize config: \(error.localizedDescription)")
        }
        defer { try? FileManager.default.removeItem(atPath: configPath) }

        let args = [
            "quantize", expandedModel, expandedOutput,
            "--config", configPath,
        ]

        let result = try await shell.run(
            executable: cli,
            arguments: args,
            environment: nil,
            workingDirectory: nil
        )

        if result.exitCode == 0 {
            return .ok("Quantization complete.\n\(result.stdout)", data: [
                "output_path": .string(outputPath),
                "policy_path": .string(policyPath),
                "backend": .string("cli"),
            ])
        } else {
            return .fail("Quantization failed (exit \(result.exitCode)):\n\(result.stderr)")
        }
    }
}

extension TesseraCLIBinaryResolver {
    /// Short human-readable string for the Settings "not found" state and
    /// for the tool error when the binary cannot be resolved.
    static func diagnosticMessage() -> String {
        switch resolvedPathOrDiagnostic(
            override: TesseraSettings.tesseraCLIPath,
            settingsKey: TesseraSettingsKey.tesseraCLIPath
        ) {
        case .found: return "tessera-cli resolved but missing at call time"
        case .notFound(let searched):
            return "tessera-cli binary not found; checked: \(searched.joined(separator: "\n  - "))"
        }
    }
}
