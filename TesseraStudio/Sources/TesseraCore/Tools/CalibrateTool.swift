import Foundation

/// Runs calibration (imatrix generation) on a model.
public struct CalibrateTool: TesseraTool {
    public let name = "calibrate"
    public let description = "Run imatrix calibration on a model using a calibration corpus. Produces an imatrix v2 file with per-tensor activation statistics."
    public let defaultApprovalLevel = ApprovalLevel.prompt

    public let parameters = JSONSchema(
        type: "object",
        properties: [
            "model_path": SchemaProperty(
                type: "string",
                description: "Path to the BF16 or FP16 GGUF model to calibrate."
            ),
            "corpus_path": SchemaProperty(
                type: "string",
                description: "Path to the calibration corpus (text file or directory)."
            ),
            "output_path": SchemaProperty(
                type: "string",
                description: "Path for the output imatrix file."
            ),
            "n_tokens": SchemaProperty(
                type: "integer",
                description: "Number of calibration tokens. Default 5000.",
                defaultValue: "5000",
                minimum: 1
            ),
            "modality": SchemaProperty(
                type: "string",
                description: "Modality to calibrate.",
                enumValues: ["text", "image", "audio", "all"],
                defaultValue: "text"
            ),
        ],
        required: ["model_path", "corpus_path", "output_path"]
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
        guard let corpusPath = arguments["corpus_path"]?.stringValue, !corpusPath.isEmpty else {
            return .fail("corpus_path is required")
        }
        guard let outputPath = arguments["output_path"]?.stringValue, !outputPath.isEmpty else {
            return .fail("output_path is required")
        }

        let nTokens = arguments["n_tokens"]?.numberValue.map { Int($0) } ?? 5000
        let modality = arguments["modality"]?.stringValue ?? "text"
        let expandedCorpus = NSString(string: corpusPath).expandingTildeInPath
        let expandedOutput = NSString(string: outputPath).expandingTildeInPath

        if TesseraFFIBridge.isAvailable {
            switch TesseraFFIBridge.calibrate(
                model: NSString(string: modelPath).expandingTildeInPath,
                corpus: expandedCorpus,
                config: [
                    "n_tokens": nTokens,
                    "modality": modality,
                    "output_path": expandedOutput,
                ]
            ) {
            case let .success(output):
                return .ok(output, data: [
                    "output_path": .string(outputPath),
                    "n_tokens": .number(Double(nTokens)),
                    "modality": .string(modality),
                    "backend": .string("ffi"),
                ])
            case .fallbackToCLI:
                break
            case let .error(code, message):
                return .fail("Calibration failed via FFI (code \(code)): \(message)")
            }
        }

        // CLI fallback: tessera-cli calibrate <corpus> --config <json>
        guard let cli = TesseraCLIBinaryResolver.resolve(
            override: TesseraSettings.tesseraCLIPath,
            settingsKey: TesseraSettingsKey.tesseraCLIPath
        ) else {
            return .fail(TesseraCLIBinaryResolver.diagnosticMessage())
        }

        let config: [String: Any] = [
            "model_path": NSString(string: modelPath).expandingTildeInPath,
            "output_path": expandedOutput,
            "n_tokens": nTokens,
            "modality": modality,
        ]
        let configPath: String
        do {
            configPath = try EngineToolSupport.writeConfigFile(config: config)
        } catch {
            return .fail("Failed to write calibrate config: \(error.localizedDescription)")
        }
        defer { try? FileManager.default.removeItem(atPath: configPath) }

        let args = [
            "calibrate", expandedCorpus,
            "--config", configPath,
        ]
        let result = try await shell.run(
            executable: cli,
            arguments: args,
            environment: nil,
            workingDirectory: nil
        )

        if result.exitCode == 0 {
            return .ok("Calibration complete.\n\(result.stdout)", data: [
                "output_path": .string(outputPath),
                "n_tokens": .number(Double(nTokens)),
                "modality": .string(modality),
                "backend": .string("cli"),
            ])
        } else {
            return .fail("Calibration failed (exit \(result.exitCode)):\n\(result.stderr)")
        }
    }
}
