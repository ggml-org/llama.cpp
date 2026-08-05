import Foundation

/// Runs AWQ-evolve to find optimal per-tensor quantization policy.
public struct EvolveTool: TesseraTool {
    public let name = "evolve"
    public let description = "Run AWQ-evolve genetic search to find the optimal per-tensor quantization policy given an imatrix and target bit budget."
    public let defaultApprovalLevel = ApprovalLevel.prompt

    public let parameters = JSONSchema(
        type: "object",
        properties: [
            "model_path": SchemaProperty(
                type: "string",
                description: "Path to the source GGUF model."
            ),
            "imatrix_path": SchemaProperty(
                type: "string",
                description: "Path to the imatrix v2 file from calibration."
            ),
            "output_path": SchemaProperty(
                type: "string",
                description: "Path for the output calibration-policy JSON."
            ),
            "target_bits": SchemaProperty(
                type: "number",
                description: "Target effective bits per weight. Default 4.0.",
                defaultValue: "4.0"
            ),
            "generations": SchemaProperty(
                type: "integer",
                description: "Number of evolution generations. Default 50.",
                defaultValue: "50"
            ),
            "population": SchemaProperty(
                type: "integer",
                description: "Population size per generation. Default 32.",
                defaultValue: "32"
            ),
        ],
        required: ["model_path", "imatrix_path", "output_path"]
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
        guard let imatrixPath = arguments["imatrix_path"]?.stringValue, !imatrixPath.isEmpty else {
            return .fail("imatrix_path is required")
        }
        guard let outputPath = arguments["output_path"]?.stringValue, !outputPath.isEmpty else {
            return .fail("output_path is required")
        }

        let targetBits = arguments["target_bits"]?.numberValue ?? 4.0
        let generations = arguments["generations"]?.numberValue.map { Int($0) } ?? 50
        let population = arguments["population"]?.numberValue.map { Int($0) } ?? 32

        let expandedModel = NSString(string: modelPath).expandingTildeInPath

        // The FFI cannot run the GA (it needs a loaded model context), so the
        // real impl and the stub both return fallbackToCLI; the gate is kept
        // so a future in-process evolve path slots in here without touching
        // the tool signature.
        if TesseraFFIBridge.isAvailable {
            let config: [String: Any] = [
                "generations": generations,
                "population": population,
                "target_bits": targetBits,
                "imatrix_path": NSString(string: imatrixPath).expandingTildeInPath,
                "policy_out_path": NSString(string: outputPath).expandingTildeInPath,
            ]
            switch TesseraFFIBridge.evolve(model: expandedModel, config: config) {
            case let .success(output):
                return .ok(output, data: [
                    "output_path": .string(outputPath),
                    "target_bits": .number(targetBits),
                    "generations": .number(Double(generations)),
                    "backend": .string("ffi"),
                ])
            case .fallbackToCLI:
                break
            case let .error(code, message):
                return .fail("Evolution failed via FFI (code \(code)): \(message)")
            }
        }

        // CLI fallback: tessera-cli evolve <model> --config <json>
        guard let cli = TesseraCLIBinaryResolver.resolve(
            override: TesseraSettings.tesseraCLIPath,
            settingsKey: TesseraSettingsKey.tesseraCLIPath
        ) else {
            return .fail(TesseraCLIBinaryResolver.diagnosticMessage())
        }

        let config: [String: Any] = [
            "imatrix_path": NSString(string: imatrixPath).expandingTildeInPath,
            "policy_out_path": NSString(string: outputPath).expandingTildeInPath,
            "target_bits": targetBits,
            "generations": generations,
            "population": population,
        ]
        let configPath: String
        do {
            configPath = try EngineToolSupport.writeConfigFile(config: config)
        } catch {
            return .fail("Failed to write evolve config: \(error.localizedDescription)")
        }
        defer { try? FileManager.default.removeItem(atPath: configPath) }

        let args = [
            "evolve", expandedModel,
            "--config", configPath,
        ]
        let result = try await shell.run(
            executable: cli,
            arguments: args,
            environment: nil,
            workingDirectory: nil
        )

        if result.exitCode == 0 {
            return .ok("Evolution complete.\n\(result.stdout)", data: [
                "output_path": .string(outputPath),
                "target_bits": .number(targetBits),
                "generations": .number(Double(generations)),
                "backend": .string("cli"),
            ])
        } else {
            return .fail("Evolution failed (exit \(result.exitCode)):\n\(result.stderr)")
        }
    }
}
