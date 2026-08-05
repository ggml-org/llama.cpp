import Foundation

/// Converts a Tessera-quantized GGUF to CoreML .mlmodelc format.
public struct ConvertTool: TesseraTool {
    public let name = "convert"
    public let description = "Convert a Tessera-quantized GGUF to a CoreML .mlmodelc package for on-device ANE inference."
    public let defaultApprovalLevel = ApprovalLevel.prompt

    public let parameters = JSONSchema(
        type: "object",
        properties: [
            "model_path": SchemaProperty(
                type: "string",
                description: "Path to the Tessera-quantized GGUF."
            ),
            "output_path": SchemaProperty(
                type: "string",
                description: "Path for the output .mlmodelc directory."
            ),
            "compute_units": SchemaProperty(
                type: "string",
                description: "CoreML compute units target.",
                enumValues: ["all", "cpuAndGPU", "cpuOnly", "cpuAndNeuralEngine"],
                defaultValue: "cpuAndNeuralEngine"
            ),
            "precision": SchemaProperty(
                type: "string",
                description: "Model precision.",
                enumValues: ["float16", "float32"],
                defaultValue: "float16"
            ),
        ],
        required: ["model_path", "output_path"]
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

        let computeUnits = arguments["compute_units"]?.stringValue ?? "cpuAndNeuralEngine"
        let precision = arguments["precision"]?.stringValue ?? "float16"
        let expandedModel = NSString(string: modelPath).expandingTildeInPath
        let expandedOutput = NSString(string: outputPath).expandingTildeInPath

        if TesseraFFIBridge.isAvailable {
            switch TesseraFFIBridge.convert(
                model: expandedModel, output: expandedOutput, format: "coreml"
            ) {
            case let .success(output):
                return .ok(output, data: [
                    "output_path": .string(outputPath),
                    "compute_units": .string(computeUnits),
                    "precision": .string(precision),
                    "backend": .string("ffi"),
                ])
            case .fallbackToCLI:
                break
            case let .error(code, message):
                return .fail("Conversion failed via FFI (code \(code)): \(message)")
            }
        }

        // CLI fallback: tessera-cli convert <model> <output> --format coreml
        // compute_units + precision are embedded in the JSON config because
        // the spec only names --format on the command line.
        guard let cli = TesseraCLIBinaryResolver.resolve(
            override: TesseraSettings.tesseraCLIPath,
            settingsKey: TesseraSettingsKey.tesseraCLIPath
        ) else {
            return .fail(TesseraCLIBinaryResolver.diagnosticMessage())
        }

        let config: [String: Any] = [
            "compute_units": computeUnits,
            "precision": precision,
        ]
        let configPath: String
        do {
            configPath = try EngineToolSupport.writeConfigFile(config: config)
        } catch {
            return .fail("Failed to write convert config: \(error.localizedDescription)")
        }
        defer { try? FileManager.default.removeItem(atPath: configPath) }

        let args = [
            "convert", expandedModel, expandedOutput,
            "--format", "coreml",
            "--config", configPath,
        ]
        let result = try await shell.run(
            executable: cli,
            arguments: args,
            environment: nil,
            workingDirectory: nil
        )

        if result.exitCode == 0 {
            return .ok("Conversion complete.\n\(result.stdout)", data: [
                "output_path": .string(outputPath),
                "compute_units": .string(computeUnits),
                "precision": .string(precision),
                "backend": .string("cli"),
            ])
        } else {
            return .fail("Conversion failed (exit \(result.exitCode)):\n\(result.stderr)")
        }
    }
}
