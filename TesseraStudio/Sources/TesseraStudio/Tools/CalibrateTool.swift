import Foundation

/// Runs calibration (imatrix generation) on a model.
struct CalibrateTool: TesseraTool {
    let name = "calibrate"
    let description = "Run imatrix calibration on a model using a calibration corpus. Produces an imatrix v2 file with per-tensor activation statistics."
    let defaultApprovalLevel = ApprovalLevel.prompt

    let parameters = JSONSchema(
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
                defaultValue: "5000"
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

    func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
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

        let runner = ProcessRunner()
        let result = try await runner.run(
            executable: "tessera-imatrix",
            arguments: [
                "--model", NSString(string: modelPath).expandingTildeInPath,
                "--corpus", NSString(string: corpusPath).expandingTildeInPath,
                "--output", NSString(string: outputPath).expandingTildeInPath,
                "--n-tokens", String(nTokens),
                "--modality", modality,
            ]
        )

        if result.exitCode == 0 {
            return .ok("Calibration complete.\n\(result.stdout)", data: [
                "output_path": .string(outputPath),
                "n_tokens": .number(Double(nTokens)),
                "modality": .string(modality),
            ])
        } else {
            return .fail("Calibration failed (exit \(result.exitCode)):\n\(result.stderr)")
        }
    }
}
