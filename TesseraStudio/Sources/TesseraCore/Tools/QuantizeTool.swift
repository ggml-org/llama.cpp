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
                defaultValue: "0"
            ),
        ],
        required: ["model_path", "output_path", "policy_path"]
    )

    public init() {}

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

        var args = [
            NSString(string: modelPath).expandingTildeInPath,
            NSString(string: outputPath).expandingTildeInPath,
            "--tessera-policy", NSString(string: policyPath).expandingTildeInPath,
        ]

        if let imatrixPath = arguments["imatrix_path"]?.stringValue, !imatrixPath.isEmpty {
            args += ["--imatrix", NSString(string: imatrixPath).expandingTildeInPath]
        }

        if let nThreads = arguments["n_threads"]?.numberValue.map({ Int($0) }), nThreads > 0 {
            args += ["--threads", String(nThreads)]
        }

        let runner = ProcessRunner()
        let result = try await runner.run(
            executable: "tessera-quantize",
            arguments: args
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
