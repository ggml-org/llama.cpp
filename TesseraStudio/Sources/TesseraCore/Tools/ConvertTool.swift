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

    public init() {}

    public func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        guard let modelPath = arguments["model_path"]?.stringValue, !modelPath.isEmpty else {
            return .fail("model_path is required")
        }
        guard let outputPath = arguments["output_path"]?.stringValue, !outputPath.isEmpty else {
            return .fail("output_path is required")
        }

        let computeUnits = arguments["compute_units"]?.stringValue ?? "cpuAndNeuralEngine"
        let precision = arguments["precision"]?.stringValue ?? "float16"

        let runner = ProcessRunner()
        let result = try await runner.run(
            executable: "tessera-convert",
            arguments: [
                "--model", NSString(string: modelPath).expandingTildeInPath,
                "--output", NSString(string: outputPath).expandingTildeInPath,
                "--compute-units", computeUnits,
                "--precision", precision,
            ]
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
