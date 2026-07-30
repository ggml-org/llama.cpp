import Foundation

/// Evaluates a quantized model: perplexity, latency, and power metrics.
struct EvaluateTool: TesseraTool {
    let name = "evaluate"
    let description = "Evaluate a quantized model by measuring perplexity on a held-out set, token latency, and ANE power draw."
    let defaultApprovalLevel = ApprovalLevel.notify

    let parameters = JSONSchema(
        type: "object",
        properties: [
            "model_path": SchemaProperty(
                type: "string",
                description: "Path to the quantized GGUF or .mlmodelc to evaluate."
            ),
            "eval_corpus": SchemaProperty(
                type: "string",
                description: "Path to the evaluation corpus for perplexity."
            ),
            "n_tokens": SchemaProperty(
                type: "integer",
                description: "Number of tokens to evaluate. Default 512.",
                defaultValue: "512"
            ),
            "runtime": SchemaProperty(
                type: "string",
                description: "Runtime backend for evaluation.",
                enumValues: TesseraRuntime.allCases.map(\.rawValue),
                defaultValue: TesseraRuntime.onDevice.rawValue
            ),
            "measure_power": SchemaProperty(
                type: "boolean",
                description: "Whether to measure IOReport power metrics. Default true.",
                defaultValue: "true"
            ),
        ],
        required: ["model_path"]
    )

    func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        guard let modelPath = arguments["model_path"]?.stringValue, !modelPath.isEmpty else {
            return .fail("model_path is required")
        }

        let evalCorpus = arguments["eval_corpus"]?.stringValue ?? ""
        let nTokens = arguments["n_tokens"]?.numberValue.map { Int($0) } ?? 512
        let runtime = arguments["runtime"]?.stringValue ?? TesseraRuntime.onDevice.rawValue
        let measurePower = arguments["measure_power"]?.stringValue != "false"

        var args = [
            "--model", NSString(string: modelPath).expandingTildeInPath,
            "--n-tokens", String(nTokens),
            "--runtime", runtime,
        ]
        if !evalCorpus.isEmpty {
            args += ["--corpus", NSString(string: evalCorpus).expandingTildeInPath]
        }
        if measurePower {
            args += ["--measure-power"]
        }

        let runner = ProcessRunner()
        let result = try await runner.run(
            executable: "tessera-evaluate",
            arguments: args
        )

        if result.exitCode == 0 {
            return .ok("Evaluation complete.\n\(result.stdout)", data: [
                "model_path": .string(modelPath),
                "runtime": .string(runtime),
                "n_tokens": .number(Double(nTokens)),
            ])
        } else {
            return .fail("Evaluation failed (exit \(result.exitCode)):\n\(result.stderr)")
        }
    }
}
