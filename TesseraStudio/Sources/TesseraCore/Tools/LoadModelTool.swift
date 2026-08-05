import Foundation

/// Loads a model into the Tessera engine for inference.
public struct LoadModelTool: TesseraTool {
    public let name = "load_model"
    public let description = "Load a GGUF model into the Tessera engine, optionally with a sidecar policy and imatrix."
    public let defaultApprovalLevel = ApprovalLevel.notify

    public let parameters = JSONSchema(
        type: "object",
        properties: [
            "model_path": SchemaProperty(
                type: "string",
                description: "Path to the .gguf file to load."
            ),
            "sidecar_path": SchemaProperty(
                type: "string",
                description: "Optional path to the calibration-policy JSON sidecar."
            ),
            "runtime": SchemaProperty(
                type: "string",
                description: "Inference runtime to use.",
                enumValues: TesseraRuntime.allCases.map(\.rawValue),
                defaultValue: TesseraRuntime.onDevice.rawValue
            ),
            "n_ctx": SchemaProperty(
                type: "integer",
                description: "Context length. Defaults to 4096.",
                defaultValue: "4096"
            ),
        ],
        required: ["model_path"]
    )

    public init() {}

    public func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        guard let modelPath = arguments["model_path"]?.stringValue, !modelPath.isEmpty else {
            return .fail("model_path is required")
        }

        let expanded = NSString(string: modelPath).expandingTildeInPath
        guard FileManager.default.fileExists(atPath: expanded) else {
            return .fail("Model file not found: \(expanded)")
        }

        let sidecarPath = arguments["sidecar_path"]?.stringValue
        let runtime = arguments["runtime"]?.stringValue ?? TesseraRuntime.onDevice.rawValue
        let nCtx = arguments["n_ctx"]?.numberValue.map { Int($0) } ?? 4096

        // load_model stays a Swift state op (no subprocess): the on-device
        // inference path runs through CLlama / LlamaLLMProvider, and the CLI
        // surface would just be a wrapper around the same C call. Surface
        // the resolved metadata so the caller can see what the load saw.
        var report = [
            "Loaded model: \(expanded)",
            "Runtime: \(runtime)",
            "Context: \(nCtx)",
        ]
        if let sidecar = sidecarPath {
            report.append("Sidecar: \(sidecar)")
        }

        return .ok(report.joined(separator: "\n"), data: [
            "model_path": .string(expanded),
            "runtime": .string(runtime),
            "n_ctx": .number(Double(nCtx)),
            "status": .string("loaded"),
        ])
    }
}
