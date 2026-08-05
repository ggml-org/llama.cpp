import Foundation

/// Runs one drafter training cycle with the native tessera-train-lk driver:
/// build the dense-label LK dataset from accumulated traces and train the
/// drafter GGUF to maximize the verifier acceptance rate. Heavy and mutating,
/// so it defaults to the driver's --dry-run (dataset built and validated,
/// nothing trained or saved) and gates behind explicit approval.
public struct RunTrainingTool: TesseraTool {
    public let name = "run_training"
    public let description = "Run one drafter training cycle with tessera-train-lk: build the LK dataset from accumulated traces and train the drafter GGUF. Requires learning.baseModelPath to be set."
    public let defaultApprovalLevel = ApprovalLevel.prompt

    public let parameters = JSONSchema(
        type: "object",
        properties: [
            "dry_run": SchemaProperty(
                type: "boolean",
                description: "If true (default), pass --dry-run to the driver: the dataset is built and validated, nothing is trained or saved.",
                defaultValue: "true"
            ),
            "max_examples": SchemaProperty(
                type: "integer",
                description: "Dataset cap (--max-examples); bounds the dense-label memory. Default 512.",
                defaultValue: "512"
            ),
        ],
        required: []
    )

    public init() {}

    public func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        guard let orchestrator = TesseraLearningCenter.shared.training else {
            return .fail("Training orchestrator is not installed.")
        }
        // Absent param falls back to the configured default, so a manual call
        // without arguments honors learning.trainingDryRun.
        let dryRun = Self.parseBool(arguments["dry_run"]) ?? TesseraSettings.learningTrainingDryRun
        let maxExamples = arguments["max_examples"]?.numberValue.map { Int($0) }
        if let maxExamples, maxExamples <= 0 {
            return .fail("max_examples must be > 0")
        }
        let record = await orchestrator.run(overrideDryRun: dryRun, maxExamples: maxExamples)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let json = (try? encoder.encode(record)).flatMap { String(data: $0, encoding: .utf8) } ?? "{}"

        // A failed cycle is an honest failure, not a no-op: surface the note
        // (which names the missing binary / driver stderr) as the error.
        if record.outcome == .trainingFailed {
            var message = record.note
            if let stderr = record.stderr, !stderr.isEmpty {
                message += "\nstderr:\n\(stderr)"
            }
            return .fail(message)
        }

        var data: [String: JSONValue] = [
            "outcome": .string(record.outcome.rawValue),
            "trace_count": .number(Double(record.traceCount)),
            "dry_run": .bool(dryRun),
        ]
        if let drafterPath = record.drafterPath { data["drafter_path"] = .string(drafterPath) }
        if let stdout = record.stdout { data["stdout"] = .string(stdout) }
        if let stderr = record.stderr { data["stderr"] = .string(stderr) }

        return .ok(json, data: data)
    }

    private static func parseBool(_ value: JSONValue?) -> Bool? {
        guard let value else { return nil }
        if case .bool(let b) = value { return b }
        if let s = value.stringValue { return s == "true" || s == "1" }
        if let n = value.numberValue { return n != 0 }
        return nil
    }
}
