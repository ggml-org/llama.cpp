import Foundation

/// Harvests drafter-training traces: runs speculative calibration
/// (llama-imatrix with --model-draft) over a corpus and appends the emitted
/// llama.tessera.spec.v1 telemetry to the trace store. This is the fuel for
/// the learning loop - tessera-train-lk reads these traces and trains a
/// drafter that maximizes the verifier acceptance rate
/// (docs/tessera-lk-training-design.md).
///
/// --telemetry-topk must be > 0 for the LK driver: it densifies the verifier
/// top-N distributions into labels, so a cheap per-step-only payload is not
/// enough. The imatrix file itself is a byproduct here and is discarded.
public struct CollectTrainingTracesTool: TesseraTool {
    public let name = "collect_training_traces"
    public let description = "Run speculative calibration (trunk + drafter) over a corpus and harvest acceptance telemetry into the drafter-training trace store. Requires llama-imatrix, resolved next to tessera-train-lk."
    public let defaultApprovalLevel = ApprovalLevel.prompt

    public let parameters = JSONSchema(
        type: "object",
        properties: [
            "model_path": SchemaProperty(
                type: "string",
                description: "Path to the trunk (verifier) GGUF model."
            ),
            "draft_model_path": SchemaProperty(
                type: "string",
                description: "Path to the drafter GGUF whose acceptance is measured."
            ),
            "corpus_path": SchemaProperty(
                type: "string",
                description: "Path to the calibration corpus (text file)."
            ),
            "n_tokens": SchemaProperty(
                type: "integer",
                description: "Number of tokens to process. Default 5000.",
                defaultValue: "5000",
                minimum: 1
            ),
            "telemetry_topk": SchemaProperty(
                type: "integer",
                description: "Top-N verifier/drafter distributions recorded per position (required for LK training). Default 64.",
                defaultValue: "64",
                minimum: 0
            ),
        ],
        required: ["model_path", "draft_model_path", "corpus_path"]
    )

    public init() {}

    /// Where harvested traces are appended. Defaults to the app-wide trace
    /// store; tests inject an isolated directory.
    var traceStoreDirectory: URL?

    public func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        guard let modelPath = arguments["model_path"]?.stringValue, !modelPath.isEmpty else {
            return .fail("model_path is required")
        }
        guard let draftPath = arguments["draft_model_path"]?.stringValue, !draftPath.isEmpty else {
            return .fail("draft_model_path is required")
        }
        guard let corpusPath = arguments["corpus_path"]?.stringValue, !corpusPath.isEmpty else {
            return .fail("corpus_path is required")
        }
        let nTokens = arguments["n_tokens"]?.numberValue.map { Int($0) } ?? 5000
        let topk = arguments["telemetry_topk"]?.numberValue.map { Int($0) } ?? 64

        let imatrix = TesseraTrainBinaryResolver.resolveImatrix(
            trainOverride: TesseraSettings.learningTrainBinary
        )
        guard FileManager.default.isExecutableFile(atPath: imatrix) else {
            return .fail(Self.missingImatrixNote(path: imatrix))
        }

        let telemetryPath = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("tessera-traces-\(UUID().uuidString).jsonl")
        let imatrixOut = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("tessera-imatrix-\(UUID().uuidString).gguf")

        let result = try await ProcessRunner().run(
            executable: imatrix,
            arguments: [
                "-m", NSString(string: modelPath).expandingTildeInPath,
                "--model-draft", NSString(string: draftPath).expandingTildeInPath,
                "-f", NSString(string: corpusPath).expandingTildeInPath,
                "-o", imatrixOut,
                "-n", String(nTokens),
                "--telemetry-out", telemetryPath,
                "--telemetry-topk", String(topk),
            ]
        )
        // The imatrix file is a byproduct of trace collection; drop it either way.
        try? FileManager.default.removeItem(atPath: imatrixOut)

        guard result.exitCode == 0 else {
            try? FileManager.default.removeItem(atPath: telemetryPath)
            return .fail("llama-imatrix exited \(result.exitCode):\n\(result.stderr)")
        }

        // Honest append: count the records the driver actually emitted. An
        // empty telemetry file is a real no-op, not a success.
        let store = traceStoreDirectory.map { TesseraTraceStore(directory: $0) } ?? TesseraTraceStore()
        var added = 0
        if FileManager.default.fileExists(atPath: telemetryPath) {
            let url = URL(fileURLWithPath: telemetryPath)
            let stored = try store.appendRun(jsonlPath: url)
            added = TesseraTraceStore.recordCount(inFile: stored)
            try? FileManager.default.removeItem(atPath: telemetryPath)
        }
        let total = store.totalRecords()

        return .ok("Collected \(added) trace record(s); \(total) total in the store.", data: [
            "traces_added": .number(Double(added)),
            "total_traces": .number(Double(total)),
            "draft_model_path": .string(draftPath),
            "telemetry_topk": .number(Double(topk)),
            "imatrix_binary": .string(imatrix),
        ])
    }

    /// Actionable message for a missing llama-imatrix: name the resolved path
    /// and the build target that produces it.
    static func missingImatrixNote(path: String) -> String {
        "llama-imatrix not found at \(path); build it in the llama.cpp checkout "
            + "(cmake --build build --target llama-imatrix) and make it reachable "
            + "next to tessera-train-lk or in /usr/local/bin"
    }
}
