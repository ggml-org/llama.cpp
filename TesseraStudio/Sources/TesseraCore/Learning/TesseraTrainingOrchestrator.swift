import Foundation

/// Orchestrates the drafter training pipeline: accumulate traces -> shell
/// out to the native tessera-train-lk driver -> record the outcome. The
/// driver reads llama.tessera.spec.v1 traces, builds the dense-label LK
/// dataset in-process, trains the drafter with GGML_OPT_LOSS_TYPE_LK, and
/// saves the trained GGUF (docs/tessera-lk-training-design.md). One binary,
/// one step - the old dataset-prep/finetune/export-lora split is gone.
///
/// Plug-in points (clearly marked, not faked):
/// - the driver binary must be built and reachable at Config.trainBinary;
///   a missing binary is reported with the expected path and the build
///   command, never silently skipped
/// - the standard finetune-style knobs (epochs, learning rate) stay at the
///   driver's defaults for now
/// - the capability-eval guard is owned by TesseraAdaptationScheduler; v1
///   run() stops at the trained drafter and leaves guardPassed/guardFailed
///   reserved
public final class TesseraTrainingOrchestrator: @unchecked Sendable {
    public struct Config: Sendable {
        public var minTracesForTraining: Int
        public var trainBinary: String        // tessera-train-lk
        public var baseModelPath: String?     // drafter GGUF to train; nil = training disabled
        public var drafterOutPath: String?    // nil = default next to base model
        public var maxExamples: Int           // dataset cap (--max-examples)
        public var dryRun: Bool               // pass --dry-run: build dataset only

        public init(
            minTracesForTraining: Int = 1000,
            trainBinary: String = "/usr/local/bin/tessera-train-lk",
            baseModelPath: String? = nil,
            drafterOutPath: String? = nil,
            maxExamples: Int = 512,
            dryRun: Bool = true
        ) {
            self.minTracesForTraining = minTracesForTraining
            self.trainBinary = trainBinary
            self.baseModelPath = baseModelPath
            self.drafterOutPath = drafterOutPath
            self.maxExamples = maxExamples
            self.dryRun = dryRun
        }
    }

    /// Outcome of one training cycle. The v1 terminal set returned by run()
    /// is {skippedInsufficientTraces, skippedNoModel, dryRun, trainingFailed,
    /// trainingCompleted}; datasetPrepared and exportCompleted are legacy
    /// outcomes persisted by the old three-step pipeline, and the guard cases
    /// are reserved for the wired capability-eval gate (plug-in point).
    public enum TrainingOutcome: String, Codable, Sendable {
        case skippedInsufficientTraces
        case skippedNoModel
        case datasetPrepared
        case trainingCompleted
        case trainingFailed
        case exportCompleted
        case guardPassed
        case guardFailed
        case dryRun
    }

    public struct TrainingRecord: Codable, Sendable {
        public let timestamp: Date
        public let outcome: TrainingOutcome
        public let traceCount: Int
        public let drafterPath: String?
        public let stdout: String?
        public let stderr: String?
        public let note: String

        public init(
            timestamp: Date = Date(),
            outcome: TrainingOutcome,
            traceCount: Int,
            drafterPath: String? = nil,
            stdout: String? = nil,
            stderr: String? = nil,
            note: String
        ) {
            self.timestamp = timestamp
            self.outcome = outcome
            self.traceCount = traceCount
            self.drafterPath = drafterPath
            self.stdout = stdout
            self.stderr = stderr
            self.note = note
        }
    }

    /// Per-stream cap on captured driver output, so the persisted record
    /// stays small. The tail is kept - final losses and errors land there.
    static let outputCap = 4000

    public let traceStore: TesseraTraceStore
    private let config: Config
    private let store: TesseraLearningStore
    private let lock = NSLock()
    private static let recordFile = "training-record.json"

    public init(
        config: Config = Config(),
        traceStore: TesseraTraceStore = TesseraTraceStore(),
        storeDirectory: URL? = nil
    ) {
        self.config = config
        self.traceStore = traceStore
        self.store = storeDirectory.map { TesseraLearningStore(directory: $0) } ?? TesseraLearningStore()
    }

    /// The most recent persisted record, if any.
    public func lastTraining() -> TrainingRecord? {
        lock.lock(); defer { lock.unlock() }
        return store.load(TrainingRecord?.self, from: Self.recordFile, default: nil)
    }

    /// Run one training cycle. `overrideDryRun` wins when non-nil; otherwise
    /// the configured dryRun applies. `maxExamples` overrides the configured
    /// dataset cap when non-nil. Every branch persists and returns an honest
    /// record - nothing is fabricated.
    public func run(overrideDryRun: Bool? = nil, maxExamples: Int? = nil) async -> TrainingRecord {
        let dryRun = overrideDryRun ?? config.dryRun
        let traceCount = traceStore.totalRecords()

        // Gate 1: enough traces to form a dataset.
        guard traceCount >= config.minTracesForTraining else {
            return finish(TrainingRecord(
                outcome: .skippedInsufficientTraces,
                traceCount: traceCount,
                note: "have \(traceCount) trace record(s), need \(config.minTracesForTraining); run more imatrix calibration first"
            ))
        }

        // Gate 2: a drafter model to train.
        guard let base = config.baseModelPath, !base.isEmpty else {
            return finish(TrainingRecord(
                outcome: .skippedNoModel,
                traceCount: traceCount,
                note: "learning.baseModelPath is not set; training disabled"
            ))
        }

        // Gate 3: the native driver binary, checked up front so a missing
        // build is an actionable message, not a bare process-launch error.
        guard FileManager.default.isExecutableFile(atPath: config.trainBinary) else {
            return finish(TrainingRecord(
                outcome: .trainingFailed,
                traceCount: traceCount,
                note: Self.missingBinaryNote(path: config.trainBinary)
            ))
        }

        let cap = maxExamples ?? config.maxExamples
        guard cap > 0 else {
            return finish(TrainingRecord(
                outcome: .trainingFailed,
                traceCount: traceCount,
                note: "max-examples must be > 0 (got \(cap))"
            ))
        }

        // Stage all trace files into one JSONL the driver reads.
        let staged: String
        do {
            staged = try stageCombinedTraces()
        } catch {
            return finish(TrainingRecord(
                outcome: .trainingFailed,
                traceCount: traceCount,
                note: "could not stage traces for tessera-train-lk: \(error.localizedDescription)"
            ))
        }
        defer { try? FileManager.default.removeItem(atPath: staged) }

        let out = resolvedDrafterPath(base: base)
        let arguments = Self.trainArguments(
            traces: staged, model: base, out: out, maxExamples: cap, dryRun: dryRun
        )

        let result = await runStep(binary: config.trainBinary, arguments: arguments)
        switch result {
        case .failure(let error):
            return finish(TrainingRecord(
                outcome: .trainingFailed,
                traceCount: traceCount,
                note: "tessera-train-lk could not start (\(error.localizedDescription)); expected at \(config.trainBinary)"
            ))
        case .success(let res) where res.exitCode != 0:
            return finish(TrainingRecord(
                outcome: .trainingFailed,
                traceCount: traceCount,
                stdout: Self.capped(res.stdout),
                stderr: Self.capped(res.stderr),
                note: "tessera-train-lk exited \(res.exitCode)"
            ))
        case .success(let res) where dryRun:
            return finish(TrainingRecord(
                outcome: .dryRun,
                traceCount: traceCount,
                stdout: Self.capped(res.stdout),
                stderr: Self.capped(res.stderr),
                note: "dry run: tessera-train-lk built the dataset from \(traceCount) trace(s) against \(base); nothing trained or saved"
            ))
        case .success(let res):
            return finish(TrainingRecord(
                outcome: .trainingCompleted,
                traceCount: traceCount,
                drafterPath: out,
                stdout: Self.capped(res.stdout),
                stderr: Self.capped(res.stderr),
                note: "drafter trained with tessera-train-lk (max-examples \(cap)) and saved to \(out)"
            ))
        }
    }

    // MARK: - Driver contract

    /// The exact tessera-train-lk invocation for one training cycle. Pure so
    /// the flag wiring is testable without a binary; flags per the driver's
    /// contract (tools/quantize/tessera/tessera-train-lk.cpp).
    static func trainArguments(
        traces: String,
        model: String,
        out: String,
        maxExamples: Int,
        dryRun: Bool
    ) -> [String] {
        var args = [
            "-m", model,
            "--traces", traces,
            "-o", out,
            "--max-examples", String(maxExamples),
        ]
        if dryRun { args.append("--dry-run") }
        return args
    }

    /// Actionable message for a missing driver binary: names the expected
    /// path and the build command that produces it.
    static func missingBinaryNote(path: String) -> String {
        "tessera-train-lk not found at \(path); build it in the llama.cpp checkout "
            + "(cmake -B build && cmake --build build --target tessera-train-lk) "
            + "and install it at that path"
    }

    // MARK: - Shell-out

    private func runStep(binary: String, arguments: [String]) async -> Result<ProcessResult, Error> {
        do {
            return .success(try await ProcessRunner().run(executable: binary, arguments: arguments))
        } catch {
            return .failure(error)
        }
    }

    // MARK: - Paths

    private func stageCombinedTraces() throws -> String {
        let path = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("tessera-train-\(UUID().uuidString).traces.jsonl")
        var out = Data()
        for file in traceStore.traceFiles() {
            guard let data = try? Data(contentsOf: file) else { continue }
            out.append(data)
            if data.last != 0x0A { out.append(0x0A) }   // keep records on separate lines
        }
        try out.write(to: URL(fileURLWithPath: path), options: .atomic)
        return path
    }

    private func resolvedDrafterPath(base: String) -> String {
        if let configured = config.drafterOutPath, !configured.isEmpty { return configured }
        let url = URL(fileURLWithPath: base)
        let stem = url.deletingPathExtension().lastPathComponent
        return url.deletingLastPathComponent()
            .appendingPathComponent("\(stem)-tessera-trained.gguf").path
    }

    /// Keep the tail of a driver stream within outputCap.
    private static func capped(_ text: String) -> String? {
        guard !text.isEmpty else { return nil }
        guard text.count > outputCap else { return text }
        return String(text.suffix(outputCap))
    }

    // MARK: - Persistence

    private func finish(_ record: TrainingRecord) -> TrainingRecord {
        lock.lock(); defer { lock.unlock() }
        try? store.save(record, to: Self.recordFile)
        return record
    }
}
