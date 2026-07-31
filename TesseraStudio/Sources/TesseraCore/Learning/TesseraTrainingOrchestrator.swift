import Foundation

/// Orchestrates the drafter training pipeline: accumulate traces -> prepare
/// dataset -> run finetune -> export-lora -> gate on capability eval.
/// All heavy steps shell out to the llama.cpp CLI tools; this service
/// manages the lifecycle and records the outcome.
///
/// Plug-in points (clearly marked, not faked):
/// - finetune and export-lora require a real model + GPU; the orchestrator
///   shells out and reports the result honestly
/// - LK loss training is not wired in v1; standard cross-entropy is used
/// - the capability-eval guard is owned by TesseraAdaptationScheduler; v1
///   run() stops at export and leaves guardPassed/guardFailed reserved
public final class TesseraTrainingOrchestrator: @unchecked Sendable {
    public struct Config: Sendable {
        public var minTracesForTraining: Int
        public var finetuneBinary: String
        public var exportLoraBinary: String
        public var datasetPrepBinary: String    // llama-quantize, tessera-dataset op
        public var baseModelPath: String?       // nil = not configured, training disabled
        public var loraOutPath: String?         // nil = default next to base model
        public var adamIterations: Int
        public var dryRun: Bool                 // record intent, don't run

        public init(
            minTracesForTraining: Int = 1000,
            finetuneBinary: String = "/usr/local/bin/llama-finetune",
            exportLoraBinary: String = "/usr/local/bin/llama-export-lora",
            datasetPrepBinary: String = "/usr/local/bin/llama-quantize",
            baseModelPath: String? = nil,
            loraOutPath: String? = nil,
            adamIterations: Int = 100,
            dryRun: Bool = true
        ) {
            self.minTracesForTraining = minTracesForTraining
            self.finetuneBinary = finetuneBinary
            self.exportLoraBinary = exportLoraBinary
            self.datasetPrepBinary = datasetPrepBinary
            self.baseModelPath = baseModelPath
            self.loraOutPath = loraOutPath
            self.adamIterations = adamIterations
            self.dryRun = dryRun
        }
    }

    /// Outcome of one training cycle. The v1 terminal set returned by run()
    /// is {skippedInsufficientTraces, skippedNoModel, dryRun, trainingFailed,
    /// trainingCompleted, exportCompleted}; datasetPrepared and the guard
    /// cases are reserved for the wired capability-eval gate (plug-in point).
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
        public let datasetPath: String?
        public let loraPath: String?
        public let mergedModelPath: String?
        public let note: String

        public init(
            timestamp: Date = Date(),
            outcome: TrainingOutcome,
            traceCount: Int,
            datasetPath: String? = nil,
            loraPath: String? = nil,
            mergedModelPath: String? = nil,
            note: String
        ) {
            self.timestamp = timestamp
            self.outcome = outcome
            self.traceCount = traceCount
            self.datasetPath = datasetPath
            self.loraPath = loraPath
            self.mergedModelPath = mergedModelPath
            self.note = note
        }
    }

    public let traceStore: TesseraTraceStore
    private let config: Config
    private let store: TesseraLearningStore
    private let lock = NSLock()
    private static let recordFile = "training-record.json"

    public init(config: Config = Config(), traceStore: TesseraTraceStore = TesseraTraceStore()) {
        self.config = config
        self.traceStore = traceStore
        self.store = TesseraLearningStore()
    }

    /// The most recent persisted record, if any.
    public func lastTraining() -> TrainingRecord? {
        lock.lock(); defer { lock.unlock() }
        return store.load(TrainingRecord?.self, from: Self.recordFile, default: nil)
    }

    /// Run one training cycle. `overrideDryRun` wins when non-nil; otherwise
    /// the configured dryRun applies. Every branch persists and returns an
    /// honest record - nothing is fabricated.
    public func run(overrideDryRun: Bool? = nil) async -> TrainingRecord {
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

        // Gate 2: a base model to fine-tune.
        guard let base = config.baseModelPath, !base.isEmpty else {
            return finish(TrainingRecord(
                outcome: .skippedNoModel,
                traceCount: traceCount,
                note: "learning.baseModelPath is not set; training disabled"
            ))
        }

        // Dry run: record intent without shelling out to anything, so it
        // stays meaningful even when the CLI tools are not installed.
        if dryRun {
            let lora = resolvedLoraPath(base: base)
            return finish(TrainingRecord(
                outcome: .dryRun,
                traceCount: traceCount,
                loraPath: lora,
                note: "would prepare a dataset from \(traceCount) trace(s), fine-tune a LoRA on \(base) "
                    + "(adam-iter \(config.adamIterations)), and export the merged model; dry run, nothing executed"
            ))
        }

        // Stage all trace files into one JSONL the dataset op can read.
        let staged: String
        do {
            staged = try stageCombinedTraces()
        } catch {
            return finish(TrainingRecord(
                outcome: .trainingFailed,
                traceCount: traceCount,
                note: "could not stage traces for dataset prep: \(error.localizedDescription)"
            ))
        }
        defer { try? FileManager.default.removeItem(atPath: staged) }

        let datasetPath = Self.datasetPath()

        // Step 3: dataset prep (llama-quantize tessera-dataset op).
        let prep = await runStep(binary: config.datasetPrepBinary, arguments: [
            "--tessera-dataset", staged,
            "--tessera-dataset-out", datasetPath,
        ])
        if let failure = failureNote("dataset prep", prep) {
            return finish(TrainingRecord(
                outcome: .trainingFailed, traceCount: traceCount, note: failure
            ))
        }

        // Step 5: fine-tune the LoRA adapter.
        let lora = resolvedLoraPath(base: base)
        let finetune = await runStep(binary: config.finetuneBinary, arguments: [
            "--model-base", base,
            "--train-data", datasetPath,
            "--lora-out", lora,
            "--adam-iter", String(config.adamIterations),
        ])
        if let failure = failureNote("finetune", finetune) {
            return finish(TrainingRecord(
                outcome: .trainingFailed, traceCount: traceCount, datasetPath: datasetPath, note: failure
            ))
        }

        // Step 6: merge the adapter into an exportable model. A failed export
        // still reports trainingCompleted - the LoRA exists, the merge did not.
        let merged = mergedPath(base: base)
        let export = await runStep(binary: config.exportLoraBinary, arguments: [
            "-m", base,
            "--lora", lora,
            "-o", merged,
        ])
        if let failure = failureNote("export-lora", export) {
            return finish(TrainingRecord(
                outcome: .trainingCompleted, traceCount: traceCount,
                datasetPath: datasetPath, loraPath: lora,
                note: "LoRA trained; export did not complete: \(failure)"
            ))
        }

        return finish(TrainingRecord(
            outcome: .exportCompleted, traceCount: traceCount,
            datasetPath: datasetPath, loraPath: lora, mergedModelPath: merged,
            note: "dataset prepared, LoRA trained (adam-iter \(config.adamIterations)), merged model exported"
        ))
    }

    // MARK: - Shell-out

    private func runStep(binary: String, arguments: [String]) async -> Result<ProcessResult, Error> {
        do {
            return .success(try await ProcessRunner().run(executable: binary, arguments: arguments))
        } catch {
            return .failure(error)
        }
    }

    /// nil on success (exit 0); an honest note on a non-zero exit or a process
    /// that could not start (missing binary).
    private func failureNote(_ step: String, _ result: Result<ProcessResult, Error>) -> String? {
        switch result {
        case .failure(let error):
            return "\(step) unavailable (\(error.localizedDescription))"
        case .success(let res) where res.exitCode != 0:
            let detail = res.stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            return "\(step) exited \(res.exitCode)\(detail.isEmpty ? "" : ": " + String(detail.prefix(200)))"
        case .success:
            return nil
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

    private static func datasetPath() -> String {
        let dir = TesseraLearningStore.defaultDirectory().appendingPathComponent("training", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("train-\(stamp(Date())).txt").path
    }

    private func resolvedLoraPath(base: String) -> String {
        if let configured = config.loraOutPath, !configured.isEmpty { return configured }
        let url = URL(fileURLWithPath: base)
        let stem = url.deletingPathExtension().lastPathComponent
        return url.deletingLastPathComponent()
            .appendingPathComponent("\(stem)-tessera-lora.gguf").path
    }

    private func mergedPath(base: String) -> String {
        let url = URL(fileURLWithPath: base)
        let stem = url.deletingPathExtension().lastPathComponent
        return url.deletingLastPathComponent()
            .appendingPathComponent("\(stem)-tessera-merged.gguf").path
    }

    private static func stamp(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        return formatter.string(from: date)
    }

    // MARK: - Persistence

    private func finish(_ record: TrainingRecord) -> TrainingRecord {
        lock.lock(); defer { lock.unlock() }
        try? store.save(record, to: Self.recordFile)
        return record
    }
}
