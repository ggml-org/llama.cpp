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

    /// Outcome of one training cycle. The terminal set returned today is
    /// {skippedInsufficientTraces, skippedNoModel, dryRun, trainingFailed,
    /// trainingCompleted}; the guard cases are reserved for the wired
    /// capability-eval gate (plug-in point).
    public enum TrainingOutcome: String, Codable, Sendable {
        case skippedInsufficientTraces
        case skippedNoModel
        case trainingCompleted
        case trainingFailed
        case guardPassed
        case guardFailed
        case dryRun

        /// Tolerant decode: records persisted by the old three-step pipeline
        /// carried outcomes that no longer exist. Map anything unknown to a
        /// failed run rather than throwing or silently dropping the record.
        public init(from decoder: Decoder) throws {
            let raw = try decoder.singleValueContainer().decode(String.self)
            self = TrainingOutcome(rawValue: raw) ?? .trainingFailed
        }
    }

    /// A live signal from one training cycle. The UI consumes this to show
    /// progress as it happens instead of blocking until the driver exits.
    public enum TrainingEvent: Sendable {
        case starting(traceCount: Int, dryRun: Bool)
        case datasetBuilt(examples: Int, memoryMiB: Double)
        case epoch(index: Int, loss: Double, agreement: Double)
        case finished(TrainingRecord)
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

    /// A driver invocation that passed the pre-flight gates and is ready to run.
    private struct Prepared {
        let stagedTraces: String
        let outPath: String
        let arguments: [String]
        let dryRun: Bool
        let traceCount: Int
        let base: String
        let maxExamples: Int
    }

    private enum GateResult {
        case finished(TrainingRecord)
        case prepared(Prepared)
    }

    /// Run one training cycle to completion and return the terminal record.
    /// `overrideDryRun` wins when non-nil; otherwise the configured dryRun
    /// applies. `maxExamples` overrides the dataset cap when non-nil. Every
    /// branch persists and returns an honest record - nothing is fabricated.
    /// For live progress, consume runStreaming() instead.
    public func run(overrideDryRun: Bool? = nil, maxExamples: Int? = nil) async -> TrainingRecord {
        switch prepare(overrideDryRun: overrideDryRun, maxExamples: maxExamples) {
        case .finished(let record):
            return record
        case .prepared(let prep):
            return await runPrepared(prep)
        }
    }

    /// Run one training cycle, streaming live progress events. The stream
    /// always ends with a single .finished(record). Cancel the consuming task
    /// to terminate the driver mid-run.
    public func runStreaming(overrideDryRun: Bool? = nil, maxExamples: Int? = nil) -> AsyncStream<TrainingEvent> {
        AsyncStream { continuation in
            let task = Task { [weak self] in
                guard let self else { continuation.finish(); return }
                switch self.prepare(overrideDryRun: overrideDryRun, maxExamples: maxExamples) {
                case .finished(let record):
                    continuation.yield(.finished(record))
                    continuation.finish()
                case .prepared(let prep):
                    continuation.yield(.starting(traceCount: prep.traceCount, dryRun: prep.dryRun))
                    await self.streamDriver(prep, continuation: continuation)
                    continuation.finish()
                }
            }
            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }

    /// Run the pre-flight gates and stage the traces. Returns either an
    /// already-persisted terminal record (a skip or early failure) or a
    /// prepared invocation for the driver. Shared by run() and runStreaming()
    /// so the two paths cannot drift.
    private func prepare(overrideDryRun: Bool?, maxExamples: Int?) -> GateResult {
        let dryRun = overrideDryRun ?? config.dryRun
        let traceCount = traceStore.totalRecords()

        // Gate 1: enough traces to form a dataset.
        guard traceCount >= config.minTracesForTraining else {
            return .finished(finish(TrainingRecord(
                outcome: .skippedInsufficientTraces,
                traceCount: traceCount,
                note: "have \(traceCount) trace record(s), need \(config.minTracesForTraining); collect more traces first"
            )))
        }

        // Gate 2: a drafter model to train.
        guard let base = config.baseModelPath, !base.isEmpty else {
            return .finished(finish(TrainingRecord(
                outcome: .skippedNoModel,
                traceCount: traceCount,
                note: "learning.baseModelPath is not set; training disabled"
            )))
        }

        // Gate 3: the native driver binary, checked up front so a missing
        // build is an actionable message, not a bare process-launch error.
        guard FileManager.default.isExecutableFile(atPath: config.trainBinary) else {
            return .finished(finish(TrainingRecord(
                outcome: .trainingFailed,
                traceCount: traceCount,
                note: Self.missingBinaryNote(path: config.trainBinary)
            )))
        }

        let cap = maxExamples ?? config.maxExamples
        guard cap > 0 else {
            return .finished(finish(TrainingRecord(
                outcome: .trainingFailed,
                traceCount: traceCount,
                note: "max-examples must be > 0 (got \(cap))"
            )))
        }

        // Stage all trace files into one JSONL the driver reads.
        let staged: String
        do {
            staged = try stageCombinedTraces()
        } catch {
            return .finished(finish(TrainingRecord(
                outcome: .trainingFailed,
                traceCount: traceCount,
                note: "could not stage traces for tessera-train-lk: \(error.localizedDescription)"
            )))
        }

        let out = resolvedDrafterPath(base: base)
        let arguments = Self.trainArguments(
            traces: staged, model: base, out: out, maxExamples: cap, dryRun: dryRun
        )
        return .prepared(Prepared(
            stagedTraces: staged, outPath: out, arguments: arguments,
            dryRun: dryRun, traceCount: traceCount, base: base, maxExamples: cap
        ))
    }

    private func runPrepared(_ prep: Prepared) async -> TrainingRecord {
        defer { try? FileManager.default.removeItem(atPath: prep.stagedTraces) }

        let result = await runStep(binary: config.trainBinary, arguments: prep.arguments)
        switch result {
        case .failure(let error):
            return finish(TrainingRecord(
                outcome: .trainingFailed,
                traceCount: prep.traceCount,
                note: "tessera-train-lk could not start (\(error.localizedDescription)); expected at \(config.trainBinary)"
            ))
        case .success(let res) where res.exitCode != 0:
            return finish(TrainingRecord(
                outcome: .trainingFailed,
                traceCount: prep.traceCount,
                stdout: Self.capped(res.stdout),
                stderr: Self.capped(res.stderr),
                note: "tessera-train-lk exited \(res.exitCode)"
            ))
        case .success(let res) where prep.dryRun:
            return finish(TrainingRecord(
                outcome: .dryRun,
                traceCount: prep.traceCount,
                stdout: Self.capped(res.stdout),
                stderr: Self.capped(res.stderr),
                note: "dry run: tessera-train-lk built the dataset from \(prep.traceCount) trace(s) against \(prep.base); nothing trained or saved"
            ))
        case .success(let res):
            return finish(TrainingRecord(
                outcome: .trainingCompleted,
                traceCount: prep.traceCount,
                drafterPath: prep.outPath,
                stdout: Self.capped(res.stdout),
                stderr: Self.capped(res.stderr),
                note: "drafter trained with tessera-train-lk (max-examples \(prep.maxExamples)) and saved to \(prep.outPath)"
            ))
        }
    }

    /// Drive the streaming process, parse live progress, accumulate the
    /// output for the terminal record, and yield .finished at the end.
    private func streamDriver(_ prep: Prepared, continuation: AsyncStream<TrainingEvent>.Continuation) async {
        var stdoutAll = ""
        var stderrAll = ""
        var lineBuffer = ""
        var exitCode: Int32 = -1

        let stream = ProcessRunner().runStreamingCombined(
            executable: config.trainBinary, arguments: prep.arguments
        )
        do {
            for try await chunk in stream {
                if Task.isCancelled { break }
                switch chunk {
                case .output(.stdout, let text):
                    stdoutAll += text
                    lineBuffer += text
                    while let newline = lineBuffer.firstIndex(of: "\n") {
                        let line = String(lineBuffer[lineBuffer.startIndex..<newline])
                        lineBuffer.removeSubrange(lineBuffer.startIndex...newline)
                        if let event = Self.parseDriverLine(line) {
                            continuation.yield(event)
                        }
                    }
                case .output(.stderr, let text):
                    stderrAll += text
                case .exited(let code):
                    exitCode = code
                }
            }
        } catch {
            // The driver could not start; the terminal record below reports it.
        }
        try? FileManager.default.removeItem(atPath: prep.stagedTraces)

        let record: TrainingRecord
        if Task.isCancelled {
            record = TrainingRecord(
                outcome: .trainingFailed, traceCount: prep.traceCount,
                stdout: Self.capped(stdoutAll), stderr: Self.capped(stderrAll),
                note: "training cancelled before the driver finished"
            )
        } else if exitCode != 0 {
            record = TrainingRecord(
                outcome: .trainingFailed, traceCount: prep.traceCount,
                stdout: Self.capped(stdoutAll), stderr: Self.capped(stderrAll),
                note: exitCode == -1
                    ? "tessera-train-lk could not start; expected at \(config.trainBinary)"
                    : "tessera-train-lk exited \(exitCode)"
            )
        } else if prep.dryRun {
            record = TrainingRecord(
                outcome: .dryRun, traceCount: prep.traceCount,
                stdout: Self.capped(stdoutAll), stderr: Self.capped(stderrAll),
                note: "dry run: tessera-train-lk built the dataset from \(prep.traceCount) trace(s) against \(prep.base); nothing trained or saved"
            )
        } else {
            record = TrainingRecord(
                outcome: .trainingCompleted, traceCount: prep.traceCount,
                drafterPath: prep.outPath,
                stdout: Self.capped(stdoutAll), stderr: Self.capped(stderrAll),
                note: "drafter trained with tessera-train-lk (max-examples \(prep.maxExamples)) and saved to \(prep.outPath)"
            )
        }
        continuation.yield(.finished(finish(record)))
    }

    /// Parse one driver stdout line into a progress event, if it is one.
    static func parseDriverLine(_ raw: String) -> TrainingEvent? {
        let line = raw.trimmingCharacters(in: .whitespaces)
        return parseDatasetLine(line) ?? parseEpochLine(line)
    }

    // "dataset: 512 examples, dense-label memory ~123.4 MiB"
    private static func parseDatasetLine(_ line: String) -> TrainingEvent? {
        guard let range = line.range(of: "dataset:") else { return nil }
        let scanner = Scanner(string: String(line[range.upperBound...]))
        guard let examples = scanner.scanInt(), examples > 0 else { return nil }
        var memoryMiB = 0.0
        if scanner.scanUpToString("~") != nil {
            _ = scanner.scanString("~")
            memoryMiB = scanner.scanDouble() ?? 0.0
        }
        return .datasetBuilt(examples: examples, memoryMiB: memoryMiB)
    }

    // "epoch 3: train LK loss 0.012345, top-1 agreement 0.9876"
    private static func parseEpochLine(_ line: String) -> TrainingEvent? {
        guard let range = line.range(of: "epoch ") else { return nil }
        let scanner = Scanner(string: String(line[range.upperBound...]))
        guard let index = scanner.scanInt() else { return nil }
        guard scanner.scanUpToString("loss") != nil else { return nil }
        _ = scanner.scanString("loss")
        guard let loss = scanner.scanDouble() else { return nil }
        _ = scanner.scanUpToString("agreement")
        _ = scanner.scanString("agreement")
        let agreement = scanner.scanDouble() ?? 0.0
        return .epoch(index: index, loss: loss, agreement: agreement)
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
        // Training consumes calibration + promoted replay records only
        // (runtime-traces spec section 12.5): raw runtime captures enter the
        // pipeline exclusively through the curation stage's replay, and s2s
        // records are Tier B local-only (voice-bearing codes), never dataset
        // fuel (s2s design section 4.2). The runtime and s2s file prefixes
        // are the first egress-filter line; the record guard is the second,
        // so a local-only record is dropped even if it ever lands in a
        // calibration- or replay-named file.
        for file in traceStore.traceFiles()
        where !file.lastPathComponent.hasPrefix(TesseraTraceStore.runtimeFilePrefix)
            && !file.lastPathComponent.hasPrefix(TesseraTraceStore.s2sFilePrefix) {
            guard let data = try? Data(contentsOf: file) else { continue }
            let text = String(decoding: data, as: UTF8.self)
            for line in text.split(separator: "\n", omittingEmptySubsequences: true)
            where TesseraEgressGuard.allows(String(line)) {
                out.append(Data(line.utf8))
                out.append(0x0A)
            }
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
