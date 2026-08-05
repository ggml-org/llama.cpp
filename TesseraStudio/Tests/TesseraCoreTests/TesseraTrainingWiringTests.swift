import XCTest
@testable import TesseraCore

// Tests for the tessera-train-lk wiring of the learning loop: the pure
// argument construction, the graceful missing-binary path, the --dry-run
// passthrough, and the shell-out capture (via a fake driver script, so no
// real binary or model is needed).

final class TesseraTrainingWiringTests: XCTestCase {

    // MARK: - Argument construction (pure)

    func testTrainArgumentsRealRun() {
        let args = TesseraTrainingOrchestrator.trainArguments(
            traces: "/tmp/traces.jsonl",
            model: "/models/drafter.gguf",
            out: "/models/drafter-trained.gguf",
            maxExamples: 512,
            dryRun: false
        )
        XCTAssertEqual(args, [
            "-m", "/models/drafter.gguf",
            "--traces", "/tmp/traces.jsonl",
            "-o", "/models/drafter-trained.gguf",
            "--max-examples", "512",
        ])
    }

    func testTrainArgumentsDryRunPassthrough() {
        let args = TesseraTrainingOrchestrator.trainArguments(
            traces: "/tmp/traces.jsonl",
            model: "/models/drafter.gguf",
            out: "/models/drafter-trained.gguf",
            maxExamples: 64,
            dryRun: true
        )
        XCTAssertEqual(args.last, "--dry-run")
        XCTAssertEqual(args, [
            "-m", "/models/drafter.gguf",
            "--traces", "/tmp/traces.jsonl",
            "-o", "/models/drafter-trained.gguf",
            "--max-examples", "64",
            "--dry-run",
        ])
    }

    func testMissingBinaryNoteIsActionable() {
        let note = TesseraTrainingOrchestrator.missingBinaryNote(path: "/usr/local/bin/tessera-train-lk")
        XCTAssertTrue(note.contains("/usr/local/bin/tessera-train-lk"))
        XCTAssertTrue(note.contains("cmake --build build --target tessera-train-lk"))
    }

    // MARK: - Run gates and failure paths

    func testInsufficientTracesSkipsWithoutBinary() async throws {
        let root = try makeTempRoot()
        let binary = root.appendingPathComponent("does-not-exist").path
        let orchestrator = try makeOrchestrator(root: root, binary: binary, minTraces: 5)
        let record = await orchestrator.run(overrideDryRun: false)
        XCTAssertEqual(record.outcome, .skippedInsufficientTraces)
    }

    func testMissingModelSkips() async throws {
        let root = try makeTempRoot()
        let orchestrator = try makeOrchestrator(root: root, binary: "/does/not/matter", baseModel: nil)
        let record = await orchestrator.run(overrideDryRun: false)
        XCTAssertEqual(record.outcome, .skippedNoModel)
        XCTAssertTrue(record.note.contains("learning.baseModelPath"))
    }

    func testMissingBinaryFailsWithActionableNote() async throws {
        let root = try makeTempRoot()
        let binary = root.appendingPathComponent("no-such-tessera-train-lk").path
        let baseModel = root.appendingPathComponent("drafter.gguf").path
        let orchestrator = try makeOrchestrator(root: root, binary: binary, baseModel: baseModel)
        let record = await orchestrator.run(overrideDryRun: false)
        XCTAssertEqual(record.outcome, .trainingFailed)
        XCTAssertTrue(record.note.contains(binary), "note should name the expected path")
        XCTAssertTrue(record.note.contains("cmake"), "note should carry the build command")
    }

    func testNonZeroDriverExitFailsAndCapturesStderr() async throws {
        let root = try makeTempRoot()
        let binary = try writeFakeDriver("echo boom >&2\nexit 3\n", in: root)
        let baseModel = root.appendingPathComponent("drafter.gguf").path
        let orchestrator = try makeOrchestrator(root: root, binary: binary, baseModel: baseModel)
        let record = await orchestrator.run(overrideDryRun: false)
        XCTAssertEqual(record.outcome, .trainingFailed)
        XCTAssertTrue(record.note.contains("exited 3"))
        XCTAssertEqual(record.stderr, "boom\n")
    }

    // MARK: - Shell-out with a fake driver

    func testDryRunPassthroughReachesTheDriver() async throws {
        let root = try makeTempRoot()
        let binary = try writeFakeDriver("printf '%s\\n' \"$@\"\n", in: root)
        let baseModel = root.appendingPathComponent("drafter.gguf").path
        let orchestrator = try makeOrchestrator(root: root, binary: binary, baseModel: baseModel)

        let record = await orchestrator.run(overrideDryRun: true, maxExamples: 7)

        XCTAssertEqual(record.outcome, .dryRun)
        XCTAssertNil(record.drafterPath, "a dry run must not report a trained drafter")
        let lines = (record.stdout ?? "").components(separatedBy: "\n")
        XCTAssertTrue(lines.contains("--dry-run"))
        XCTAssertTrue(lines.contains("-m"))
        XCTAssertTrue(lines.contains(baseModel))
        // The max_examples override reaches the driver as --max-examples.
        guard let capIndex = lines.firstIndex(of: "--max-examples") else {
            return XCTFail("--max-examples missing from driver argv: \(lines)")
        }
        XCTAssertEqual(lines[capIndex + 1], "7")
        // The staged traces file is cleaned up after the run.
        guard let tracesIndex = lines.firstIndex(of: "--traces") else {
            return XCTFail("--traces missing from driver argv: \(lines)")
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: lines[tracesIndex + 1]))
    }

    func testRealRunCompletesAndNamesTheTrainedDrafter() async throws {
        let root = try makeTempRoot()
        let binary = try writeFakeDriver("printf '%s\\n' \"$@\"\n", in: root)
        let baseModel = root.appendingPathComponent("drafter.gguf").path
        let orchestrator = try makeOrchestrator(root: root, binary: binary, baseModel: baseModel)

        let record = await orchestrator.run(overrideDryRun: false)

        XCTAssertEqual(record.outcome, .trainingCompleted)
        XCTAssertEqual(
            record.drafterPath,
            root.appendingPathComponent("drafter-tessera-trained.gguf").path
        )
        XCTAssertTrue(record.note.contains("tessera-train-lk"))
    }

    // MARK: - Driver output parsing (pure)

    func testParseDatasetLine() {
        let event = TesseraTrainingOrchestrator.parseDriverLine(
            "dataset: 512 examples, dense-label memory ~123.4 MiB"
        )
        guard case let .datasetBuilt(examples, memoryMiB)? = event else {
            return XCTFail("expected .datasetBuilt, got \(String(describing: event))")
        }
        XCTAssertEqual(examples, 512)
        XCTAssertEqual(memoryMiB, 123.4, accuracy: 0.001)
    }

    func testParseEpochLine() {
        let event = TesseraTrainingOrchestrator.parseDriverLine(
            "epoch 3: train LK loss 0.012345, top-1 agreement 0.9876"
        )
        guard case let .epoch(index, loss, agreement)? = event else {
            return XCTFail("expected .epoch, got \(String(describing: event))")
        }
        XCTAssertEqual(index, 3)
        XCTAssertEqual(loss, 0.012345, accuracy: 0.000001)
        XCTAssertEqual(agreement, 0.9876, accuracy: 0.00001)
    }

    func testParseIgnoresUnrelatedDriverLines() {
        XCTAssertNil(TesseraTrainingOrchestrator.parseDriverLine("llama_model_loader: loaded 123 tensors"))
        XCTAssertNil(TesseraTrainingOrchestrator.parseDriverLine(""))
        XCTAssertNil(TesseraTrainingOrchestrator.parseDriverLine("epoch without numbers"))
        XCTAssertNil(TesseraTrainingOrchestrator.parseDriverLine("dataset: no counts here"))
    }

    // MARK: - Tolerant record decoding

    func testUnknownLegacyOutcomeDecodesAsFailed() throws {
        let json = #"{"timestamp":0,"outcome":"datasetPrepared","traceCount":3,"note":"old pipeline"}"#
        let record = try JSONDecoder().decode(
            TesseraTrainingOrchestrator.TrainingRecord.self,
            from: Data(json.utf8)
        )
        XCTAssertEqual(record.outcome, .trainingFailed)
        XCTAssertEqual(record.traceCount, 3)
    }

    func testKnownOutcomeRoundTrips() throws {
        let json = #"{"timestamp":0,"outcome":"dryRun","traceCount":7,"note":"dry"}"#
        let record = try JSONDecoder().decode(
            TesseraTrainingOrchestrator.TrainingRecord.self,
            from: Data(json.utf8)
        )
        XCTAssertEqual(record.outcome, .dryRun)
    }

    // MARK: - Streaming

    func testStreamingYieldsSingleFinishedEventWhenGated() async throws {
        let root = try makeTempRoot()
        let orchestrator = try makeOrchestrator(
            root: root, binary: "/does/not/matter", baseModel: nil, minTraces: 5
        )
        var events: [TesseraTrainingOrchestrator.TrainingEvent] = []
        for await event in orchestrator.runStreaming() {
            events.append(event)
        }
        XCTAssertEqual(events.count, 1)
        guard case let .finished(record)? = events.last else {
            return XCTFail("stream must end with .finished")
        }
        XCTAssertEqual(record.outcome, .skippedInsufficientTraces)
    }

    func testStreamingYieldsLiveProgressThenFinished() async throws {
        let root = try makeTempRoot()
        let binary = try writeFakeDriver(
            "echo 'dataset: 512 examples, dense-label memory ~123.4 MiB'\n"
            + "echo 'epoch 0: train LK loss 0.500000, top-1 agreement 0.7500'\n"
            + "echo 'epoch 1: train LK loss 0.250000, top-1 agreement 0.8750'\n",
            in: root
        )
        let baseModel = root.appendingPathComponent("drafter.gguf").path
        let orchestrator = try makeOrchestrator(root: root, binary: binary, baseModel: baseModel)

        var events: [TesseraTrainingOrchestrator.TrainingEvent] = []
        for await event in orchestrator.runStreaming(overrideDryRun: false) {
            events.append(event)
        }

        guard case .starting? = events.first else {
            return XCTFail("stream must start with .starting, got \(events.first.map(String.init(describing:)) ?? "nil")")
        }
        XCTAssertTrue(events.contains { if case .datasetBuilt(512, _) = $0 { true } else { false } },
                      "expected the datasetBuilt event, got \(events)")
        XCTAssertTrue(events.contains { if case .epoch(1, _, _) = $0 { true } else { false } },
                      "expected the epoch 1 event, got \(events)")
        guard case let .finished(record)? = events.last else {
            return XCTFail("stream must end with .finished")
        }
        XCTAssertEqual(record.outcome, .trainingCompleted)
        XCTAssertTrue((record.stdout ?? "").contains("epoch 1"))
    }

    // MARK: - Trace producer (collect_training_traces)

    func testImatrixResolverDerivesFromTrainBinaryDirectory() {
        let resolved = TesseraTrainBinaryResolver.resolveImatrix(
            trainOverride: "/tmp/somewhere/tessera-train-lk",
            isExecutable: { $0 == "/tmp/somewhere/llama-imatrix" }
        )
        XCTAssertEqual(resolved, "/tmp/somewhere/llama-imatrix")
    }

    func testImatrixResolverFallsBackToKnownLocations() {
        let known = TesseraTrainBinaryResolver.imatrixKnownLocations[1]
        let resolved = TesseraTrainBinaryResolver.resolveImatrix(
            trainOverride: "/tmp/somewhere/tessera-train-lk",
            isExecutable: { $0 == known }
        )
        XCTAssertEqual(resolved, known)
    }

    func testImatrixResolverReturnsDerivedPathWhenNothingFound() {
        // Nothing executable anywhere: the resolver still returns a path so
        // the failure message names exactly what is missing.
        let resolved = TesseraTrainBinaryResolver.resolveImatrix(
            trainOverride: "",
            isExecutable: { _ in false }
        )
        XCTAssertEqual(resolved, "/usr/local/bin/llama-imatrix")
    }

    func testMissingImatrixNoteIsActionable() {
        let note = CollectTrainingTracesTool.missingImatrixNote(path: "/usr/local/bin/llama-imatrix")
        XCTAssertTrue(note.contains("/usr/local/bin/llama-imatrix"))
        XCTAssertTrue(note.contains("cmake --build build --target llama-imatrix"))
    }

    func testTraceHarvestAppendsEmittedRecords() async throws {
        let root = try makeTempRoot()
        let binDir = root.appendingPathComponent("bin", isDirectory: true)
        try FileManager.default.createDirectory(at: binDir, withIntermediateDirectories: true)
        // The train binary only anchors the directory; the fake imatrix does
        // the work: write two telemetry records to --telemetry-out.
        try "placeholder\n".write(
            to: binDir.appendingPathComponent("tessera-train-lk"),
            atomically: true, encoding: .utf8
        )
        let fakeImatrix = binDir.appendingPathComponent("llama-imatrix").path
        try """
        #!/bin/sh
        out=""
        while [ $# -gt 0 ]; do
          if [ "$1" = "--telemetry-out" ]; then out="$2"; fi
          shift
        done
        printf '%s\\n' '{"kind":"spec","step":1}' '{"kind":"spec","step":2}' > "$out"
        """.write(toFile: fakeImatrix, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fakeImatrix)

        let corpus = root.appendingPathComponent("corpus.txt")
        try "some calibration text\n".write(to: corpus, atomically: true, encoding: .utf8)
        let tracesDir = root.appendingPathComponent("traces", isDirectory: true)

        var tool = CollectTrainingTracesTool()
        tool.traceStoreDirectory = tracesDir

        let result = await withSettings([
            TesseraSettingsKey.learningTrainBinary: binDir.appendingPathComponent("tessera-train-lk").path,
        ]) {
            await (try? tool.execute(arguments: [
                "model_path": .string(root.appendingPathComponent("trunk.gguf").path),
                "draft_model_path": .string(root.appendingPathComponent("drafter.gguf").path),
                "corpus_path": .string(corpus.path),
            ])) ?? .fail("execute threw")
        }

        XCTAssertTrue(result.success, "harvest should succeed, got: \(result.error ?? "")")
        XCTAssertTrue(result.output.contains("Collected 2 trace record(s)"))
        XCTAssertEqual(TesseraTraceStore(directory: tracesDir).totalRecords(), 2)
    }

    func testTraceHarvestRequiresArguments() async throws {
        // Hermetic argument gates: no process is launched for a missing
        // required path. (The missing-binary branch itself depends on the
        // real llama-imatrix being absent, so it is covered by the pure
        // resolver + note tests above instead.)
        let tool = CollectTrainingTracesTool()
        let result = await try tool.execute(arguments: [:])
        XCTAssertFalse(result.success)
        XCTAssertEqual(result.error, "model_path is required")

        let noDraft = await try tool.execute(arguments: [
            "model_path": .string("/m.gguf"),
        ])
        XCTAssertEqual(noDraft.error, "draft_model_path is required")

        let noCorpus = await try tool.execute(arguments: [
            "model_path": .string("/m.gguf"),
            "draft_model_path": .string("/d.gguf"),
        ])
        XCTAssertEqual(noCorpus.error, "corpus_path is required")
    }

    // MARK: - Driver binary resolution

    func testResolverOverrideWinsEvenWhenMissing() {
        let resolved = TesseraTrainBinaryResolver.resolve(
            override: "/custom/spot/tessera-train-lk",
            isExecutable: { _ in false }
        )
        XCTAssertEqual(resolved, "/custom/spot/tessera-train-lk")
    }

    func testResolverWhitespaceOverrideFallsBackToAuto() {
        let resolved = TesseraTrainBinaryResolver.resolve(
            override: "   ",
            isExecutable: { _ in false }
        )
        XCTAssertEqual(resolved, TesseraTrainBinaryResolver.expectedLocation)
    }

    func testResolverPrefersEarlierKnownLocation() {
        let resolved = TesseraTrainBinaryResolver.resolve(
            override: "",
            isExecutable: { _ in true }
        )
        XCTAssertEqual(resolved, TesseraTrainBinaryResolver.knownLocations[0])
    }

    func testResolverSkipsNonExecutableKnownLocations() {
        let first = TesseraTrainBinaryResolver.knownLocations[0]
        let resolved = TesseraTrainBinaryResolver.resolve(
            override: "",
            isExecutable: { $0 != first }
        )
        XCTAssertEqual(resolved, TesseraTrainBinaryResolver.knownLocations[1])
    }

    func testResolverFallsBackToExpectedLocation() {
        let resolved = TesseraTrainBinaryResolver.resolve(
            override: "",
            isExecutable: { _ in false }
        )
        XCTAssertEqual(resolved, TesseraTrainBinaryResolver.expectedLocation)
        XCTAssertTrue(TesseraTrainBinaryResolver.knownLocations.contains(TesseraTrainBinaryResolver.expectedLocation))
    }

    // MARK: - Idle scheduler

    func testIdleSweepSkipsWhenAutoTrainDisabled() async throws {
        let root = try makeTempRoot()
        let orchestrator = try makeOrchestrator(root: root, binary: "/does/not/matter", baseModel: nil)
        let scheduler = TesseraTrainingScheduler(orchestrator: orchestrator)
        let record = await withSettings([TesseraSettingsKey.learningAutoTrain: false]) {
            await scheduler.sweep()
        }
        XCTAssertNil(record, "auto-train off must stop the sweep before touching the orchestrator")
    }

    func testIdleSweepRunsTheOrchestratorGates() async throws {
        let root = try makeTempRoot()
        // One seeded trace with a min-traces gate of 5 -> the sweep reaches the
        // orchestrator and comes back with the honest gate record.
        let orchestrator = try makeOrchestrator(root: root, binary: "/does/not/matter", baseModel: nil, minTraces: 5)
        let scheduler = TesseraTrainingScheduler(orchestrator: orchestrator)
        let finished = LockedFlag()
        scheduler.onFinished = { _ in finished.set() }

        let record = await withSettings([
            TesseraSettingsKey.learningAutoTrain: true,
            TesseraSettingsKey.learningOnPowerOnly: false,
        ]) {
            await scheduler.sweep()
        }

        XCTAssertEqual(record?.outcome, .skippedInsufficientTraces)
        XCTAssertTrue(finished.isSet, "onFinished must fire for every sweep that reaches the orchestrator")
    }

    // MARK: - Helpers

    /// Run body with temporary UserDefaults overrides, restoring prior values after.
    private func withSettings<T>(
        _ overrides: [String: Any],
        _ body: () async -> T
    ) async -> T {
        let saved: [(String, Any?)] = overrides.keys.map { ($0, UserDefaults.standard.object(forKey: $0)) }
        for (key, value) in overrides { UserDefaults.standard.set(value, forKey: key) }
        let result = await body()
        for (key, value) in saved {
            if let value { UserDefaults.standard.set(value, forKey: key) } else { UserDefaults.standard.removeObject(forKey: key) }
        }
        return result
    }

    private final class LockedFlag: @unchecked Sendable {
        private let lock = NSLock()
        private var flag = false
        func set() { lock.lock(); flag = true; lock.unlock() }
        var isSet: Bool { lock.lock(); defer { lock.unlock() }; return flag }
    }

    private func makeTempRoot() throws -> URL {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("tessera-training-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }

    /// One trace record so the min-traces gate passes by default.
    private func seedTraces(in root: URL) throws -> URL {
        let tracesDir = root.appendingPathComponent("traces", isDirectory: true)
        try FileManager.default.createDirectory(at: tracesDir, withIntermediateDirectories: true)
        try "{}\n".write(
            to: tracesDir.appendingPathComponent("traces-test.jsonl"),
            atomically: true, encoding: .utf8
        )
        return tracesDir
    }

    private func writeFakeDriver(_ body: String, in root: URL) throws -> String {
        let path = root.appendingPathComponent("fake-tessera-train-lk").path
        try ("#!/bin/sh\n" + body).write(toFile: path, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: path)
        return path
    }

    private func makeOrchestrator(
        root: URL,
        binary: String,
        baseModel: String? = nil,
        minTraces: Int = 1
    ) throws -> TesseraTrainingOrchestrator {
        let tracesDir = try seedTraces(in: root)
        return TesseraTrainingOrchestrator(
            config: TesseraTrainingOrchestrator.Config(
                minTracesForTraining: minTraces,
                trainBinary: binary,
                baseModelPath: baseModel,
                dryRun: false
            ),
            traceStore: TesseraTraceStore(directory: tracesDir),
            storeDirectory: root.appendingPathComponent("store", isDirectory: true)
        )
    }
}
