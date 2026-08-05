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

    // MARK: - Helpers

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
