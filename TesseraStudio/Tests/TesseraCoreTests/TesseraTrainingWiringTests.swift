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
