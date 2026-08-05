import XCTest
@testable import TesseraCore

// Runtime-traces spec sections 8, 9, 12.5 tests: the record-level egress
// guard (second filter line after the runtime file prefix) and the staging
// invariant that no provenance:runtime record ever reaches the dataset.

// MARK: - Guard unit tests

final class TesseraEgressGuardTests: XCTestCase {
    func testCalibrationRecordWithoutProvenancePasses() {
        XCTAssertTrue(TesseraEgressGuard.allows(
            "{\"schema\":\"llama.tessera.spec.v1\",\"step_idx\":0,\"drafted\":3,\"accepted\":2}"))
    }

    func testReplayRecordWithExactPromotionStampPasses() {
        XCTAssertTrue(TesseraEgressGuard.allows(
            "{\"schema\":\"llama.tessera.spec.v1\",\"step_idx\":0,\"provenance\":\"replay\",\"replayed_from\":\"runtime\"}"))
    }

    func testRuntimeRecordDrops() {
        // Spec section 9 invariant: no record with provenance:runtime may
        // reach dataset staging.
        XCTAssertFalse(TesseraEgressGuard.allows(
            "{\"schema\":\"llama.tessera.spec.v1\",\"step_idx\":0,\"provenance\":\"runtime\",\"sid\":\"abc\"}"))
    }

    func testReplayRecordMissingStampDrops() {
        XCTAssertFalse(TesseraEgressGuard.allows(
            "{\"schema\":\"llama.tessera.spec.v1\",\"provenance\":\"replay\"}"))
    }

    func testReplayRecordWithForeignReplayedFromDrops() {
        XCTAssertFalse(TesseraEgressGuard.allows(
            "{\"schema\":\"llama.tessera.spec.v1\",\"provenance\":\"replay\",\"replayed_from\":\"calibration\"}"))
    }

    func testUnknownProvenanceDrops() {
        XCTAssertFalse(TesseraEgressGuard.allows(
            "{\"schema\":\"llama.tessera.spec.v1\",\"provenance\":\"synthetic\"}"))
    }

    func testUnparseableLineMentioningProvenanceDrops() {
        XCTAssertFalse(TesseraEgressGuard.allows("not json \"provenance\" anywhere"))
    }

    func testEmptyAndBlankLinesDrop() {
        XCTAssertFalse(TesseraEgressGuard.allows(""))
        XCTAssertFalse(TesseraEgressGuard.allows("   \n"))
    }

    func testNonStringProvenanceDrops() {
        XCTAssertFalse(TesseraEgressGuard.allows(
            "{\"schema\":\"llama.tessera.spec.v1\",\"provenance\":7}"))
    }
}

// MARK: - Staging integration: the guard is the second filter line

final class TesseraEgressStagingTests: XCTestCase {
    private var roots: [URL] = []

    override func tearDown() {
        for root in roots { try? FileManager.default.removeItem(at: root) }
        roots.removeAll()
        unsetenv("TESSERA_TEST_CAPTURE")
        super.tearDown()
    }

    private func makeTempRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("tessera-egress-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        roots.append(root)
        return root
    }

    // A runtime-provenance record smuggled into a calibration-NAMED file
    // must still drop: the file prefix is only the first filter line.
    func testStagingDropsRuntimeProvenanceInsideCalibrationFile() async throws {
        let root = try makeTempRoot()
        let tracesDir = root.appendingPathComponent("traces")
        try FileManager.default.createDirectory(at: tracesDir, withIntermediateDirectories: true)
        let store = TesseraTraceStore(directory: tracesDir)

        // appendRun stores the source verbatim under a traces-<date>.jsonl
        // (calibration) name, including the smuggled runtime record.
        let calibrationSource = root.appendingPathComponent("calib.jsonl")
        try ("{\"schema\":\"llama.tessera.spec.v1\",\"step_idx\":0,\"drafted\":3,\"accepted\":2}\n"
            + "{\"schema\":\"llama.tessera.spec.v1\",\"step_idx\":1,\"drafted\":3,\"accepted\":2,\"provenance\":\"runtime\",\"sid\":\"smuggled\"}\n")
            .write(to: calibrationSource, atomically: true, encoding: .utf8)
        try store.appendRun(jsonlPath: calibrationSource)
        try store.appendReplay(records: [
            "{\"schema\":\"llama.tessera.spec.v1\",\"step_idx\":0,\"drafted\":3,\"accepted\":2,\"provenance\":\"replay\",\"replayed_from\":\"runtime\"}",
        ])
        try store.appendRuntime(records: [
            "{\"schema\":\"llama.tessera.spec.v1\",\"step_idx\":0,\"drafted\":3,\"accepted\":2,\"provenance\":\"runtime\",\"sid\":\"r1\"}",
        ])
        XCTAssertEqual(store.totalRecords(), 4)

        let capturePath = root.appendingPathComponent("captured-dataset.jsonl").path
        setenv("TESSERA_TEST_CAPTURE", capturePath, 1)
        let driverPath = root.appendingPathComponent("fake-tessera-train-lk").path
        try ("""
        #!/bin/sh
        while [ $# -gt 0 ]; do
          if [ "$1" = "--traces" ]; then cp "$2" "$TESSERA_TEST_CAPTURE"; fi
          shift
        done
        echo 'dataset: 2 examples, dense-label memory ~1 MiB'
        echo 'epoch 0: train LK loss 0.500000, top-1 agreement 0.7500'
        """).write(toFile: driverPath, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: driverPath)

        let baseModel = root.appendingPathComponent("base.gguf")
        try "placeholder\n".write(to: baseModel, atomically: true, encoding: .utf8)

        let orchestrator = TesseraTrainingOrchestrator(
            config: TesseraTrainingOrchestrator.Config(
                minTracesForTraining: 4,
                trainBinary: driverPath,
                baseModelPath: baseModel.path,
                dryRun: false
            ),
            traceStore: store,
            storeDirectory: root.appendingPathComponent("store")
        )

        let record = await orchestrator.run()
        XCTAssertEqual(record.outcome, .trainingCompleted, record.stderr ?? "")

        let captured = try String(contentsOfFile: capturePath, encoding: .utf8)
        XCTAssertFalse(captured.contains("\"provenance\":\"runtime\""),
                       "the record guard must drop runtime-provenance lines even inside calibration-named files")
        XCTAssertFalse(captured.contains("smuggled"))
        XCTAssertTrue(captured.contains("\"provenance\":\"replay\""))
        XCTAssertEqual(captured.split(separator: "\n").count, 2)
    }
}
