import XCTest
@testable import TesseraCore

// Runtime-traces spec section 12 tests: the session reader, replay store
// writes, provenance stamping, and the curation stage sweep (analysis,
// verdicts, dedup, resumable replay).

// MARK: - Fakes

private final class FakeSessionDecoder: TesseraSessionDecoder {
    let nVocab: Int32
    let pieces: [Int32: String]

    init(nVocab: Int32, pieces: [Int32: String]) {
        self.nVocab = nVocab
        self.pieces = pieces
    }

    func detokenize(_ tokens: [Int32]) -> String? {
        tokens.map { pieces[$0] ?? "" }.joined(separator: " ")
    }

    func piece(for token: Int32) -> String? {
        pieces[token]
    }
}

private final class ReplayRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var _corpora: [String] = []
    private var _topks: [Int] = []

    var lines: [String] = ["{\"schema\":\"llama.tessera.spec.v1\",\"step_idx\":0,\"drafted\":3,\"accepted\":2}"]
    var error: Error?

    var corpora: [String] { lock.lock(); defer { lock.unlock() }; return _corpora }
    var topks: [Int] { lock.lock(); defer { lock.unlock() }; return _topks }

    func run(corpus: String, topk: Int) throws -> [String] {
        lock.lock()
        _corpora.append(corpus)
        _topks.append(topk)
        let error = self.error
        let lines = self.lines
        lock.unlock()
        if let error { throw error }
        return lines
    }
}

// MARK: - Record fixtures

private func runtimeStepRecord(
    sid: String, step: Int, drafted: Int, accepted: Int, tokens: [Int32]
) -> String {
    let toks = tokens.map(String.init).joined(separator: ",")
    return "{\"schema\":\"llama.tessera.spec.v1\",\"step_idx\":\(step),"
        + "\"drafted\":\(drafted),\"accepted\":\(accepted),"
        + "\"accepted_tokens\":[\(toks)],\"provenance\":\"runtime\",\"sid\":\"\(sid)\"}"
}

/// Steps of a clean, promotable session: 16 steps x 4 accepted tokens =
/// 64 tokens (meets the floor), acceptance 0.75, unique words (no
/// repetition, no garbage, no probe hits).
private func cleanSessionSteps(sid: String, tokenBase: Int32 = 0) -> [String] {
    var records: [String] = []
    var token = tokenBase
    for step in 0..<16 {
        let toks = (0..<4).map { _ in
            defer { token += 1 }
            return token
        }
        records.append(runtimeStepRecord(
            sid: sid, step: step, drafted: 4, accepted: 3, tokens: toks))
    }
    return records
}

private func cleanSessionPieces(tokenBase: Int32 = 0) -> [Int32: String] {
    var pieces: [Int32: String] = [:]
    for i in 0..<64 {
        pieces[tokenBase + Int32(i)] = "word\(tokenBase + Int32(i))"
    }
    return pieces
}

// MARK: - Session reader

final class TesseraSessionTraceReaderTests: XCTestCase {
    private var dirs: [URL] = []

    override func tearDown() {
        for dir in dirs { try? FileManager.default.removeItem(at: dir) }
        dirs.removeAll()
        super.tearDown()
    }

    private func makeStore() throws -> TesseraTraceStore {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tessera-reader-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        dirs.append(dir)
        return TesseraTraceStore(directory: dir)
    }

    func testGroupsBySidOrdersStepsAndConcatenatesTokens() throws {
        let store = try makeStore()
        try store.appendRuntime(records: [
            runtimeStepRecord(sid: "B", step: 0, drafted: 2, accepted: 1, tokens: [10, 11]),
            runtimeStepRecord(sid: "A", step: 1, drafted: 2, accepted: 2, tokens: [21, 22, 23]),
            runtimeStepRecord(sid: "A", step: 0, drafted: 2, accepted: 0, tokens: [20]),
        ])

        let sessions = TesseraSessionTraceReader.sessions(in: store)
        XCTAssertEqual(sessions.map { $0.sid }, ["B", "A"])

        let a = sessions[1]
        XCTAssertEqual(a.steps.map { $0.stepIdx }, [0, 1])
        XCTAssertEqual(a.acceptedTokens, [20, 21, 22, 23])
        XCTAssertEqual(a.steps[0].drafted, 2)
        XCTAssertEqual(a.steps[1].accepted, 2)
    }

    func testRetriedFlushDedupesStepsKeepingFirst() throws {
        let store = try makeStore()
        try store.appendRuntime(records: [
            runtimeStepRecord(sid: "A", step: 0, drafted: 2, accepted: 1, tokens: [1, 2]),
        ])
        // Same-second append lands in a suffixed file; a retried flush
        // re-writes the identical step plus a fresh one.
        Thread.sleep(forTimeInterval: 1.1)
        try store.appendRuntime(records: [
            runtimeStepRecord(sid: "A", step: 0, drafted: 2, accepted: 1, tokens: [1, 2]),
            runtimeStepRecord(sid: "A", step: 1, drafted: 2, accepted: 2, tokens: [3, 4, 5]),
        ])

        let sessions = TesseraSessionTraceReader.sessions(in: store)
        XCTAssertEqual(sessions.count, 1)
        let steps = sessions[0].steps
        XCTAssertEqual(steps.map { $0.stepIdx }, [0, 1])
        XCTAssertEqual(sessions[0].acceptedTokens, [1, 2, 3, 4, 5])
    }

    func testSkipsSidlessAndMalformedLines() throws {
        let store = try makeStore()
        let file = store.directoryURL
            .appendingPathComponent("traces-runtime-20260101-000000.jsonl")
        try ([
            runtimeStepRecord(sid: "A", step: 0, drafted: 2, accepted: 1, tokens: [1, 2]),
            "{\"schema\":\"llama.tessera.spec.v1\",\"step_idx\":1,\"drafted\":2,\"accepted\":1}",
            "not json at all",
            "",
        ].joined(separator: "\n") + "\n").write(to: file, atomically: true, encoding: .utf8)

        let sessions = TesseraSessionTraceReader.sessions(in: store)
        XCTAssertEqual(sessions.count, 1)
        XCTAssertEqual(sessions[0].steps.count, 1)
    }
}

// MARK: - Trace store: replay writes

final class TesseraTraceStoreReplayTests: XCTestCase {
    private var dirs: [URL] = []

    override func tearDown() {
        for dir in dirs { try? FileManager.default.removeItem(at: dir) }
        dirs.removeAll()
        super.tearDown()
    }

    private func makeStore() throws -> TesseraTraceStore {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tessera-replay-store-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        dirs.append(dir)
        return TesseraTraceStore(directory: dir)
    }

    func testAppendReplayWritesDatedReplayFile() throws {
        let store = try makeStore()
        let records = [
            "{\"schema\":\"llama.tessera.spec.v1\",\"step_idx\":0,\"provenance\":\"replay\",\"replayed_from\":\"runtime\"}",
            "{\"schema\":\"llama.tessera.spec.v1\",\"step_idx\":1,\"provenance\":\"replay\",\"replayed_from\":\"runtime\"}",
        ]
        let url = try store.appendReplay(records: records)
        XCTAssertNotNil(url)
        guard let url else { return }
        XCTAssertTrue(url.lastPathComponent.hasPrefix(TesseraTraceStore.replayFilePrefix))
        XCTAssertTrue(url.lastPathComponent.hasSuffix(".jsonl"))
        XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), records.joined(separator: "\n") + "\n")

        // Replay records count toward the combined training-gate total.
        XCTAssertEqual(store.totalRecords(), 2)
        XCTAssertEqual(store.replayFiles().map { $0.lastPathComponent }, [url.lastPathComponent])
        XCTAssertEqual(store.runtimeFiles().count, 0)
    }

    func testAppendReplayEmptyIsNoop() throws {
        let store = try makeStore()
        XCTAssertNil(try store.appendReplay(records: []))
        XCTAssertEqual(store.replayFiles().count, 0)
    }

    func testAppendReplaySparedByRuntimeRollingCap() throws {
        let store = try makeStore()
        try store.appendReplay(records: ["{\"schema\":\"llama.tessera.spec.v1\",\"step_idx\":0}"])
        try store.appendRuntime(records: [
            "{\"schema\":\"llama.tessera.spec.v1\",\"drafted\":2,\"accepted\":1,\"provenance\":\"runtime\",\"sid\":\"r1\"}",
        ])

        // Zero budget: the runtime file goes, the replay file stays.
        let removed = try store.trimRuntimeToBudget(budgetBytes: 0)
        XCTAssertEqual(removed, 1)
        XCTAssertEqual(store.replayFiles().count, 1)
        XCTAssertEqual(store.runtimeFiles().count, 0)
    }
}

// MARK: - Curation stage

final class TesseraSessionCurationStageTests: XCTestCase {
    private var roots: [URL] = []

    override func tearDown() {
        for root in roots { try? FileManager.default.removeItem(at: root) }
        roots.removeAll()
        super.tearDown()
    }

    /// Layout: root/traces (store), root/curation-ledger.jsonl,
    /// root/session-curation-state.json - matching the app's learning root.
    private func makeStage(
        decoder: TesseraSessionDecoder?,
        recorder: ReplayRecorder,
        trunk: String = "/fake/trunk.gguf"
    ) throws -> (TesseraSessionCurationStage, TesseraTraceStore, TesseraCurationLedger) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("tessera-curation-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        roots.append(root)
        let store = TesseraTraceStore(directory: root.appendingPathComponent("traces"))
        let ledger = TesseraCurationLedger(directory: root)
        let stage = TesseraSessionCurationStage(
            store: store,
            ledger: ledger,
            decoderProvider: { _ in decoder },
            replayDriver: { corpus, topk in try recorder.run(corpus: corpus, topk: topk) },
            trunkPathProvider: { trunk })
        return (stage, store, ledger)
    }

    func testSweepPromotesCleanSessionAndReplays() async throws {
        let recorder = ReplayRecorder()
        let (stage, store, ledger) = try makeStage(
            decoder: FakeSessionDecoder(nVocab: 1000, pieces: cleanSessionPieces()),
            recorder: recorder)
        try store.appendRuntime(records: cleanSessionSteps(sid: "P"))

        let report = await stage.sweep()
        XCTAssertEqual(report.analyzed, 1)
        XCTAssertEqual(report.promoted, 1)
        XCTAssertEqual(report.quarantined, 0)
        XCTAssertEqual(report.dropped, 0)
        XCTAssertEqual(report.replayedSessions, 1)
        XCTAssertEqual(report.replayRecords, 1)
        XCTAssertTrue(report.pendingReplay.isEmpty)

        XCTAssertEqual(ledger.verdict(for: "P"), .promoted)
        XCTAssertEqual(ledger.latestVerdicts()["P"]?.reasons, ["probe:none", "dedup:kept", "low-repetition"])

        // The replay driver saw the decoded corpus at deepened topk.
        XCTAssertEqual(recorder.topks, [TesseraSessionCurationStage.replayTopkDefault])
        XCTAssertEqual(recorder.corpora.count, 1)
        XCTAssertTrue(recorder.corpora[0].contains("word0"))
        XCTAssertTrue(recorder.corpora[0].contains("word63"))

        // Stamped replay records land in a traces-replay- file.
        let replayFiles = store.replayFiles()
        XCTAssertEqual(replayFiles.count, 1)
        let text = try String(contentsOf: replayFiles[0], encoding: .utf8)
        XCTAssertTrue(text.contains("\"provenance\":\"replay\""))
        XCTAssertTrue(text.contains("\"replayed_from\":\"runtime\""))

        // Second sweep: the session is curated; nothing left to do.
        let again = await stage.sweep()
        XCTAssertEqual(again.analyzed, 0)
        XCTAssertEqual(again.note, "no uncurated sessions")
        XCTAssertEqual(recorder.corpora.count, 1)
    }

    func testSweepQuarantinesProbeHitAndSkipsReplay() async throws {
        var pieces = cleanSessionPieces()
        pieces[7] = "user@example.com"
        let recorder = ReplayRecorder()
        let (stage, store, ledger) = try makeStage(
            decoder: FakeSessionDecoder(nVocab: 1000, pieces: pieces),
            recorder: recorder)
        try store.appendRuntime(records: cleanSessionSteps(sid: "Q"))

        let report = await stage.sweep()
        XCTAssertEqual(report.quarantined, 1)
        XCTAssertEqual(report.promoted, 0)
        XCTAssertEqual(ledger.verdict(for: "Q"), .quarantined)
        XCTAssertEqual(ledger.latestVerdicts()["Q"]?.reasons, ["probe:email"])
        XCTAssertEqual(ledger.quarantinedSids(), ["Q"])
        // Quarantined sessions never reach replay.
        XCTAssertTrue(recorder.corpora.isEmpty)
        XCTAssertEqual(store.replayFiles().count, 0)
    }

    func testSweepDropsModelMismatch() async throws {
        let recorder = ReplayRecorder()
        let (stage, store, ledger) = try makeStage(
            decoder: FakeSessionDecoder(nVocab: 50, pieces: cleanSessionPieces()),
            recorder: recorder)
        // Token 500 falls outside the current trunk's vocab (50): the ids no
        // longer decode, so the session drops before anything else is asked.
        try store.appendRuntime(records: cleanSessionSteps(sid: "M"))

        let report = await stage.sweep()
        XCTAssertEqual(report.dropped, 1)
        XCTAssertEqual(report.promoted, 0)
        XCTAssertEqual(ledger.verdict(for: "M"), .dropped)
        XCTAssertEqual(ledger.latestVerdicts()["M"]?.reasons, ["model-mismatch"])
        XCTAssertTrue(recorder.corpora.isEmpty)
    }

    func testSweepCollapsesDuplicates() async throws {
        let recorder = ReplayRecorder()
        let (stage, store, ledger) = try makeStage(
            decoder: FakeSessionDecoder(nVocab: 1000, pieces: cleanSessionPieces()),
            recorder: recorder)
        // Two distinct sids, identical decoded text: the second collapses.
        try store.appendRuntime(records: cleanSessionSteps(sid: "A"))
        Thread.sleep(forTimeInterval: 1.1)
        try store.appendRuntime(records: cleanSessionSteps(sid: "B"))

        let report = await stage.sweep()
        XCTAssertEqual(report.promoted, 1)
        XCTAssertEqual(report.dropped, 1)
        XCTAssertEqual(ledger.verdict(for: "A"), .promoted)
        XCTAssertEqual(ledger.verdict(for: "B"), .dropped)
        XCTAssertEqual(ledger.latestVerdicts()["B"]?.reasons, ["duplicate"])
        // The duplicate's text is not replayed twice.
        XCTAssertEqual(recorder.corpora.count, 1)
        XCTAssertFalse(recorder.corpora[0].contains("\n\n"))
    }

    func testSweepDropsSessionsBelowTokenFloor() async throws {
        let recorder = ReplayRecorder()
        let (stage, store, ledger) = try makeStage(
            decoder: FakeSessionDecoder(nVocab: 1000, pieces: [1: "one", 2: "two"]),
            recorder: recorder)
        try store.appendRuntime(records: [
            runtimeStepRecord(sid: "S", step: 0, drafted: 2, accepted: 1, tokens: [1, 2]),
        ])

        let report = await stage.sweep()
        XCTAssertEqual(report.dropped, 1)
        XCTAssertEqual(ledger.verdict(for: "S"), .dropped)
        XCTAssertEqual(ledger.latestVerdicts()["S"]?.reasons, ["below-token-floor"])
    }

    func testSweepDegradesWithoutDecoder() async throws {
        let recorder = ReplayRecorder()
        let (stage, store, ledger) = try makeStage(decoder: nil, recorder: recorder)
        try store.appendRuntime(records: cleanSessionSteps(sid: "P"))

        let report = await stage.sweep()
        XCTAssertEqual(report.analyzed, 0)
        XCTAssertEqual(report.note, "trunk vocab unavailable; curation deferred")
        XCTAssertTrue(ledger.entries().isEmpty)
        XCTAssertTrue(recorder.corpora.isEmpty)
    }

    func testSweepDegradesWithoutTrunkConfigured() async throws {
        let recorder = ReplayRecorder()
        let (stage, store, ledger) = try makeStage(
            decoder: FakeSessionDecoder(nVocab: 1000, pieces: [:]),
            recorder: recorder, trunk: "")
        try store.appendRuntime(records: cleanSessionSteps(sid: "P"))

        let report = await stage.sweep()
        XCTAssertEqual(report.analyzed, 0)
        XCTAssertEqual(report.note, "no trunk model configured; curation deferred")
        XCTAssertTrue(ledger.entries().isEmpty)
    }

    func testSweepRetriesPendingReplayAfterDriverFailure() async throws {
        let recorder = ReplayRecorder()
        recorder.error = TesseraSessionCurationError.replayUnavailable("llama-imatrix not found")
        let (stage, store, ledger) = try makeStage(
            decoder: FakeSessionDecoder(nVocab: 1000, pieces: cleanSessionPieces()),
            recorder: recorder)
        try store.appendRuntime(records: cleanSessionSteps(sid: "P"))

        // First sweep: the verdict lands, replay degrades open and stays
        // pending - the sweep is resumable, not lossy.
        let first = await stage.sweep()
        XCTAssertEqual(first.promoted, 1)
        XCTAssertEqual(first.replayedSessions, 0)
        XCTAssertEqual(first.pendingReplay, ["P"])
        XCTAssertTrue(first.note?.contains("replay deferred") == true)
        XCTAssertEqual(store.replayFiles().count, 0)

        // Second sweep: nothing new to analyze; the pending replay retries
        // and lands.
        recorder.error = nil
        let second = await stage.sweep()
        XCTAssertEqual(second.analyzed, 0)
        XCTAssertEqual(second.replayedSessions, 1)
        XCTAssertEqual(second.replayRecords, 1)
        XCTAssertTrue(second.pendingReplay.isEmpty)
        XCTAssertEqual(store.replayFiles().count, 1)
        XCTAssertEqual(ledger.verdict(for: "P"), .promoted)

        // Third sweep: fully drained.
        let third = await stage.sweep()
        XCTAssertEqual(third.note, "no uncurated sessions")
        XCTAssertEqual(recorder.corpora.count, 2)
    }

    func testStampReplayLineAppendsProvenance() {
        let stamped = TesseraSessionCurationStage.stampReplayLine(
            "{\"schema\":\"llama.tessera.spec.v1\",\"step_idx\":0}")
        XCTAssertEqual(
            stamped,
            "{\"schema\":\"llama.tessera.spec.v1\",\"step_idx\":0,"
                + "\"provenance\":\"replay\",\"replayed_from\":\"runtime\"}")
        XCTAssertNil(TesseraSessionCurationStage.stampReplayLine("not json"))
        XCTAssertNil(TesseraSessionCurationStage.stampReplayLine(""))
        XCTAssertNil(TesseraSessionCurationStage.stampReplayLine("[1,2,3]"))
    }

    func testReplayContextSizeClampsToCalibrationBounds() {
        XCTAssertEqual(TesseraSessionCurationStage.replayContextSize(for: ""), 32)
        XCTAssertEqual(TesseraSessionCurationStage.replayContextSize(for: String(repeating: "x", count: 1000)), 200)
        XCTAssertEqual(TesseraSessionCurationStage.replayContextSize(for: String(repeating: "x", count: 100_000)), 4096)
    }
}

// MARK: - Scheduler gates

final class TesseraSessionCurationSchedulerTests: XCTestCase {
    private var roots: [URL] = []

    override func tearDown() {
        for root in roots { try? FileManager.default.removeItem(at: root) }
        roots.removeAll()
        super.tearDown()
    }

    private func withSettings<T>(_ overrides: [String: Any], _ body: () async -> T) async -> T {
        let saved: [(String, Any?)] = overrides.keys.map { ($0, UserDefaults.standard.object(forKey: $0)) }
        for (key, value) in overrides { UserDefaults.standard.set(value, forKey: key) }
        let result = await body()
        for (key, value) in saved {
            if let value { UserDefaults.standard.set(value, forKey: key) } else { UserDefaults.standard.removeObject(forKey: key) }
        }
        return result
    }

    private func makeIdleStage() throws -> TesseraSessionCurationStage {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("tessera-curation-sched-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        roots.append(root)
        return TesseraSessionCurationStage(
            store: TesseraTraceStore(directory: root.appendingPathComponent("traces")),
            ledger: TesseraCurationLedger(directory: root),
            decoderProvider: { _ in nil },
            replayDriver: { _, _ in [] },
            trunkPathProvider: { "" })
    }

    func testSweepSkipsWhenCurationDisabled() async throws {
        let scheduler = TesseraSessionCurationScheduler(stage: try makeIdleStage())
        let report = await withSettings([TesseraSettingsKey.learningSessionCuration: false]) {
            await scheduler.sweep()
        }
        XCTAssertNil(report)
    }

    func testSweepRunsWhenCurationEnabled() async throws {
        let scheduler = TesseraSessionCurationScheduler(stage: try makeIdleStage())
        let report = await withSettings([
            TesseraSettingsKey.learningSessionCuration: true,
            TesseraSettingsKey.learningOnPowerOnly: false,
        ]) {
            await scheduler.sweep()
        }
        XCTAssertNotNil(report)
        XCTAssertEqual(report?.sessionsSeen, 0)
    }
}

// MARK: - Training consumption excludes runtime records (spec 12.5)

final class TesseraTrainingRuntimeExclusionTests: XCTestCase {
    private var roots: [URL] = []

    override func tearDown() {
        for root in roots { try? FileManager.default.removeItem(at: root) }
        roots.removeAll()
        unsetenv("TESSERA_TEST_CAPTURE")
        super.tearDown()
    }

    private func makeTempRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("tessera-exclusion-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        roots.append(root)
        return root
    }

    func testTrainingStagingExcludesRuntimeRecords() async throws {
        let root = try makeTempRoot()
        let tracesDir = root.appendingPathComponent("traces")
        try FileManager.default.createDirectory(at: tracesDir, withIntermediateDirectories: true)
        let store = TesseraTraceStore(directory: tracesDir)

        // One calibration, one replay, two runtime records: the training gate
        // counts all four, but the staged dataset keeps calibration + replay.
        let calibrationSource = root.appendingPathComponent("calib.jsonl")
        try ("{\"schema\":\"llama.tessera.spec.v1\",\"step_idx\":0,\"drafted\":3,\"accepted\":2}\n")
            .write(to: calibrationSource, atomically: true, encoding: .utf8)
        try store.appendRun(jsonlPath: calibrationSource)
        try store.appendReplay(records: [
            "{\"schema\":\"llama.tessera.spec.v1\",\"step_idx\":0,\"drafted\":3,\"accepted\":2,\"provenance\":\"replay\",\"replayed_from\":\"runtime\"}",
        ])
        try store.appendRuntime(records: [
            "{\"schema\":\"llama.tessera.spec.v1\",\"step_idx\":0,\"drafted\":3,\"accepted\":2,\"provenance\":\"runtime\",\"sid\":\"r1\"}",
            "{\"schema\":\"llama.tessera.spec.v1\",\"step_idx\":1,\"drafted\":3,\"accepted\":2,\"provenance\":\"runtime\",\"sid\":\"r1\"}",
        ])
        XCTAssertEqual(store.totalRecords(), 4)

        // The fake driver copies its --traces input where the test can read
        // it, then prints the progress lines the orchestrator parses.
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
        XCTAssertTrue(captured.contains("\"step_idx\":0,\"drafted\":3,\"accepted\":2}"))
        XCTAssertTrue(captured.contains("\"provenance\":\"replay\""))
        XCTAssertFalse(captured.contains("\"provenance\":\"runtime\""),
                       "raw runtime captures must never reach training; they enter through the curation stage's replay")
        XCTAssertEqual(captured.split(separator: "\n").count, 2)
    }
}
