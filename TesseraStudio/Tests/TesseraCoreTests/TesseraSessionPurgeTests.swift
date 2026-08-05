import XCTest
@testable import TesseraCore

// Runtime-traces spec sections 9, 10, 12.4 tests: user-initiated session
// purge (the ONLY path that removes quarantined sessions), the curation
// state counts behind the dashboard capture row, the quarantine list
// display infos, and the probe-class mapping shared with the scrub wall.

// MARK: - Fixtures

private func runtimeRecord(sid: String, step: Int, drafted: Int = 2, accepted: Int = 1) -> String {
    "{\"schema\":\"llama.tessera.spec.v1\",\"step_idx\":\(step),"
        + "\"drafted\":\(drafted),\"accepted\":\(accepted),"
        + "\"accepted_tokens\":[1,2],\"provenance\":\"runtime\",\"sid\":\"\(sid)\"}"
}

private func makeTempDir(_ label: String) throws -> URL {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("tessera-\(label)-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

private func readLines(_ file: URL) throws -> [String] {
    let text = try String(contentsOf: file, encoding: .utf8)
    return text.split(separator: "\n").map(String.init).filter { !$0.isEmpty }
}

// MARK: - Trace store session purge

final class TesseraTraceStorePurgeTests: XCTestCase {
    private var dirs: [URL] = []

    override func tearDown() {
        for dir in dirs { try? FileManager.default.removeItem(at: dir) }
        dirs.removeAll()
        super.tearDown()
    }

    private func makeStore() throws -> TesseraTraceStore {
        let dir = try makeTempDir("purge-store")
        dirs.append(dir)
        return TesseraTraceStore(directory: dir)
    }

    func testPurgeSessionRemovesOnlyMatchingLines() throws {
        let store = try makeStore()
        try store.appendRuntime(records: [
            runtimeRecord(sid: "A", step: 0),
            runtimeRecord(sid: "B", step: 0),
            runtimeRecord(sid: "A", step: 1),
            runtimeRecord(sid: "B", step: 1),
        ])
        let file = try XCTUnwrap(store.runtimeFiles().first)

        let removed = try store.purgeSession(sid: "A")

        XCTAssertEqual(removed, 2)
        let lines = try readLines(file)
        XCTAssertEqual(lines.count, 2)
        for line in lines {
            XCTAssertTrue(line.contains("\"sid\":\"B\""), "session A record survived purge")
        }
        // The runtime summary no longer knows session A.
        let summary = store.runtimeSummary()
        XCTAssertEqual(summary.sessions.map { $0.sid }, ["B"])
    }

    func testPurgeSessionDeletesEmptiedFile() throws {
        let store = try makeStore()
        try store.appendRuntime(records: [
            runtimeRecord(sid: "A", step: 0),
            runtimeRecord(sid: "A", step: 1),
        ])
        XCTAssertEqual(store.runtimeFiles().count, 1)

        let removed = try store.purgeSession(sid: "A")

        XCTAssertEqual(removed, 2)
        XCTAssertTrue(store.runtimeFiles().isEmpty)
        XCTAssertEqual(store.totalRecords(), 0)
    }

    func testPurgeSessionAcrossSplitFiles() throws {
        let store = try makeStore()
        // Two flushes of the same session land in two files (same-day
        // collision suffix); purge must reach both.
        try store.appendRuntime(records: [runtimeRecord(sid: "A", step: 0)])
        try store.appendRuntime(records: [runtimeRecord(sid: "A", step: 1)])
        XCTAssertEqual(store.runtimeFiles().count, 2)

        let removed = try store.purgeSession(sid: "A")

        XCTAssertEqual(removed, 2)
        XCTAssertTrue(store.runtimeFiles().isEmpty)
    }

    func testPurgeSessionSparesOtherProvenances() throws {
        let store = try makeStore()
        try store.appendRuntime(records: [runtimeRecord(sid: "A", step: 0)])
        let replay = try XCTUnwrap(store.appendReplay(records: [
            "{\"schema\":\"llama.tessera.spec.v1\",\"step_idx\":0,\"provenance\":\"replay\",\"replayed_from\":\"runtime\"}",
        ]))
        let calibrationSource = try makeTempDir("purge-calib-src")
            .appendingPathComponent("calib.jsonl")
        try "{\"schema\":\"llama.tessera.spec.v1\",\"step_idx\":0}\n"
            .write(to: calibrationSource, atomically: true, encoding: .utf8)
        let calibration = try store.appendRun(jsonlPath: calibrationSource)

        _ = try store.purgeSession(sid: "A")

        XCTAssertEqual(try readLines(replay).count, 1, "replay file touched by session purge")
        XCTAssertEqual(try readLines(calibration).count, 1, "calibration file touched by session purge")
        XCTAssertTrue(store.runtimeFiles().isEmpty)
    }

    func testPurgeSessionEmptySidIsNoop() throws {
        let store = try makeStore()
        try store.appendRuntime(records: [runtimeRecord(sid: "A", step: 0)])
        XCTAssertEqual(try store.purgeSession(sid: ""), 0)
        XCTAssertEqual(store.runtimeSummary().totalRecords, 1)
    }

    func testPurgeSessionUnknownSidIsNoop() throws {
        let store = try makeStore()
        try store.appendRuntime(records: [runtimeRecord(sid: "A", step: 0)])
        XCTAssertEqual(try store.purgeSession(sid: "nope"), 0)
        XCTAssertEqual(store.runtimeSummary().totalRecords, 1)
    }
}

// MARK: - Ledger purge verdict, counts, quarantine infos

final class TesseraCurationLedgerPurgeTests: XCTestCase {
    private var dirs: [URL] = []

    override func tearDown() {
        for dir in dirs { try? FileManager.default.removeItem(at: dir) }
        dirs.removeAll()
        super.tearDown()
    }

    private func makeLedger() throws -> TesseraCurationLedger {
        let dir = try makeTempDir("purge-ledger")
        dirs.append(dir)
        return TesseraCurationLedger(directory: dir)
    }

    private func judgement(
        sid: String, verdict: TesseraSessionVerdict,
        reasons: [String], tokens: Int = 100
    ) -> TesseraCurationLedgerEntry {
        TesseraCurationLedgerEntry(
            sid: sid, verdict: verdict, reasons: reasons,
            score: .init(acceptance: 0.5, tokens: tokens, repetition: 0.1))
    }

    func testMarkPurgedRemovesSessionFromQuarantine() throws {
        let ledger = try makeLedger()
        try ledger.append(judgement(sid: "Q1", verdict: .quarantined, reasons: ["probe:email"]))
        XCTAssertEqual(ledger.quarantinedSids(), ["Q1"])

        try ledger.markPurged(sid: "Q1")

        // Latest wins: no longer quarantined, never re-analyzed, gone from
        // the quarantine list.
        XCTAssertTrue(ledger.quarantinedSids().isEmpty)
        XCTAssertTrue(ledger.quarantinedSessions().isEmpty)
        XCTAssertEqual(ledger.verdict(for: "Q1"), .purged)
        XCTAssertTrue(ledger.quarantinedSessionInfos().isEmpty)
    }

    func testForStorePointsAtLearningRoot() throws {
        let root = try makeTempDir("purge-root")
        dirs.append(root)
        let store = TesseraTraceStore(directory: root.appendingPathComponent("traces"))
        let ledger = TesseraCurationLedger.forStore(store)
        XCTAssertEqual(ledger.url.deletingLastPathComponent().standardizedFileURL.path,
                       root.standardizedFileURL.path)
        XCTAssertEqual(ledger.url.lastPathComponent, TesseraCurationLedger.fileName)
    }

    func testCurationCountsOverStoreSessions() throws {
        let ledger = try makeLedger()
        try ledger.append(judgement(sid: "P", verdict: .promoted, reasons: ["probe:none"]))
        try ledger.append(judgement(sid: "Q", verdict: .quarantined, reasons: ["probe:email"]))
        try ledger.append(judgement(sid: "D", verdict: .dropped, reasons: ["duplicate"]))
        try ledger.append(judgement(sid: "X", verdict: .purged, reasons: ["user-purge"]))

        // "X" and "gone" are not in the store anymore: judged sessions whose
        // records were trimmed (or purged) do not appear in the counts.
        let counts = ledger.curationCounts(sessionSids: ["P", "Q", "D", "fresh"])

        XCTAssertEqual(counts, TesseraCurationCounts(promoted: 1, quarantined: 1, pending: 1))
    }

    func testQuarantinedSessionInfosExposeOnlySafeFields() throws {
        let ledger = try makeLedger()
        let ts = TesseraCurationLedger.timestamp(Date())
        try ledger.append(TesseraCurationLedgerEntry(
            sid: "Q9", verdict: .quarantined,
            reasons: ["probe:pem-key", "probe:email"],
            score: .init(acceptance: 0.4, tokens: 123, repetition: 0.0),
            ts: ts))

        let infos = ledger.quarantinedSessionInfos()
        XCTAssertEqual(infos.count, 1)
        let info = try XCTUnwrap(infos.first)
        XCTAssertEqual(info.sid, "Q9")
        XCTAssertEqual(info.tokenCount, 123)
        XCTAssertEqual(info.probeClasses, [TesseraProbeClass.secrets, TesseraProbeClass.contactInfo])
        XCTAssertEqual(info.date, TesseraCurationLedger.date(fromTimestamp: ts))
    }

    func testTimestampRoundTrip() {
        let date = Date(timeIntervalSince1970: 1_770_000_000)
        let ts = TesseraCurationLedger.timestamp(date)
        XCTAssertEqual(ts, "2026-02-02T02:40:00Z")
        XCTAssertEqual(TesseraCurationLedger.date(fromTimestamp: ts), date)
        XCTAssertNil(TesseraCurationLedger.date(fromTimestamp: "not-a-timestamp"))
    }
}

// MARK: - Probe class mapping (shared vocabulary with the scrub wall)

final class TesseraProbeClassTests: XCTestCase {
    func testEveryScrubRuleMapsToKnownClass() {
        // Drift guard: the quarantine surface names classes for every rule
        // the scrub wall ships. A new rule that lands without a class
        // mapping fails here, not in production.
        let known: Set<String> = [
            TesseraProbeClass.secrets,
            TesseraProbeClass.contactInfo,
            TesseraProbeClass.paths,
            TesseraProbeClass.modelMismatch,
        ]
        for rule in TesseraScrubRules.all {
            XCTAssertTrue(
                known.contains(TesseraProbeClass.label(forRule: rule.id)),
                "rule \(rule.id) has no probe-class mapping")
        }
    }

    func testClassesFromLedgerReasonsDeduplicateInOrder() {
        let classes = TesseraProbeClass.classes(forLedgerReasons: [
            "probe:email", "probe:pem-key", "probe:phone", "low-acceptance",
        ])
        XCTAssertEqual(classes, [TesseraProbeClass.contactInfo, TesseraProbeClass.secrets])
    }

    func testProbeNoneAndUnknownReasonsAreIgnored() {
        XCTAssertEqual(TesseraProbeClass.classes(forLedgerReasons: ["probe:none"]), [])
        XCTAssertEqual(
            TesseraProbeClass.classes(forLedgerReasons: ["below-token-floor", "duplicate"]), [])
    }

    func testModelMismatchHasItsOwnClass() {
        XCTAssertEqual(
            TesseraProbeClass.classes(forLedgerReasons: ["model-mismatch"]),
            [TesseraProbeClass.modelMismatch])
    }

    func testUnknownRuleIdSurfacesItsOwnId() {
        // A future rule the mapping has not caught up with stays honest:
        // the id shows, never the matched content.
        XCTAssertEqual(TesseraProbeClass.label(forRule: "future-rule"), "future-rule")
        XCTAssertEqual(
            TesseraProbeClass.classes(forLedgerReasons: ["probe:future-rule"]), ["future-rule"])
    }
}

// MARK: - Stage + scheduler integration with purge

final class TesseraSessionPurgeIntegrationTests: XCTestCase {
    private var dirs: [URL] = []

    override func tearDown() {
        for dir in dirs { try? FileManager.default.removeItem(at: dir) }
        dirs.removeAll()
        super.tearDown()
    }

    private func makeTempDir(_ label: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tessera-\(label)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        dirs.append(dir)
        return dir
    }

    func testStageSkipsPurgedSessions() async throws {
        let root = try makeTempDir("purge-stage")
        let store = TesseraTraceStore(directory: root.appendingPathComponent("traces"))
        let ledger = TesseraCurationLedger.forStore(store)
        try store.appendRuntime(records: [runtimeRecord(sid: "P1", step: 0)])
        try ledger.markPurged(sid: "P1")

        let stage = TesseraSessionCurationStage(store: store, ledger: ledger)
        let report = await stage.sweep()

        // Purged is a terminal verdict: the session is neither re-analyzed
        // nor counted as pending.
        XCTAssertEqual(report.sessionsSeen, 1)
        XCTAssertEqual(report.analyzed, 0)
        XCTAssertTrue(report.pendingReplay.isEmpty)
        XCTAssertEqual(ledger.verdict(for: "P1"), .purged)
    }

    func testQuarantineExemptionReadsLedgerAtTrimTime() throws {
        // The provider and the stage pass ledger.quarantinedSids() into the
        // store trimmers at each flush/sweep, so a session quarantined AFTER
        // capture becomes exempt from the rolling cap on the next trim
        // without re-flushing.
        let root = try makeTempDir("purge-exempt")
        let store = TesseraTraceStore(directory: root.appendingPathComponent("traces"))
        let ledger = TesseraCurationLedger.forStore(store)
        try store.appendRuntime(records: [runtimeRecord(sid: "Q", step: 0)])
        try ledger.append(TesseraCurationLedgerEntry(
            sid: "Q", verdict: .quarantined, reasons: ["probe:email"],
            score: .init(acceptance: 0.5, tokens: 100, repetition: 0.0)))

        let exempt = TesseraCurationLedger.forStore(store).quarantinedSids()
        XCTAssertEqual(exempt, ["Q"])

        // A budget of 1 byte would trim everything not exempt; the
        // quarantined session survives.
        try store.trimRuntimeToBudget(budgetBytes: 1, exemptSids: exempt)
        XCTAssertEqual(store.runtimeSummary().totalRecords, 1)
        XCTAssertEqual(store.runtimeSummary().sessions.map(\.sid), ["Q"])
    }
}

// MARK: - Replay stamp idempotence

final class TesseraReplayStampTests: XCTestCase {
    func testStampIsIdempotent() {
        let once = TesseraSessionCurationStage.stampReplayLine(
            "{\"schema\":\"llama.tessera.spec.v1\",\"step_idx\":0}")
        let twice = once.flatMap { TesseraSessionCurationStage.stampReplayLine($0) }
        XCTAssertEqual(once, twice)
        XCTAssertEqual(twice?.contains("\"replayed_from\":\"runtime\""), true)
    }

    func testStampRejectsForeignProvenance() {
        let stamped = TesseraSessionCurationStage.stampReplayLine(
            "{\"schema\":\"llama.tessera.spec.v1\",\"provenance\":\"calibration\"}")
        XCTAssertNil(stamped, "foreign provenance must not be re-stamped as replay")
    }

    func testStampRejectsMalformedJson() {
        XCTAssertNil(TesseraSessionCurationStage.stampReplayLine("not json"))
    }
}
