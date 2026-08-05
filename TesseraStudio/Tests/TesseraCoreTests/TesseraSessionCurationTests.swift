import XCTest
@testable import TesseraCore

// S4 (runtime-traces spec): the session curation stage's pure core -
// the shared versioned scrub rule set (section 12.3 sensitivity probe),
// the scorecard thresholds, and the append-only verdict ledger (12.4).

// MARK: - Scrub rules: one set for wall and probe

final class TesseraScrubRulesTests: XCTestCase {
    private let samples: [(id: String, text: String)] = [
        ("pem-key", "-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----"),
        ("bearer-token", "Authorization: Bearer abc123.def-456_ghi"),
        ("secret-key", "use sk-abcdefgh12345678 for the call"),
        ("credential-assignment", "export OPENAI_API_KEY=secret-value-here"),
        ("email", "write to jane.doe@example.com for details"),
        ("phone", "call me at (555) 123-4567 tomorrow"),
        ("fs-path", "the model lives at /Users/jane/models/base.gguf"),
    ]

    func testProbeDetectsEveryRuleClass() {
        for sample in samples {
            let hits = TesseraScrubRules.probe(sample.text)
            XCTAssertTrue(hits.contains(sample.id),
                "expected \(sample.id) hit for: \(sample.text), got \(hits)")
        }
    }

    func testProbeAndScrubShareTheSameRuleSet() {
        // The probe is the read-only twin of the wall: whenever the probe
        // fires, the scrub must change the text (and vice versa).
        for sample in samples {
            XCTAssertFalse(TesseraScrubRules.probe(sample.text).isEmpty)
            XCTAssertNotEqual(TesseraScrubRules.scrub(sample.text), sample.text)
        }
        let clean = "a perfectly ordinary sentence about weather and tea"
        XCTAssertTrue(TesseraScrubRules.probe(clean).isEmpty)
        XCTAssertEqual(TesseraScrubRules.scrub(clean), clean)
    }

    func testProbeNeverReportsMatchedContent() {
        // The quarantine list shows rule ids, never the secret itself.
        let hits = TesseraScrubRules.probe("key sk-abcdefgh12345678 here")
        XCTAssertEqual(hits, ["secret-key"])
    }

    func testVersionStampFormat() {
        XCTAssertEqual(TesseraScrubRules.requiredVersionStamp, ">=\(TesseraScrubRules.version)")
    }
}

// MARK: - Scorecard thresholds (section 12.3)

final class TesseraSessionScorecardTests: XCTestCase {
    private func makeAnalysis(
        tokenCount: Int = 500,
        acceptanceRate: Double = 0.6,
        meanAcceptedRun: Double = 1.8,
        repetitionRatio: Double = 0.05,
        garbageRatio: Double = 0.01,
        probeHits: [String] = [],
        fingerprint: String = "abc123",
        modelCompatible: Bool = true
    ) -> TesseraSessionAnalysis {
        TesseraSessionAnalysis(
            tokenCount: tokenCount,
            acceptanceRate: acceptanceRate,
            meanAcceptedRun: meanAcceptedRun,
            repetitionRatio: repetitionRatio,
            garbageRatio: garbageRatio,
            probeHits: probeHits,
            fingerprint: fingerprint,
            modelCompatible: modelCompatible)
    }

    func testCleanSessionPromotes() {
        let judgement = TesseraSessionScorecard.judge(makeAnalysis(), isDuplicate: false)
        XCTAssertEqual(judgement.verdict, .promoted)
        XCTAssertTrue(judgement.reasons.contains("probe:none"))
        XCTAssertTrue(judgement.reasons.contains("dedup:kept"))
    }

    func testProbeHitQuarantinesRegardlessOfQuality() {
        let judgement = TesseraSessionScorecard.judge(
            makeAnalysis(probeHits: ["pem-key", "email"]), isDuplicate: false)
        XCTAssertEqual(judgement.verdict, .quarantined)
        XCTAssertEqual(judgement.reasons, ["probe:pem-key", "probe:email"])
    }

    func testDuplicateCollapses() {
        let judgement = TesseraSessionScorecard.judge(makeAnalysis(), isDuplicate: true)
        XCTAssertEqual(judgement.verdict, .dropped)
        XCTAssertEqual(judgement.reasons, ["duplicate"])
    }

    func testModelMismatchDropsAndOutranksProbe() {
        // Token ids no longer decode: nothing else about the session is
        // trustworthy, not even the sensitivity probe.
        let judgement = TesseraSessionScorecard.judge(
            makeAnalysis(probeHits: ["email"], modelCompatible: false), isDuplicate: false)
        XCTAssertEqual(judgement.verdict, .dropped)
        XCTAssertEqual(judgement.reasons, ["model-mismatch"])
    }

    func testQualityFloorsDrop() {
        XCTAssertEqual(TesseraSessionScorecard.judge(
            makeAnalysis(tokenCount: 10), isDuplicate: false).reasons, ["below-token-floor"])
        XCTAssertEqual(TesseraSessionScorecard.judge(
            makeAnalysis(acceptanceRate: 0.02), isDuplicate: false).reasons, ["low-acceptance"])
        XCTAssertEqual(TesseraSessionScorecard.judge(
            makeAnalysis(repetitionRatio: 0.9), isDuplicate: false).reasons, ["high-repetition"])
        XCTAssertEqual(TesseraSessionScorecard.judge(
            makeAnalysis(garbageRatio: 0.5), isDuplicate: false).reasons, ["garbage"])
    }

    // MARK: Metric derivations

    func testAcceptanceRateAndMeanRun() {
        let steps = [
            TesseraSessionStepStats(drafted: 3, accepted: 3),
            TesseraSessionStepStats(drafted: 3, accepted: 1),
            TesseraSessionStepStats(drafted: 2, accepted: 0),
        ]
        XCTAssertEqual(TesseraSessionScorecard.acceptanceRate(steps: steps), 4.0 / 8.0)
        XCTAssertEqual(TesseraSessionScorecard.meanAcceptedRun(steps: steps), 4.0 / 3.0)
        XCTAssertEqual(TesseraSessionScorecard.acceptanceRate(steps: []), 0)
    }

    func testRepetitionRatio() {
        let varied = "the quick brown fox jumps over a lazy dog near the river bank at dawn today"
        let looped = "repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat repeat"
        XCTAssertLessThan(TesseraSessionScorecard.repetitionRatio(of: varied), 0.2)
        XCTAssertGreaterThan(TesseraSessionScorecard.repetitionRatio(of: looped), 0.6)
        // Short texts have nothing to overlap.
        XCTAssertEqual(TesseraSessionScorecard.repetitionRatio(of: "one two three"), 0)
    }

    func testGarbageClassification() {
        XCTAssertTrue(TesseraSessionScorecard.isGarbage(""))        // EOS-ish
        XCTAssertTrue(TesseraSessionScorecard.isGarbage("\u{07}"))  // control junk
        XCTAssertFalse(TesseraSessionScorecard.isGarbage("\n"))     // normal text
        XCTAssertFalse(TesseraSessionScorecard.isGarbage(" "))      // normal text
        XCTAssertFalse(TesseraSessionScorecard.isGarbage("hello"))
        let pieces = ["hello", "", "\n", "\u{07}", "world"]
        XCTAssertEqual(TesseraSessionScorecard.garbageRatio(pieces: pieces), 2.0 / 5.0)
    }

    func testFingerprintIsNormalizedAndStable() {
        let a = TesseraSessionScorecard.fingerprint(of: "Hello   World\nNext")
        let b = TesseraSessionScorecard.fingerprint(of: "hello world next")
        let c = TesseraSessionScorecard.fingerprint(of: "a different session")
        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)
    }
}

// MARK: - Verdict ledger (section 12.4)

final class TesseraCurationLedgerTests: XCTestCase {
    private var dirs: [URL] = []

    override func tearDown() {
        for dir in dirs { try? FileManager.default.removeItem(at: dir) }
        dirs.removeAll()
        super.tearDown()
    }

    private func makeLedger() throws -> TesseraCurationLedger {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tessera-ledger-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        dirs.append(dir)
        return TesseraCurationLedger(directory: dir)
    }

    private func score() -> TesseraCurationLedgerEntry.Score {
        TesseraCurationLedgerEntry.Score(acceptance: 0.71, tokens: 1204, repetition: 0.06)
    }

    func testAppendRoundTripAndSchemaFields() throws {
        let ledger = try makeLedger()
        try ledger.append(TesseraCurationLedgerEntry(
            sid: "s1", verdict: .promoted, reasons: ["probe:none"], score: score()))

        let entries = ledger.entries()
        XCTAssertEqual(entries.count, 1)
        XCTAssertEqual(entries[0].schema, TesseraCurationLedger.schema)
        XCTAssertEqual(entries[0].sid, "s1")
        XCTAssertEqual(entries[0].verdictValue, .promoted)
        XCTAssertEqual(entries[0].anonymizerRequiredVersion, TesseraScrubRules.requiredVersionStamp)

        // The wire format keeps the spec's snake_case field.
        let raw = try String(contentsOf: ledger.url, encoding: .utf8)
        XCTAssertTrue(raw.contains("\"anonymizer_required_version\""))
        XCTAssertTrue(raw.hasSuffix("\n"))
    }

    func testAppendOnlyLatestWins() throws {
        let ledger = try makeLedger()
        // Quarantined under rule set v1, re-analyzed and promoted later.
        try ledger.append(TesseraCurationLedgerEntry(
            sid: "s1", verdict: .quarantined, reasons: ["probe:email"], score: score()))
        try ledger.append(TesseraCurationLedgerEntry(
            sid: "s1", verdict: .promoted, reasons: ["probe:none"], score: score()))
        try ledger.append(TesseraCurationLedgerEntry(
            sid: "s2", verdict: .dropped, reasons: ["duplicate"], score: score()))

        XCTAssertEqual(ledger.entries().count, 3)  // nothing was rewritten
        XCTAssertEqual(ledger.verdict(for: "s1"), .promoted)
        XCTAssertEqual(ledger.verdict(for: "s2"), .dropped)
        XCTAssertNil(ledger.verdict(for: "unknown"))
        XCTAssertTrue(ledger.promotedSids().contains("s1"))
        XCTAssertTrue(ledger.quarantinedSids().isEmpty)
    }

    func testTolerantDecodeSkipsJunkLines() throws {
        let ledger = try makeLedger()
        try ledger.append(TesseraCurationLedgerEntry(
            sid: "s1", verdict: .promoted, reasons: [], score: score()))
        // Hand-edit: junk line plus a foreign schema entry.
        var raw = try String(contentsOf: ledger.url, encoding: .utf8)
        raw += "not json at all\n"
        raw += "{\"schema\":\"some.other.v9\",\"sid\":\"x\",\"verdict\":\"promoted\",\"reasons\":[],\"score\":{\"acceptance\":0,\"tokens\":0,\"repetition\":0},\"anonymizer_required_version\":\">=1\",\"ts\":\"t\"}\n"
        try raw.write(to: ledger.url, atomically: true, encoding: .utf8)

        let entries = ledger.entries()
        XCTAssertEqual(entries.count, 1)
        XCTAssertEqual(entries[0].sid, "s1")
    }

    func testQuarantinedSessionsNewestFirst() throws {
        let ledger = try makeLedger()
        try ledger.append(TesseraCurationLedgerEntry(
            sid: "old", verdict: .quarantined, reasons: ["probe:phone"], score: score(),
            ts: "2026-08-01T00:00:00Z"))
        try ledger.append(TesseraCurationLedgerEntry(
            sid: "new", verdict: .quarantined, reasons: ["probe:pem-key"], score: score(),
            ts: "2026-08-04T00:00:00Z"))

        let sessions = ledger.quarantinedSessions()
        XCTAssertEqual(sessions.map { $0.sid }, ["new", "old"])
        // The list carries date, tokens, probe class - never matched content.
        XCTAssertEqual(sessions[0].reasons, ["probe:pem-key"])
        XCTAssertEqual(sessions[0].score.tokens, 1204)
    }

    func testTimestampIsISO8601UTC() {
        let ts = TesseraCurationLedger.timestamp(Date(timeIntervalSince1970: 0))
        XCTAssertEqual(ts, "1970-01-01T00:00:00Z")
    }
}
