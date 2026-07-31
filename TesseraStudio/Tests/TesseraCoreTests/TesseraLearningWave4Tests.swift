import XCTest
@testable import TesseraCore

// Wave 4 (Swift) self-improving-loop tests. Covers the pure, offline logic:
// the capability-eval harness exchange (axis mapping, tally reduction, the
// instances-JSON both harness subcommands consume) and the curation analytics
// (content-hash dedup, quality, preference-pair formation, informativeness).
// Nothing here shells out to the binary or touches the learning store.

final class CapabilityEvalHarnessTests: XCTestCase {
    private let service = TesseraCapabilityEvalService()

    func testAxisMappingCoversAllAxesAndIsReversible() {
        for axis in TesseraCapabilityScore.axisNames {
            XCTAssertNotNil(TesseraCapabilityEvalService.axisToHarnessKey[axis], "missing harness key for \(axis)")
        }
        // The harness keys are the snake_case ts_capability_score field names.
        XCTAssertEqual(TesseraCapabilityEvalService.axisToHarnessKey["apiCurrency"], "api_currency")
        XCTAssertEqual(TesseraCapabilityEvalService.axisToHarnessKey["generalCompetence"], "general_competence")
        // Five distinct axes -> five distinct harness keys.
        XCTAssertEqual(Set(TesseraCapabilityEvalService.axisToHarnessKey.values).count, 5)
    }

    func testTallyReducesPassFailAndIgnoresUnknownAxis() {
        let results = [
            TesseraEvalInstanceResult(instanceId: "1", axis: "mechanical", passed: true),
            TesseraEvalInstanceResult(instanceId: "2", axis: "mechanical", passed: true),
            TesseraEvalInstanceResult(instanceId: "3", axis: "mechanical", passed: false),
            TesseraEvalInstanceResult(instanceId: "4", axis: "hardTail", passed: false),
            TesseraEvalInstanceResult(instanceId: "5", axis: "notAnAxis", passed: true),
        ]
        let tallies = service.tally(from: results)

        // Every axis is present (zeroed), so the harness always gets all five.
        for axis in TesseraCapabilityScore.axisNames {
            XCTAssertNotNil(tallies[axis])
        }
        XCTAssertEqual(tallies["mechanical"]?.pass, 2)
        XCTAssertEqual(tallies["mechanical"]?.fail, 1)
        XCTAssertEqual(tallies["mechanical"]?.fraction ?? -1, 2.0 / 3.0, accuracy: 1e-9)
        XCTAssertEqual(tallies["hardTail"]?.fail, 1)
        XCTAssertEqual(tallies["apiCurrency"]?.total, 0)
        XCTAssertEqual(tallies["apiCurrency"]?.fraction, 0)
    }

    func testScoreMatchesTallyFractions() {
        let results = [
            TesseraEvalInstanceResult(instanceId: "1", axis: "mechanical", passed: true),
            TesseraEvalInstanceResult(instanceId: "2", axis: "mechanical", passed: false),
            TesseraEvalInstanceResult(instanceId: "3", axis: "generalCompetence", passed: true),
        ]
        let score = service.score(from: results)
        XCTAssertEqual(score.mechanical, 0.5, accuracy: 1e-9)
        XCTAssertEqual(score.generalCompetence, 1.0, accuracy: 1e-9)
        XCTAssertEqual(score.apiCurrency, 0.0, accuracy: 1e-9)
    }

    func testSerializeInstancesEmitsAllFiveAxesAndSchema() throws {
        let tallies = service.tally(from: [
            TesseraEvalInstanceResult(instanceId: "1", axis: "mechanical", passed: true),
        ])
        let data = try service.serializeInstancesJSON(tallies: tallies)
        let obj = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])

        XCTAssertEqual(obj["schema_version"] as? Int, 1)
        XCTAssertNil(obj["baseline"], "no baseline requested -> key omitted")

        let axes = try XCTUnwrap(obj["axes"] as? [String: Any])
        for key in ["mechanical", "api_currency", "hard_tail", "personal_style", "general_competence"] {
            let axis = try XCTUnwrap(axes[key] as? [String: Any], "missing axis \(key)")
            XCTAssertNotNil(axis["pass"])
            XCTAssertNotNil(axis["fail"])
        }
        let mechanical = try XCTUnwrap(axes["mechanical"] as? [String: Any])
        XCTAssertEqual(mechanical["pass"] as? Int, 1)
        XCTAssertEqual(mechanical["fail"] as? Int, 0)
    }

    func testSerializeInstancesIncludesBaselineWhenGiven() throws {
        let baseline = TesseraCapabilityScore(
            mechanical: 0.7, apiCurrency: 0.5, hardTail: 0.5, personalStyle: 0.6, generalCompetence: 0.85
        )
        let data = try service.serializeInstancesJSON(tallies: [:], baseline: baseline)
        let obj = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])

        let b = try XCTUnwrap(obj["baseline"] as? [String: Any])
        // The harness requires all five baseline keys, snake_case.
        XCTAssertEqual(b["general_competence"] as? Double, 0.85)
        XCTAssertEqual(b["api_currency"] as? Double, 0.5)
        XCTAssertEqual(b.count, 5)
    }
}

final class CapabilityScoreLensTests: XCTestCase {
    func testWeightedSumExcludesGuardAxis() {
        let score = TesseraCapabilityScore(
            mechanical: 0.8, apiCurrency: 0.6, hardTail: 0.5, personalStyle: 0.7, generalCompetence: 0.0
        )
        // Uniform weights over the four optimization axes -> their mean; the
        // zeroed guard axis must not pull the sum down.
        let sum = score.weightedSum(weights: TesseraCapabilityEvalService.uniformWeights)
        XCTAssertEqual(sum, (0.8 + 0.6 + 0.5 + 0.7) / 4.0, accuracy: 1e-9)
    }

    func testGuardComparesGeneralCompetenceOnly() {
        let baseline = TesseraCapabilityScore(generalCompetence: 0.9)
        let regressed = TesseraCapabilityScore(mechanical: 1.0, generalCompetence: 0.5)
        let held = TesseraCapabilityScore(mechanical: 0.1, generalCompetence: 0.89)
        XCTAssertFalse(regressed.passesGuard(baseline: baseline, epsilon: 0.02))
        XCTAssertTrue(held.passesGuard(baseline: baseline, epsilon: 0.02))
        // No baseline -> nothing to regress against -> passes trivially.
        XCTAssertTrue(regressed.passesGuard(baseline: nil, epsilon: 0.02))
    }
}

final class CurationAnalyticsTests: XCTestCase {
    private let curation = TesseraCurationService()

    private func outcome(_ kind: TesseraWorldOutcomeKind, _ success: Bool, _ detail: String = "") -> TesseraWorldOutcome {
        TesseraWorldOutcome(kind: kind, success: success, detail: detail)
    }

    func testContentHashIsStableAndContentSensitive() {
        let a = outcome(.test, true, "fixed the off-by-one")
        let b = outcome(.test, true, "fixed the off-by-one")
        let c = outcome(.test, false, "fixed the off-by-one")
        let d = outcome(.test, true, "a different change")
        XCTAssertEqual(curation.contentHash(a), curation.contentHash(b), "same content -> same hash")
        XCTAssertNotEqual(curation.contentHash(a), curation.contentHash(c), "success differs -> hash differs")
        XCTAssertNotEqual(curation.contentHash(a), curation.contentHash(d), "detail differs -> hash differs")
    }

    func testContentHashCollapsesSecretVariants() {
        // Two outcomes that differ only in a secret value scrub to the same
        // material, so they dedup together.
        let a = outcome(.build, true, "token=sk-abcdefgh123")
        let b = outcome(.build, true, "token=sk-zyxwvuts987")
        XCTAssertEqual(curation.contentHash(a), curation.contentHash(b))
    }

    func testQualityScoreIsBoundedAndRewardsSuccess() {
        let pass = outcome(.commit, true, "landed the fix after reviewing the failing test and the type error")
        let fail = outcome(.build, false, "")
        let qPass = curation.qualityScore(pass)
        let qFail = curation.qualityScore(fail)
        XCTAssertGreaterThanOrEqual(qPass, 0.0)
        XCTAssertLessThanOrEqual(qPass, 1.0)
        XCTAssertGreaterThanOrEqual(qFail, 0.0)
        XCTAssertLessThanOrEqual(qFail, 1.0)
        XCTAssertGreaterThan(qPass, qFail)
    }

    func testPreferencePairsFormPerClassFromPassVsFail() {
        let outcomes = [
            outcome(.test, true, "p1"), outcome(.test, true, "p2"),
            outcome(.test, false, "f1"),
            outcome(.build, true, "only-a-pass"),   // no fail -> no pair
            outcome(.commit, false, "f2"), outcome(.commit, false, "f3"),  // no pass -> no pair
        ]
        let pairs = curation.preferencePairs(from: outcomes)

        // Only the "test" class has both a pass and a fail -> exactly one pair
        // (min(2 passes, 1 fail)).
        XCTAssertEqual(pairs.count, 1)
        let pair = try? XCTUnwrap(pairs.first)
        XCTAssertEqual(pair?.problemClass, "test")
        XCTAssertEqual(pair?.chosen.success, true)
        XCTAssertEqual(pair?.rejected.success, false)
    }

    func testPreferencePairsEmptyWhenNoClasses() {
        XCTAssertTrue(curation.preferencePairs(from: []).isEmpty)
        // All passes, no fails -> nothing to pair.
        let onlyPasses = [outcome(.test, true), outcome(.build, true)]
        XCTAssertTrue(curation.preferencePairs(from: onlyPasses).isEmpty)
    }

    func testInformativenessIsBoundedAndFavorsNovelty() {
        let stored = [outcome(.test, true, "seen content"), outcome(.test, true, "seen content 2")]
        let novel = outcome(.commit, false, "brand new failure")
        let repeated = outcome(.test, true, "seen content")

        let iNovel = curation.informativeness(of: novel, against: stored)
        let iRepeat = curation.informativeness(of: repeated, against: stored)

        for v in [iNovel, iRepeat] {
            XCTAssertGreaterThanOrEqual(v, 0.0)
            XCTAssertLessThanOrEqual(v, 1.0)
        }
        XCTAssertGreaterThan(iNovel, iRepeat, "unseen content outranks a repeat")
        // Against an empty store everything is novel and rare -> maximal-ish.
        XCTAssertGreaterThan(curation.informativeness(of: novel, against: []), 0.5)
    }
}
