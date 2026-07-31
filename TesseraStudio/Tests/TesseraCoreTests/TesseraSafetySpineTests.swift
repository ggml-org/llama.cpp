import XCTest
@testable import TesseraCore

// Safety/verification spine tests: layered permission (S4), the fail-closed
// verifier (S2), pre/post state-diff verification (S1), and the denial
// circuit-breaker (S3). Everything here is pure and offline.

private struct VerifierError: Error {}

final class TesseraSafetySpineTests: XCTestCase {

    private func check(
        _ policy: ApprovalLevel,
        _ profile: TesseraPermissionProfile,
        _ sandbox: Bool,
        _ risk: TesseraActionRisk
    ) -> TesseraSafetyCheck {
        TesseraSafetyDecision(
            approvalPolicy: policy,
            permissionProfile: profile,
            sandboxEnforceable: sandbox,
            actionRisk: risk
        ).check
    }

    // MARK: - S4 layered permission truth table

    func testAutoApproveOnlyWhenSandboxableAndLowRisk() {
        // The fail-safe core: with a permissive policy and standard profile.
        XCTAssertEqual(check(.auto, .standard, true, .low), .autoApprove)
        XCTAssertEqual(check(.notify, .standard, true, .low), .autoApprove)

        // Not sandboxable -> ask, even at low risk.
        XCTAssertEqual(check(.auto, .standard, false, .low), .askUser)

        // Sandboxable but elevated risk -> ask.
        XCTAssertEqual(check(.auto, .standard, true, .medium), .askUser)
        XCTAssertEqual(check(.auto, .standard, true, .high), .askUser)

        // Neither sandboxable nor low -> ask.
        XCTAssertEqual(check(.auto, .standard, false, .high), .askUser)
    }

    func testForbiddenIsAlwaysRejected() {
        for policy in ApprovalLevel.allCases {
            for sandbox in [true, false] {
                XCTAssertEqual(check(policy, .standard, sandbox, .forbidden), .reject)
            }
        }
    }

    func testDeniedPolicyAndPromptPolicy() {
        // A disabled tool never runs, regardless of risk or sandbox.
        XCTAssertEqual(check(.denied, .standard, true, .low), .reject)
        // An explicit prompt policy always asks.
        XCTAssertEqual(check(.prompt, .standard, true, .low), .askUser)
    }

    func testRestrictedProfileNeverAutoApproves() {
        // Even the ideal sandboxable low-risk case asks under a restricted profile.
        XCTAssertEqual(check(.auto, .restricted, true, .low), .askUser)
        // Elevated behaves like the standard fail-safe.
        XCTAssertEqual(check(.auto, .elevated, true, .low), .autoApprove)
    }

    // MARK: - S2 fail-closed verifier

    func testVerifierFailsClosedOnError() {
        let verifier = TesseraActionVerifier(assess: { _ in throw VerifierError() })
        let decision = verifier.verify(PendingAction(toolName: "list_models"))
        XCTAssertFalse(decision.authorized)
        XCTAssertEqual(decision.riskLevel, .high)
    }

    func testRuleBasedVerifierAuthorizesByRisk() {
        let verifier = TesseraActionVerifier()

        let benign = verifier.verify(PendingAction(toolName: "list_models"))
        XCTAssertTrue(benign.authorized)
        XCTAssertEqual(benign.riskLevel, .low)

        let mutating = verifier.verify(PendingAction(toolName: "quantize"))
        XCTAssertTrue(mutating.authorized)
        XCTAssertEqual(mutating.riskLevel, .medium)

        let destructive = verifier.verify(PendingAction(toolName: "delete_model"))
        XCTAssertFalse(destructive.authorized)
        XCTAssertEqual(destructive.riskLevel, .high)

        // Unknown tools are treated cautiously (medium), never trusted as low.
        let unknown = verifier.verify(PendingAction(toolName: "frobnicate"))
        XCTAssertEqual(unknown.riskLevel, .medium)
        XCTAssertTrue(unknown.authorized)
    }

    // MARK: - S1 pre/post state-diff verification

    func testVerifyStateChangeDetectsChangedKey() {
        let pre = ["status": "pending", "count": "1"]
        let post = ["status": "done", "count": "1"]
        XCTAssertTrue(TesseraActionVerifier.verifyStateChange(pre: pre, post: post, expect: "status"))
        // An untouched key is not a change.
        XCTAssertFalse(TesseraActionVerifier.verifyStateChange(pre: pre, post: post, expect: "count"))
    }

    func testVerifyStateChangeDetectsAddRemoveAndAbsence() {
        // Key added.
        XCTAssertTrue(TesseraActionVerifier.verifyStateChange(pre: [:], post: ["file": "x.gguf"], expect: "file"))
        // Key removed.
        XCTAssertTrue(TesseraActionVerifier.verifyStateChange(pre: ["file": "x.gguf"], post: [:], expect: "file"))
        // Absent in both -> no change.
        XCTAssertFalse(TesseraActionVerifier.verifyStateChange(pre: [:], post: [:], expect: "file"))
    }

    func testDictionaryEvidenceSnapshot() {
        var state = ["status": "pending"]
        let evidence = TesseraDictionaryEvidence { state }
        let pre = evidence.snapshot()
        state["status"] = "done"
        let post = evidence.snapshot()
        XCTAssertTrue(TesseraActionVerifier.verifyStateChange(pre: pre, post: post, expect: "status"))
    }

    // MARK: - S3 denial circuit-breaker

    func testThreeConsecutiveDenialsTrips() {
        let breaker = TesseraDenialCircuitBreaker()
        breaker.record(denied: true)
        breaker.record(denied: true)
        XCTAssertFalse(breaker.isTripped)
        breaker.record(denied: true)
        XCTAssertTrue(breaker.isTripped)
    }

    func testConsecutiveRunResetsOnApproval() {
        let breaker = TesseraDenialCircuitBreaker()
        breaker.record(denied: true)
        breaker.record(denied: true)
        breaker.record(denied: false)   // breaks the consecutive run
        breaker.record(denied: true)
        breaker.record(denied: true)
        XCTAssertFalse(breaker.isTripped)   // only 2 consecutive, 4 total
    }

    func testTenOfLastFiftyTrips() {
        let breaker = TesseraDenialCircuitBreaker()
        // 10 denials, each separated by 4 approvals: never 3 consecutive,
        // but 10 of the last 50 are denials.
        for _ in 0..<10 {
            breaker.record(denied: true)
            for _ in 0..<4 { breaker.record(denied: false) }
        }
        XCTAssertTrue(breaker.isTripped)
    }

    func testNineOfFiftyDoesNotTrip() {
        let breaker = TesseraDenialCircuitBreaker()
        for _ in 0..<9 {
            breaker.record(denied: true)
            for _ in 0..<4 { breaker.record(denied: false) }
        }
        // 9 denials, 36 approvals = 45 outcomes; pad to a full 50-window.
        for _ in 0..<5 { breaker.record(denied: false) }
        XCTAssertFalse(breaker.isTripped)
    }

    func testResetClearsTrippedState() {
        let breaker = TesseraDenialCircuitBreaker()
        for _ in 0..<3 { breaker.record(denied: true) }
        XCTAssertTrue(breaker.isTripped)
        breaker.reset()
        XCTAssertFalse(breaker.isTripped)
    }

    // MARK: - Approval engine hook (integration)

    @MainActor
    func testApprovalEngineSafetyCheckHook() {
        let engine = TesseraApprovalEngine()
        engine.setOverride(.auto, for: "list_models")

        // Benign, contained, low-risk action auto-approves.
        let benign = engine.safetyCheck(
            for: PendingAction(toolName: "list_models"),
            sandboxEnforceable: true
        )
        XCTAssertEqual(benign, .autoApprove)

        // Three rejected actions trip the breaker.
        engine.setOverride(.denied, for: "bad_tool")
        for _ in 0..<3 {
            let rejected = engine.safetyCheck(
                for: PendingAction(toolName: "bad_tool"),
                sandboxEnforceable: true
            )
            XCTAssertEqual(rejected, .reject)
        }
        XCTAssertTrue(engine.circuitBreaker.isTripped)

        // Once tripped, even a benign action is rejected.
        let afterTrip = engine.safetyCheck(
            for: PendingAction(toolName: "list_models"),
            sandboxEnforceable: true
        )
        XCTAssertEqual(afterTrip, .reject)
    }
}
