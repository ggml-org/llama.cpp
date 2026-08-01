import XCTest
@testable import TesseraCore

// Autonomy calibration Phase A tests (autonomy-calibration-design.md 17):
// classifier, irreversible guard, ratchet state machine, precedence,
// breaker interaction, floor/ceiling, persistence, receipt emission,
// and recommendation-confirmation. Everything is pure and offline.

final class TesseraAutonomyTests: XCTestCase {

    /// A temp-directory store so tests never touch the real ApplicationSupport.
    private func makeService() -> TesseraAutonomyService {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tessera-autonomy-tests-\(UUID().uuidString)", isDirectory: true)
        let store = TesseraLearningStore(directory: dir)
        return TesseraAutonomyService(store: store)
    }

    // MARK: - Classifier (section 3)

    func testVerbPrefixClassGit() {
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])
        XCTAssertEqual(TesseraActionClass.classify(action), "bash:git")
    }

    func testVerbPrefixClassNpm() {
        let action = PendingAction(toolName: "shell", arguments: ["cmd": .string("npm install")])
        XCTAssertEqual(TesseraActionClass.classify(action), "shell:npm")
    }

    func testVerbPrefixStripsPath() {
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("/usr/bin/git diff")])
        XCTAssertEqual(TesseraActionClass.classify(action), "bash:git")
    }

    func testVerbPrefixUnknownProgram() {
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("python script.py")])
        XCTAssertEqual(TesseraActionClass.classify(action), "bash:python")
    }

    func testVerbPrefixNoCommand() {
        let action = PendingAction(toolName: "bash", arguments: [:])
        XCTAssertEqual(TesseraActionClass.classify(action), "bash")
    }

    func testPathGlobClassDefaultDepth() {
        let action = PendingAction(toolName: "file_write", arguments: ["path": .string("src/Agent/Loop.swift")])
        XCTAssertEqual(TesseraActionClass.classify(action), "file_write:src/**")
    }

    func testPathGlobClassDepthTwo() {
        let action = PendingAction(toolName: "file_write", arguments: ["path": .string("src/Agent/Loop.swift")])
        XCTAssertEqual(TesseraActionClass.classify(action, pathGlobDepth: 2), "file_write:src/Agent/**")
    }

    func testPathGlobExternalAbsolute() {
        let action = PendingAction(toolName: "file_write", arguments: ["path": .string("/etc/passwd")])
        XCTAssertEqual(TesseraActionClass.classify(action), "file_write:<external>")
    }

    func testPathGlobExternalHome() {
        let action = PendingAction(toolName: "file_edit", arguments: ["path": .string("~/Documents/notes.md")])
        XCTAssertEqual(TesseraActionClass.classify(action), "file_edit:<external>")
    }

    func testPathGlobExternalParent() {
        let action = PendingAction(toolName: "file_write", arguments: ["path": .string("../other/file.txt")])
        XCTAssertEqual(TesseraActionClass.classify(action), "file_write:<external>")
    }

    func testArgShapeClass() {
        let a1 = PendingAction(toolName: "quantize", arguments: ["model": .string("a.gguf"), "bits": .number(4)])
        let a2 = PendingAction(toolName: "quantize", arguments: ["model": .string("b.gguf"), "bits": .number(8)])
        let c1 = TesseraActionClass.classify(a1)
        let c2 = TesseraActionClass.classify(a2)
        // Same argument keys -> same class regardless of values.
        XCTAssertEqual(c1, c2)
        XCTAssertTrue(c1.hasPrefix("quantize#"))
    }

    func testArgShapeDifferentKeys() {
        let a1 = PendingAction(toolName: "quantize", arguments: ["model": .string("a.gguf")])
        let a2 = PendingAction(toolName: "quantize", arguments: ["path": .string("b.gguf")])
        XCTAssertNotEqual(TesseraActionClass.classify(a1), TesseraActionClass.classify(a2))
    }

    func testFallbackToolOnly() {
        let action = PendingAction(toolName: "frobnicate", arguments: [:])
        XCTAssertEqual(TesseraActionClass.classify(action), "frobnicate")
    }

    func testClassifierIsDeterministic() {
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("cargo build")])
        let first = TesseraActionClass.classify(action)
        let second = TesseraActionClass.classify(action)
        XCTAssertEqual(first, second)
    }

    // MARK: - Irreversible guard (section 4)

    func testIrreversibleDestructiveVerb() {
        XCTAssertTrue(TesseraActionClass.isIrreversible("bash:rm", risk: .low))
        XCTAssertTrue(TesseraActionClass.isIrreversible("bash:sudo", risk: .low))
        XCTAssertTrue(TesseraActionClass.isIrreversible("bash:kill", risk: .medium))
    }

    func testIrreversibleHighRisk() {
        XCTAssertTrue(TesseraActionClass.isIrreversible("delete_model", risk: .high))
        XCTAssertTrue(TesseraActionClass.isIrreversible("something", risk: .forbidden))
    }

    func testIrreversibleExternalWrite() {
        XCTAssertTrue(TesseraActionClass.isIrreversible("file_write:<external>", risk: .low))
    }

    func testIrreversibleManualDenylist() {
        XCTAssertTrue(TesseraActionClass.isIrreversible("bash:git", risk: .low, denylist: ["bash:git"]))
    }

    func testNotIrreversibleSafeClass() {
        XCTAssertFalse(TesseraActionClass.isIrreversible("bash:git", risk: .low))
        XCTAssertFalse(TesseraActionClass.isIrreversible("file_write:src/**", risk: .medium))
        XCTAssertFalse(TesseraActionClass.isIrreversible("list_models", risk: .low))
    }

    func testVerbHeadExtraction() {
        XCTAssertEqual(TesseraActionClass.verbHead(of: "bash:git"), "git")
        XCTAssertEqual(TesseraActionClass.verbHead(of: "bash:rm"), "rm")
        XCTAssertNil(TesseraActionClass.verbHead(of: "file_write:src/**"))
        XCTAssertNil(TesseraActionClass.verbHead(of: "file_write:<external>"))
        XCTAssertNil(TesseraActionClass.verbHead(of: "list_models"))
    }

    // MARK: - Ratchet state machine (section 6)

    func testGrantAfterThreshold() {
        let svc = makeService()
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])

        // 5 approvals across 3 sessions -> granted.
        for session in ["s1", "s1", "s2", "s2", "s3"] {
            svc.record(action: action, risk: .low, sandboxed: true, decision: .askUser,
                       userChoice: .approved, source: "rule", sessionID: session)
        }
        let entry = svc.entry(for: "bash:git")
        XCTAssertNotNil(entry)
        XCTAssertTrue(entry!.granted)
        XCTAssertEqual(entry!.consecutiveApprovals, 5)
        XCTAssertEqual(entry!.distinctSessions, 3)
    }

    func testNoGrantBeforeThreshold() {
        let svc = makeService()
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])

        // Only 4 approvals (need 5).
        for session in ["s1", "s1", "s2", "s3"] {
            svc.record(action: action, risk: .low, sandboxed: true, decision: .askUser,
                       userChoice: .approved, source: "rule", sessionID: session)
        }
        XCTAssertFalse(svc.entry(for: "bash:git")!.granted)
    }

    func testNoGrantWithoutEnoughSessions() {
        let svc = makeService()
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])

        // 5 approvals but only 2 sessions (need 3).
        for session in ["s1", "s1", "s1", "s2", "s2"] {
            svc.record(action: action, risk: .low, sandboxed: true, decision: .askUser,
                       userChoice: .approved, source: "rule", sessionID: session)
        }
        XCTAssertFalse(svc.entry(for: "bash:git")!.granted)
    }

    func testSingleDenialRevokes() {
        let svc = makeService()
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])

        // Grant it.
        for session in ["s1", "s1", "s2", "s2", "s3"] {
            svc.record(action: action, risk: .low, sandboxed: true, decision: .askUser,
                       userChoice: .approved, source: "rule", sessionID: session)
        }
        XCTAssertTrue(svc.entry(for: "bash:git")!.granted)

        // One denial revokes.
        svc.record(action: action, risk: .low, sandboxed: true, decision: .askUser,
                   userChoice: .denied, source: "rule", sessionID: "s4")
        let entry = svc.entry(for: "bash:git")!
        XCTAssertFalse(entry.granted)
        XCTAssertEqual(entry.consecutiveApprovals, 0)
        XCTAssertEqual(entry.totalDenials, 1)
    }

    func testIrreversibleNeverGrants() {
        let svc = makeService()
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("rm -rf /")])

        // Many approvals, but rm is irreversible.
        for session in ["s1", "s1", "s2", "s2", "s3", "s3", "s4"] {
            svc.record(action: action, risk: .high, sandboxed: false, decision: .askUser,
                       userChoice: .approved, source: "rule", sessionID: session)
        }
        let entry = svc.entry(for: "bash:rm")!
        XCTAssertTrue(entry.irreversible)
        XCTAssertFalse(entry.granted)
    }

    func testRevokedStaysRevoked() {
        let svc = makeService()
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])

        // Grant it.
        for session in ["s1", "s1", "s2", "s2", "s3"] {
            svc.record(action: action, risk: .low, sandboxed: true, decision: .askUser,
                       userChoice: .approved, source: "rule", sessionID: session)
        }
        XCTAssertTrue(svc.entry(for: "bash:git")!.granted)

        // Revoke manually.
        svc.revoke("bash:git")
        XCTAssertTrue(svc.entry(for: "bash:git")!.revoked)
        XCTAssertFalse(svc.entry(for: "bash:git")!.granted)

        // More approvals do not re-grant a revoked class.
        for session in ["s4", "s4", "s5", "s5", "s6"] {
            svc.record(action: action, risk: .low, sandboxed: true, decision: .askUser,
                       userChoice: .approved, source: "rule", sessionID: session)
        }
        XCTAssertFalse(svc.entry(for: "bash:git")!.granted)
    }

    func testUnrevokeResumesFromZero() {
        let svc = makeService()
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])

        for session in ["s1", "s1", "s2", "s2", "s3"] {
            svc.record(action: action, risk: .low, sandboxed: true, decision: .askUser,
                       userChoice: .approved, source: "rule", sessionID: session)
        }
        svc.revoke("bash:git")
        svc.unrevoke("bash:git")

        let entry = svc.entry(for: "bash:git")!
        XCTAssertFalse(entry.revoked)
        XCTAssertEqual(entry.consecutiveApprovals, 0)
        XCTAssertFalse(entry.granted)
    }

    func testDistinctSessionCounting() {
        let svc = makeService()
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])

        // Same session multiple times -> 1 distinct session.
        for _ in 0..<5 {
            svc.record(action: action, risk: .low, sandboxed: true, decision: .askUser,
                       userChoice: .approved, source: "rule", sessionID: "s1")
        }
        XCTAssertEqual(svc.entry(for: "bash:git")!.distinctSessions, 1)
    }

    // MARK: - Precedence (section 7)

    func testPrecedenceRejectBeatsEverything() {
        let svc = makeService()
        let resolution = svc.resolve(base: .reject, actionClass: "bash:git", risk: .low, sandboxEnforceable: true)
        XCTAssertEqual(resolution.check, .reject)
    }

    func testPrecedenceIrreversibleBlocksAutoApprove() {
        let svc = makeService()
        // Even if the base says autoApprove, an irreversible class asks.
        let resolution = svc.resolve(base: .autoApprove, actionClass: "bash:rm", risk: .high, sandboxEnforceable: true)
        XCTAssertEqual(resolution.check, .askUser)
    }

    func testPrecedenceBaseAutoApprovePassesThrough() {
        let svc = makeService()
        let resolution = svc.resolve(base: .autoApprove, actionClass: "bash:git", risk: .low, sandboxEnforceable: true)
        XCTAssertEqual(resolution.check, .autoApprove)
        XCTAssertEqual(resolution.source, "rule")
    }

    func testPrecedenceLearnedPromotion() {
        let svc = makeService()
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])

        // Grant the class.
        for session in ["s1", "s1", "s2", "s2", "s3"] {
            svc.record(action: action, risk: .low, sandboxed: true, decision: .askUser,
                       userChoice: .approved, source: "rule", sessionID: session)
        }

        // Now a base askUser promotes to autoApprove.
        let resolution = svc.resolve(base: .askUser, actionClass: "bash:git", risk: .low, sandboxEnforceable: true)
        XCTAssertEqual(resolution.check, .autoApprove)
        XCTAssertEqual(resolution.source, "ratchet")
    }

    func testPrecedenceUngrantedStaysAskUser() {
        let svc = makeService()
        let resolution = svc.resolve(base: .askUser, actionClass: "bash:git", risk: .low, sandboxEnforceable: true)
        XCTAssertEqual(resolution.check, .askUser)
        XCTAssertEqual(resolution.source, "rule")
    }

    // MARK: - Floor/ceiling (section 9)

    func testContainedLowRiskCeilingBlocksMediumRisk() {
        let svc = makeService()
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("npm install")])

        // Grant the class (medium risk).
        for session in ["s1", "s1", "s2", "s2", "s3"] {
            svc.record(action: action, risk: .medium, sandboxed: true, decision: .askUser,
                       userChoice: .approved, source: "rule", sessionID: session)
        }

        // Default ceiling is containedLowRiskOnly: medium risk does NOT promote.
        let resolution = svc.resolve(base: .askUser, actionClass: "bash:npm", risk: .medium, sandboxEnforceable: true)
        XCTAssertEqual(resolution.check, .askUser)
    }

    func testAnyNonIrreversibleCeilingAllowsMediumRisk() {
        let svc = makeService()
        svc.updateConfig { $0.ceiling = .anyNonIrreversible }
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("npm install")])

        for session in ["s1", "s1", "s2", "s2", "s3"] {
            svc.record(action: action, risk: .medium, sandboxed: true, decision: .askUser,
                       userChoice: .approved, source: "rule", sessionID: session)
        }

        let resolution = svc.resolve(base: .askUser, actionClass: "bash:npm", risk: .medium, sandboxEnforceable: true)
        XCTAssertEqual(resolution.check, .autoApprove)
        XCTAssertEqual(resolution.source, "ratchet")
    }

    func testContainedLowRiskCeilingRequiresSandbox() {
        let svc = makeService()
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])

        for session in ["s1", "s1", "s2", "s2", "s3"] {
            svc.record(action: action, risk: .low, sandboxed: false, decision: .askUser,
                       userChoice: .approved, source: "rule", sessionID: session)
        }

        // Low risk but not sandboxed -> no promotion under containedLowRiskOnly.
        let resolution = svc.resolve(base: .askUser, actionClass: "bash:git", risk: .low, sandboxEnforceable: false)
        XCTAssertEqual(resolution.check, .askUser)
    }

    // MARK: - Breaker interaction (section 8)

    @MainActor
    func testBreakerTripSuspendsGrants() {
        let svc = makeService()
        let engine = TesseraApprovalEngine()
        engine.autonomy = svc

        let action = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])

        // Grant the class.
        for session in ["s1", "s1", "s2", "s2", "s3"] {
            svc.record(action: action, risk: .low, sandboxed: true, decision: .askUser,
                       userChoice: .approved, source: "rule", sessionID: session)
        }

        // Trip the breaker.
        for _ in 0..<3 { engine.circuitBreaker.record(denied: true) }
        XCTAssertTrue(engine.circuitBreaker.isTripped)

        // gateCheck should reject (breaker outranks ratchet).
        let gate = engine.gateCheck(for: action, sandboxEnforceable: true)
        XCTAssertEqual(gate.check, .reject)
    }

    // MARK: - Receipt emission (section 14)

    func testReceiptEmittedOnApproval() {
        let svc = makeService()
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])

        let receipt = svc.record(action: action, risk: .low, sandboxed: true, decision: .askUser,
                                 userChoice: .approved, source: "rule", sessionID: "s1")
        XCTAssertEqual(receipt.kind, "approval")
        XCTAssertEqual(receipt.payload["actionClass"], .string("bash:git"))
        XCTAssertEqual(receipt.payload["userChoice"], .string("approved"))
        XCTAssertEqual(receipt.payload["source"], .string("rule"))
        XCTAssertEqual(receipt.payload["sessionID"], .string("s1"))
    }

    func testReceiptEmittedOnAutoApprove() {
        let svc = makeService()
        let action = PendingAction(toolName: "list_models", arguments: [:])

        let receipt = svc.record(action: action, risk: .low, sandboxed: true, decision: .autoApprove,
                                 userChoice: .none, source: "rule", sessionID: "s1")
        XCTAssertEqual(receipt.kind, "approval")
        XCTAssertEqual(receipt.payload["userChoice"], .string("none"))
        XCTAssertEqual(receipt.payload["decision"], .string("autoApprove"))
    }

    func testReceiptsPersisted() {
        let svc = makeService()
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])

        svc.record(action: action, risk: .low, sandboxed: true, decision: .askUser,
                   userChoice: .approved, source: "rule", sessionID: "s1")
        svc.record(action: action, risk: .low, sandboxed: true, decision: .askUser,
                   userChoice: .approved, source: "rule", sessionID: "s2")

        let receipts = svc.recentReceipts(limit: 10)
        XCTAssertEqual(receipts.count, 2)
        // Newest first.
        XCTAssertEqual(receipts[0].payload["sessionID"], .string("s2"))
    }

    // MARK: - Recommendations (section 11.9)

    func testRecommendationTrigger() {
        let svc = makeService()
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])

        // 3 approvals (recommendationFloor default = 3) across 2 sessions.
        for session in ["s1", "s1", "s2"] {
            svc.record(action: action, risk: .low, sandboxed: true, decision: .askUser,
                       userChoice: .approved, source: "rule", sessionID: session)
        }

        let recs = svc.recommendations()
        XCTAssertEqual(recs.count, 1)
        XCTAssertEqual(recs[0].actionClass, "bash:git")
        XCTAssertTrue(recs[0].message.contains("bash:git"))
    }

    func testNoRecommendationWhenGranted() {
        let svc = makeService()
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])

        // Grant it (5 approvals, 3 sessions).
        for session in ["s1", "s1", "s2", "s2", "s3"] {
            svc.record(action: action, risk: .low, sandboxed: true, decision: .askUser,
                       userChoice: .approved, source: "rule", sessionID: session)
        }

        // Already granted -> no recommendation.
        XCTAssertTrue(svc.recommendations().isEmpty)
    }

    func testNoRecommendationForIrreversible() {
        let svc = makeService()
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("rm file.txt")])

        for session in ["s1", "s1", "s2"] {
            svc.record(action: action, risk: .high, sandboxed: false, decision: .askUser,
                       userChoice: .approved, source: "rule", sessionID: session)
        }

        // Irreversible -> no recommendation.
        XCTAssertTrue(svc.recommendations().isEmpty)
    }

    func testConfirmRecommendationGrants() {
        let svc = makeService()
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])

        for session in ["s1", "s1", "s2"] {
            svc.record(action: action, risk: .low, sandboxed: true, decision: .askUser,
                       userChoice: .approved, source: "rule", sessionID: session)
        }

        let receipt = svc.confirmRecommendation(actionClass: "bash:git", choice: .confirm, sessionID: "s3")
        XCTAssertTrue(svc.entry(for: "bash:git")!.granted)
        XCTAssertEqual(receipt.payload["source"], .string("recommendation"))
    }

    func testNeverRecommendationDenylists() {
        let svc = makeService()
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])

        for session in ["s1", "s1", "s2"] {
            svc.record(action: action, risk: .low, sandboxed: true, decision: .askUser,
                       userChoice: .approved, source: "rule", sessionID: session)
        }

        svc.confirmRecommendation(actionClass: "bash:git", choice: .never, sessionID: "s3")
        XCTAssertTrue(svc.entry(for: "bash:git")!.irreversible)

        // Now it is irreversible: resolve should askUser even with a grant.
        let resolution = svc.resolve(base: .askUser, actionClass: "bash:git", risk: .low, sandboxEnforceable: true)
        XCTAssertEqual(resolution.check, .askUser)
    }

    func testNotNowLeavesAccumulating() {
        let svc = makeService()
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])

        for session in ["s1", "s1", "s2"] {
            svc.record(action: action, risk: .low, sandboxed: true, decision: .askUser,
                       userChoice: .approved, source: "rule", sessionID: session)
        }

        svc.confirmRecommendation(actionClass: "bash:git", choice: .notNow, sessionID: "s3")
        let entry = svc.entry(for: "bash:git")!
        XCTAssertFalse(entry.granted)
        XCTAssertFalse(entry.irreversible)
        XCTAssertEqual(entry.consecutiveApprovals, 3)
    }

    // MARK: - Audit and revocation (section 13)

    func testDenylistMarksIrreversible() {
        let svc = makeService()
        svc.denylist("bash:git")
        XCTAssertTrue(svc.isIrreversible("bash:git", risk: .low))
    }

    func testUndenylistRemoves() {
        let svc = makeService()
        svc.denylist("bash:git")
        svc.undenylist("bash:git")
        XCTAssertFalse(svc.isIrreversible("bash:git", risk: .low))
    }

    func testResetAllClearsGrants() {
        let svc = makeService()
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])

        for session in ["s1", "s1", "s2", "s2", "s3"] {
            svc.record(action: action, risk: .low, sandboxed: true, decision: .askUser,
                       userChoice: .approved, source: "rule", sessionID: session)
        }
        XCTAssertTrue(svc.entry(for: "bash:git")!.granted)

        svc.resetAll()
        XCTAssertFalse(svc.entry(for: "bash:git")!.granted)
        // Entries kept for audit.
        XCTAssertNotNil(svc.entry(for: "bash:git"))
    }

    // MARK: - Persistence (section 5)

    func testPersistenceRoundTrip() {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tessera-autonomy-persist-\(UUID().uuidString)", isDirectory: true)
        let store = TesseraLearningStore(directory: dir)

        // Write with one instance.
        let svc1 = TesseraAutonomyService(store: store)
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])
        for session in ["s1", "s1", "s2", "s2", "s3"] {
            svc1.record(action: action, risk: .low, sandboxed: true, decision: .askUser,
                        userChoice: .approved, source: "rule", sessionID: session)
        }
        XCTAssertTrue(svc1.entry(for: "bash:git")!.granted)

        // Read with a fresh instance.
        let svc2 = TesseraAutonomyService(store: store)
        XCTAssertTrue(svc2.entry(for: "bash:git")!.granted)
        XCTAssertEqual(svc2.entry(for: "bash:git")!.totalApprovals, 5)
    }

    func testCorruptFileDecodesToDefault() {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tessera-autonomy-corrupt-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        // Write garbage.
        try? "not json".data(using: .utf8)!.write(to: dir.appendingPathComponent("learned-permissions.json"))

        let store = TesseraLearningStore(directory: dir)
        let svc = TesseraAutonomyService(store: store)
        // Should get defaults, not crash.
        XCTAssertTrue(svc.entries().isEmpty)
        XCTAssertEqual(svc.config.grantThresholdN, 5)
    }

    // MARK: - Purge

    func testPurgeClearsEverything() {
        let svc = makeService()
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])
        svc.record(action: action, risk: .low, sandboxed: true, decision: .askUser,
                   userChoice: .approved, source: "rule", sessionID: "s1")

        let removed = try! svc.purgeTrainingData()
        XCTAssertEqual(removed, 1)
        XCTAssertTrue(svc.entries().isEmpty)
        XCTAssertTrue(svc.recentReceipts().isEmpty)
    }

    // MARK: - Gate integration (approval engine)

    @MainActor
    func testGateCheckWithNoopAutonomy() {
        let engine = TesseraApprovalEngine()
        // Default: noop autonomy, so gateCheck == safetyCheck behavior.
        let action = PendingAction(toolName: "list_models", arguments: [:])
        engine.setOverride(.auto, for: "list_models")
        let gate = engine.gateCheck(for: action, sandboxEnforceable: true)
        XCTAssertEqual(gate.check, .autoApprove)
        XCTAssertEqual(gate.source, "rule")
    }

    @MainActor
    func testGateCheckWithRatchetPromotion() {
        let svc = makeService()
        let engine = TesseraApprovalEngine()
        engine.autonomy = svc

        // Use a read-only tool so ruleBasedRisk gives .low.
        let action = PendingAction(toolName: "list_models", arguments: [:])
        engine.setOverride(.auto, for: "list_models")

        // Grant the class (5 approvals, 3 sessions).
        for session in ["s1", "s1", "s2", "s2", "s3"] {
            svc.record(action: action, risk: .low, sandboxed: true, decision: .askUser,
                       userChoice: .approved, source: "rule", sessionID: session)
        }

        // With sandbox enforceable + low risk, base is already autoApprove.
        let gate = engine.gateCheck(for: action, sandboxEnforceable: true)
        XCTAssertEqual(gate.check, .autoApprove)
        XCTAssertEqual(gate.source, "rule")

        // Without sandbox, base is askUser. Ratchet promotes because the class
        // is granted, low risk, and ceiling is containedLowRiskOnly (but
        // sandboxEnforceable=false blocks the ceiling check).
        let gate2 = engine.gateCheck(for: action, sandboxEnforceable: false)
        XCTAssertEqual(gate2.check, .askUser)

        // Raise the ceiling: now the ratchet promotes even without sandbox.
        svc.updateConfig { $0.ceiling = .anyNonIrreversible }
        let gate3 = engine.gateCheck(for: action, sandboxEnforceable: false)
        XCTAssertEqual(gate3.check, .autoApprove)
        XCTAssertEqual(gate3.source, "ratchet")
    }

    @MainActor
    func testRecordOutcomeEmitsReceipt() {
        let svc = makeService()
        let engine = TesseraApprovalEngine()
        engine.autonomy = svc

        let action = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])
        let receipt = engine.recordOutcome(
            action: action, risk: .low, sandboxed: true,
            decision: .askUser, userChoice: .approved, source: "rule", sessionID: "s1"
        )
        XCTAssertEqual(receipt.kind, "approval")
        XCTAssertEqual(svc.recentReceipts().count, 1)
    }
}
