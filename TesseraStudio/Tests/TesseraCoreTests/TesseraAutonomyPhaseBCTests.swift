import XCTest
@testable import TesseraCore

// Autonomy calibration Phase B + C tests (autonomy-calibration-design.md 17):
// scoped YOLO, dispositional floor, breaker suspension/restore, miscalibration
// regime-shift detection, and the leashed neural approver (fail-closed, leash,
// collapse guard, smart YOLO, NL decoupling). Everything is pure and offline.

final class TesseraAutonomyPhaseBCTests: XCTestCase {

    /// A temp-directory store so tests never touch the real ApplicationSupport.
    private func makeService() -> TesseraAutonomyService {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tessera-autonomy-bc-tests-\(UUID().uuidString)", isDirectory: true)
        let store = TesseraLearningStore(directory: dir)
        return TesseraAutonomyService(store: store)
    }

    /// Grant a class the slow way: N approvals across M sessions.
    private func grant(_ svc: TesseraAutonomyService, command: String = "git status", risk: TesseraActionRisk = .low) {
        let action = PendingAction(toolName: "bash", arguments: ["command": .string(command)])
        for session in ["s1", "s1", "s2", "s2", "s3"] {
            svc.record(action: action, risk: risk, sandboxed: true, decision: .askUser,
                       userChoice: .approved, source: "rule", sessionID: session)
        }
    }

    /// Feature probe matching what resolve() feeds the net for a fresh class.
    private func features(
        _ actionClass: String,
        risk: TesseraActionRisk = .low,
        sandboxed: Bool = true,
        yoloActive: Bool = false
    ) -> [Double] {
        TesseraApproverFeatures.extract(
            actionClass: actionClass,
            risk: risk,
            sandboxed: sandboxed,
            entry: nil,
            yoloActive: yoloActive,
            recentDenialRate: 0,
            secondsSinceLastDenial: nil,
            config: TesseraPermissionConfig()
        )
    }

    // MARK: - Scoped YOLO (section 10)

    func testYoloAutoApprovesInScope() {
        let svc = makeService()
        svc.startYolo(sessionID: "s1", minutes: 30)
        let resolution = svc.resolve(base: .askUser, actionClass: "bash:git",
                                     risk: .low, sandboxEnforceable: false, sessionID: "s1")
        XCTAssertEqual(resolution.check, .autoApprove)
        XCTAssertEqual(resolution.source, "yolo")
    }

    func testYoloWrongSessionDoesNotApply() {
        let svc = makeService()
        svc.startYolo(sessionID: "s1", minutes: 30)
        let resolution = svc.resolve(base: .askUser, actionClass: "bash:git",
                                     risk: .low, sandboxEnforceable: false, sessionID: "other")
        XCTAssertEqual(resolution.check, .askUser)
    }

    func testYoloExpiredDoesNotApply() {
        let svc = makeService()
        // Negative duration: already expired on arrival.
        svc.startYolo(sessionID: "s1", minutes: -1)
        XCTAssertNil(svc.activeYolo(for: "s1"))
        let resolution = svc.resolve(base: .askUser, actionClass: "bash:git",
                                     risk: .low, sandboxEnforceable: false, sessionID: "s1")
        XCTAssertEqual(resolution.check, .askUser)
    }

    func testYoloIrreversibleStillPrompts() {
        let svc = makeService()
        svc.startYolo(sessionID: "s1", minutes: 30)
        let resolution = svc.resolve(base: .askUser, actionClass: "bash:rm",
                                     risk: .high, sandboxEnforceable: false, sessionID: "s1")
        XCTAssertEqual(resolution.check, .askUser)
        XCTAssertEqual(resolution.source, "rule")
    }

    func testYoloLeavesRejectUnchanged() {
        let svc = makeService()
        svc.startYolo(sessionID: "s1", minutes: 30)
        let resolution = svc.resolve(base: .reject, actionClass: "bash:git",
                                     risk: .low, sandboxEnforceable: true, sessionID: "s1")
        XCTAssertEqual(resolution.check, .reject)
    }

    func testYoloUsesConfiguredDefaultMinutes() {
        let svc = makeService()
        svc.updateConfig { $0.yoloDefaultMinutes = 45 }
        let session = svc.startYolo(sessionID: "s1")
        let remaining = session.expiresAt.timeIntervalSinceNow
        XCTAssertGreaterThan(remaining, 40 * 60)
        XCTAssertLessThanOrEqual(remaining, 45 * 60)
    }

    func testYoloSummaryCountsActionsAndDenials() {
        let svc = makeService()
        svc.startYolo(goal: "ship it", sessionID: "s1", reason: "test", minutes: 30)
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])
        svc.record(action: action, risk: .low, sandboxed: true, decision: .autoApprove,
                   userChoice: .none, source: "yolo", sessionID: "s1")
        svc.record(action: action, risk: .low, sandboxed: true, decision: .askUser,
                   userChoice: .denied, source: "yolo", sessionID: "s1")
        let summary = svc.endYolo()
        XCTAssertEqual(summary?.actionCount, 2)
        XCTAssertEqual(summary?.denials, 1)
        XCTAssertEqual(summary?.classes, ["bash:git"])
        // Ended: YOLO no longer applies.
        XCTAssertNil(svc.activeYolo(for: "s1"))
        let resolution = svc.resolve(base: .askUser, actionClass: "bash:git",
                                     risk: .low, sandboxEnforceable: false, sessionID: "s1")
        XCTAssertEqual(resolution.check, .askUser)
    }

    func testYoloReceiptFlagged() {
        let svc = makeService()
        svc.startYolo(sessionID: "s1", minutes: 30)
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])
        let receipt = svc.record(action: action, risk: .low, sandboxed: true, decision: .autoApprove,
                                 userChoice: .none, source: "yolo", sessionID: "s1")
        XCTAssertEqual(receipt.payload["yoloActive"], .bool(true))
    }

    func testYoloNeverPersistsAcrossInstances() {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tessera-yolo-persist-\(UUID().uuidString)", isDirectory: true)
        let store = TesseraLearningStore(directory: dir)
        let svc1 = TesseraAutonomyService(store: store)
        svc1.startYolo(sessionID: "s1", minutes: 30)
        XCTAssertNotNil(svc1.activeYolo(for: "s1"))
        // A fresh instance over the same store must NOT inherit YOLO.
        let svc2 = TesseraAutonomyService(store: store)
        XCTAssertNil(svc2.activeYolo(for: "s1"))
    }

    // MARK: - Dispositional floor (section 9)

    func testRestrictedFloorBlocksLearnedPromotion() {
        let svc = makeService()
        svc.updateConfig { $0.floor = .restricted }
        grant(svc)
        // The grant is still RECORDED...
        XCTAssertTrue(svc.entry(for: "bash:git")!.granted)
        // ...but never HONORED: restricted floor blocks all learned promotion.
        let resolution = svc.resolve(base: .askUser, actionClass: "bash:git",
                                     risk: .low, sandboxEnforceable: true)
        XCTAssertEqual(resolution.check, .askUser)
    }

    func testFloorChangeTakesEffectImmediately() {
        let svc = makeService()
        grant(svc)
        let promoted = svc.resolve(base: .askUser, actionClass: "bash:git",
                                   risk: .low, sandboxEnforceable: true)
        XCTAssertEqual(promoted.check, .autoApprove)

        svc.updateConfig { $0.floor = .restricted }
        let blocked = svc.resolve(base: .askUser, actionClass: "bash:git",
                                  risk: .low, sandboxEnforceable: true)
        XCTAssertEqual(blocked.check, .askUser)

        svc.updateConfig { $0.floor = .standard }
        let restored = svc.resolve(base: .askUser, actionClass: "bash:git",
                                   risk: .low, sandboxEnforceable: true)
        XCTAssertEqual(restored.check, .autoApprove)
    }

    // MARK: - Breaker suspension (section 8)

    @MainActor
    func testBreakerResetRestoresGrants() {
        let svc = makeService()
        let engine = TesseraApprovalEngine()
        engine.autonomy = svc
        svc.updateConfig { $0.ceiling = .anyNonIrreversible }
        grant(svc)

        let action = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])

        // Trip the breaker: everything rejects (breaker outranks ratchet).
        for _ in 0..<3 { engine.circuitBreaker.record(denied: true) }
        XCTAssertTrue(engine.circuitBreaker.isTripped)
        XCTAssertEqual(engine.gateCheck(for: action, sandboxEnforceable: false).check, .reject)

        // Reset: grants RESTORE (suspension, not deletion).
        engine.circuitBreaker.reset()
        let gate = engine.gateCheck(for: action, sandboxEnforceable: false)
        XCTAssertEqual(gate.check, .autoApprove)
        XCTAssertEqual(gate.source, "ratchet")
    }

    // MARK: - Miscalibration detector (section 12)

    func testRegimeShiftTightens() {
        var detector = TesseraMiscalibrationDetector(windowSize: 20, hiThreshold: 0.8, loThreshold: 0.3)
        // Consistently approved: sets the high-regime latch.
        for _ in 0..<20 {
            XCTAssertFalse(detector.record(actionClass: "bash:git", approved: true))
        }
        XCTAssertFalse(detector.isTightened("bash:git"))
        // Flip to consistently denied: regime shift triggers tightening.
        var triggered = false
        for _ in 0..<15 {
            triggered = detector.record(actionClass: "bash:git", approved: false) || triggered
        }
        XCTAssertTrue(triggered)
        XCTAssertTrue(detector.isTightened("bash:git"))
    }

    func testStableApprovalStreamDoesNotTighten() {
        var detector = TesseraMiscalibrationDetector()
        for _ in 0..<50 {
            detector.record(actionClass: "bash:git", approved: true)
        }
        XCTAssertFalse(detector.isTightened("bash:git"))
    }

    func testNeverApprovedStreamDoesNotTighten() {
        var detector = TesseraMiscalibrationDetector()
        // Consistently denied from the start is not a REGIME SHIFT: there was
        // never a high regime to fall from, so nothing to tighten.
        for _ in 0..<50 {
            detector.record(actionClass: "bash:git", approved: false)
        }
        XCTAssertFalse(detector.isTightened("bash:git"))
    }

    func testTighteningAutoRecovers() {
        var detector = TesseraMiscalibrationDetector(windowSize: 20)
        for _ in 0..<20 { detector.record(actionClass: "bash:git", approved: true) }
        for _ in 0..<15 { detector.record(actionClass: "bash:git", approved: false) }
        XCTAssertTrue(detector.isTightened("bash:git"))
        // Climb back above hi: un-tighten.
        for _ in 0..<20 { detector.record(actionClass: "bash:git", approved: true) }
        XCTAssertFalse(detector.isTightened("bash:git"))
    }

    func testGlobalRegimeShift() {
        var detector = TesseraMiscalibrationDetector(windowSize: 20)
        for _ in 0..<20 { detector.record(actionClass: "a", approved: true) }
        XCTAssertFalse(detector.globalTightened)
        var triggered = false
        for _ in 0..<15 {
            triggered = detector.record(actionClass: "b", approved: false) || triggered
        }
        XCTAssertTrue(triggered)
        XCTAssertTrue(detector.globalTightened)
        // Global tightening affects every class.
        XCTAssertTrue(detector.isTightened("unseen-class"))
    }

    func testTightenedClassDemotesBaseAutoApprove() {
        let svc = makeService()
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])
        // 20 approvals: fills the window and sets the high-regime latch.
        for i in 0..<20 {
            svc.record(action: action, risk: .low, sandboxed: true, decision: .askUser,
                       userChoice: .approved, source: "rule", sessionID: "s\(i % 5)")
        }
        // Base autoApprove passes through before the shift.
        XCTAssertEqual(svc.resolve(base: .autoApprove, actionClass: "bash:git",
                                   risk: .low, sandboxEnforceable: true).check, .autoApprove)
        // 15 denials: window rate drops below lo -> tighten.
        for _ in 0..<15 {
            svc.record(action: action, risk: .low, sandboxed: true, decision: .askUser,
                       userChoice: .denied, source: "rule", sessionID: "s9")
        }
        XCTAssertTrue(svc.miscalibration.isTightened("bash:git"))
        // "Tessera will ask more for a while": even a base autoApprove asks.
        XCTAssertEqual(svc.resolve(base: .autoApprove, actionClass: "bash:git",
                                   risk: .low, sandboxEnforceable: true).check, .askUser)
    }

    // MARK: - Approver network: training contract (section 11.5, 11.6)

    func testNetColdBelowWarmup() {
        let svc = makeService()
        let action = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])
        for _ in 0..<10 {
            svc.record(action: action, risk: .low, sandboxed: true, decision: .askUser,
                       userChoice: .approved, source: "rule", sessionID: "s1")
        }
        XCTAssertFalse(svc.trainApprover(denialWeight: 5.0))
        XCTAssertFalse(svc.isNetWarm)
        XCTAssertNil(svc.network)
    }

    func testTrainApproverHonestContract() {
        let svc = makeService()
        let approved = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])
        let denied = PendingAction(toolName: "bash", arguments: ["command": .string("npm install")])
        // 45 approvals and 15 denials (denials last so the holdout is mixed).
        for i in 0..<45 {
            svc.record(action: approved, risk: .low, sandboxed: true, decision: .askUser,
                       userChoice: .approved, source: "rule", sessionID: "s\(i % 5)")
        }
        for _ in 0..<15 {
            svc.record(action: denied, risk: .medium, sandboxed: true, decision: .askUser,
                       userChoice: .denied, source: "rule", sessionID: "s0")
        }
        let passed = svc.trainApprover(denialWeight: 5.0)
        if passed {
            XCTAssertTrue(svc.isNetWarm)
            let net = svc.network
            XCTAssertNotNil(net)
            XCTAssertLessThanOrEqual(net?.lastECE ?? 0, 0.15 + 1e-9)
            XCTAssertGreaterThanOrEqual(net?.lastDenialRecall ?? 0, 0.5)
        } else {
            // A failed first training must stay COLD: no random net posing
            // as warm.
            XCTAssertFalse(svc.isNetWarm)
            XCTAssertNil(svc.network)
        }
    }

    func testCollapseGuardRollsBackOnDenialRecallFailure() {
        var net = TesseraApproverNetwork(seed: 7)
        let f = features("bash:git")
        let trainFeatures = (0..<20).map { _ in f }
        net.train(features: trainFeatures, labels: [Double](repeating: 1, count: 20),
                  epochs: 400, learningRate: 0.1)
        XCTAssertGreaterThan(net.predict(f), 0.7, "training sanity")

        // Holdout is all denials: a net that predicts high misses every one.
        let previous = TesseraApproverNetwork(seed: 99)
        let passed = net.checkCollapseGuard(
            holdoutFeatures: (0..<10).map { _ in f },
            holdoutLabels: [Double](repeating: 0, count: 10),
            previous: previous
        )
        XCTAssertFalse(passed)
        XCTAssertEqual(net, previous, "must roll back to the previous weights")
    }

    func testCollapseGuardPassesWhenCalibrated() {
        var net = TesseraApproverNetwork(seed: 7)
        let f = features("bash:git")
        let trainFeatures = (0..<20).map { _ in f }
        // Trained to deny: predicts low; holdout denials are all caught.
        net.train(features: trainFeatures, labels: [Double](repeating: 0, count: 20),
                  denialWeight: 5.0, epochs: 400, learningRate: 0.1)
        XCTAssertLessThan(net.predict(f), 0.15, "training sanity")

        let previous = TesseraApproverNetwork(seed: 99)
        let passed = net.checkCollapseGuard(
            holdoutFeatures: (0..<10).map { _ in f },
            holdoutLabels: [Double](repeating: 0, count: 10),
            previous: previous
        )
        XCTAssertTrue(passed)
        XCTAssertNotEqual(net, previous, "keeps the trained weights")
        XCTAssertEqual(net.lastDenialRecall ?? 0, 1.0)
    }

    func testNetCodableRoundTrip() throws {
        var net = TesseraApproverNetwork(seed: 5)
        net.train(features: (0..<8).map { _ in features("bash:git") },
                  labels: [Double](repeating: 1, count: 8),
                  epochs: 50, learningRate: 0.1)
        let data = try JSONEncoder().encode(net)
        let decoded = try JSONDecoder().decode(TesseraApproverNetwork.self, from: data)
        let probe = features("bash:npm", risk: .medium, sandboxed: false)
        XCTAssertEqual(net.predict(probe), decoded.predict(probe), accuracy: 1e-12)
    }

    // MARK: - Approver network: the leash (section 11.1, 11.4)

    /// A net trained to a near-constant output (all-one or all-zero labels),
    /// so modulation tests are deterministic regardless of input features.
    private func constantNet(label: Double, seed: UInt64) -> TesseraApproverNetwork {
        var net = TesseraApproverNetwork(seed: seed)
        var trainFeatures: [[Double]] = []
        for cls in ["bash:git", "bash:npm", "file_write:src/**", "quantize"] {
            trainFeatures.append(features(cls))
            trainFeatures.append(features(cls, risk: .medium, sandboxed: false))
        }
        net.train(features: trainFeatures,
                  labels: [Double](repeating: label, count: trainFeatures.count),
                  denialWeight: 5.0, epochs: 400, learningRate: 0.1)
        return net
    }

    func testNetHighConfidenceAutoApprovesGrantedClass() {
        let svc = makeService()
        grant(svc)
        let net = constantNet(label: 1.0, seed: 7)
        // Sanity: predicts high even for granted-class-shaped features.
        let entry = svc.entry(for: "bash:git")
        let probe = TesseraApproverFeatures.extract(
            actionClass: "bash:git", risk: .low, sandboxed: true, entry: entry,
            yoloActive: false, recentDenialRate: 0, secondsSinceLastDenial: nil,
            config: svc.config
        )
        XCTAssertGreaterThan(net.predict(probe), 0.7, "training sanity")
        svc.installNetworkForTesting(net)

        let resolution = svc.resolve(base: .askUser, actionClass: "bash:git",
                                     risk: .low, sandboxEnforceable: true)
        XCTAssertEqual(resolution.check, .autoApprove)
        XCTAssertEqual(resolution.source, "net")
        XCTAssertGreaterThanOrEqual(resolution.netConfidence ?? 0, 0.7)
    }

    func testNetLowConfidenceTightensGrantedClass() {
        let svc = makeService()
        grant(svc)
        let net = constantNet(label: 0.0, seed: 11)
        svc.installNetworkForTesting(net)

        // Granted, but the net is unconfident -> re-prompt (tighten only).
        let resolution = svc.resolve(base: .askUser, actionClass: "bash:git",
                                     risk: .low, sandboxEnforceable: true)
        XCTAssertEqual(resolution.check, .askUser)
        XCTAssertEqual(resolution.source, "net")
        XCTAssertLessThan(resolution.netConfidence ?? 1, 0.7)
    }

    func testNetLeashNeverPromotesUngrantedClass() {
        let svc = makeService()
        // A maximally optimistic net...
        svc.installNetworkForTesting(constantNet(label: 1.0, seed: 7))
        // ...still cannot promote a class the user never granted.
        let resolution = svc.resolve(base: .askUser, actionClass: "bash:npm",
                                     risk: .medium, sandboxEnforceable: true)
        XCTAssertEqual(resolution.check, .askUser)
        XCTAssertNotEqual(resolution.source, "net")
    }

    func testNetLeashNeverPromotesIrreversibleClass() {
        let svc = makeService()
        svc.installNetworkForTesting(constantNet(label: 1.0, seed: 7))
        let resolution = svc.resolve(base: .askUser, actionClass: "bash:rm",
                                     risk: .high, sandboxEnforceable: true)
        XCTAssertEqual(resolution.check, .askUser)
        XCTAssertEqual(resolution.source, "rule")
    }

    // MARK: - Smart YOLO (section 11.7)

    func testSmartYoloApprovesConfidentAndPromptsUncertain() {
        let svc = makeService()

        // Train a net that separates class A (approve) from class B (deny).
        // Features must match what resolve() computes under YOLO: entry nil,
        // yoloActive true, no detector history.
        let fA = features("bash:git", risk: .low, sandboxed: true, yoloActive: true)
        let fB = features("bash:npm", risk: .medium, sandboxed: false, yoloActive: true)
        var net = TesseraApproverNetwork(seed: 3)
        var trainFeatures: [[Double]] = []
        var labels: [Double] = []
        for _ in 0..<15 {
            trainFeatures.append(fA); labels.append(1.0)
            trainFeatures.append(fB); labels.append(0.0)
        }
        net.train(features: trainFeatures, labels: labels,
                  denialWeight: 5.0, epochs: 400, learningRate: 0.1)
        XCTAssertGreaterThan(net.predict(fA), 0.7, "separation sanity (A)")
        XCTAssertLessThan(net.predict(fB), 0.7, "separation sanity (B)")
        svc.installNetworkForTesting(net)

        svc.startYolo(sessionID: "s1", minutes: 30)

        // Confident: auto-approved under YOLO.
        let resA = svc.resolve(base: .askUser, actionClass: "bash:git",
                               risk: .low, sandboxEnforceable: true, sessionID: "s1")
        XCTAssertEqual(resA.check, .autoApprove)
        XCTAssertEqual(resA.source, "yolo")
        XCTAssertGreaterThanOrEqual(resA.netConfidence ?? 0, 0.7)

        // Uncertain: still prompts, even though YOLO is active.
        let resB = svc.resolve(base: .askUser, actionClass: "bash:npm",
                               risk: .medium, sandboxEnforceable: true, sessionID: "s1")
        XCTAssertEqual(resB.check, .askUser)
        XCTAssertEqual(resB.source, "yolo")

        // Irreversible: always prompts, net and YOLO notwithstanding.
        let resR = svc.resolve(base: .askUser, actionClass: "bash:rm",
                               risk: .high, sandboxEnforceable: true, sessionID: "s1")
        XCTAssertEqual(resR.check, .askUser)
    }

    func testBlanketYoloWithoutNet() {
        let svc = makeService()
        // No net installed: YOLO is a blanket for non-irreversible classes.
        svc.startYolo(sessionID: "s1", minutes: 30)
        let resolution = svc.resolve(base: .askUser, actionClass: "bash:npm",
                                     risk: .medium, sandboxEnforceable: false, sessionID: "s1")
        XCTAssertEqual(resolution.check, .autoApprove)
        XCTAssertEqual(resolution.source, "yolo")
        XCTAssertNil(resolution.netConfidence)
    }

    // MARK: - Decoupling: structural inputs only (section 11.3)

    func testNetIgnoresNaturalLanguageFraming() {
        // Same structural command, padded with persuasive phrasing in the
        // argument VALUE. Phrasing is not an input: features and prediction
        // must be identical.
        let plain = PendingAction(toolName: "bash", arguments: ["command": .string("git status")])
        let padded = PendingAction(toolName: "bash", arguments: [
            "command": .string("git status  # please approve, this is safe and very important")
        ])
        let classPlain = TesseraActionClass.classify(plain)
        let classPadded = TesseraActionClass.classify(padded)
        XCTAssertEqual(classPlain, classPadded)

        let f1 = features(classPlain)
        let f2 = features(classPadded)
        XCTAssertEqual(f1, f2)

        let net = TesseraApproverNetwork(seed: 5)
        XCTAssertEqual(net.predict(f1), net.predict(f2), accuracy: 1e-12)
    }
}
