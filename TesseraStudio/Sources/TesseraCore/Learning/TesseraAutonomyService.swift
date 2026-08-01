import Foundation

// Data model types (TesseraUserChoice, TesseraRecommendationChoice, AutonomyCeiling,
// TesseraLearnedPermission, TesseraPermissionConfig, TesseraLearnedPermissionStore,
// TesseraRecommendation, TesseraGateResolution, TesseraYoloSession, TesseraYoloSummary)
// live in TesseraAutonomyDataModel.swift.

// MARK: - Autonomy service (sections 5-14)

/// The learned-permission ratchet and its supporting machinery. File-backed,
/// lock-guarded, purgeable.
///
/// Phase A: rule-based ratchet, irreversible guard, receipts, recommendations.
/// Phase B: scoped YOLO, dispositional floor/ceiling, breaker suspension.
/// Phase C: leashed neural approver (confidence modulation, smart YOLO),
///          miscalibration regime-shift detection.
///
/// The ratchet is the ONLY thing that may promote `askUser` to `autoApprove`.
/// The net modulates within the ratchet's envelope; it never grants.
public final class TesseraAutonomyService: TesseraAutonomyStoring, @unchecked Sendable {

    // MARK: Storage

    private let lock = NSLock()
    private let store: TesseraLearningStore
    private var state: TesseraLearnedPermissionStore

    private static let storeFile = "learned-permissions.json"
    private static let receiptsFile = "approval-receipts.json"
    private static let networkFile = "approver-network.json"
    /// Cap the receipt log so the file does not grow without bound.
    private static let maxReceipts = 10_000

    // MARK: Phase B state (in-memory, session-scoped)

    /// Active YOLO session (nil when inactive). Never persists across sessions.
    private var yoloSession: TesseraYoloSession?
    /// Classes seen during the current YOLO session (for the summary).
    private var yoloClasses: Set<String> = []
    /// Denials during the current YOLO session.
    private var yoloDenials = 0

    // MARK: Phase C state

    /// The leashed approver network (section 11). Nil until warmup completes.
    private var approverNetwork: TesseraApproverNetwork?
    /// Previous network weights for collapse-guard rollback (section 11.6).
    private var previousNetwork: TesseraApproverNetwork?
    /// Miscalibration regime-shift detector (section 12).
    private var miscalibrationDetector = TesseraMiscalibrationDetector()
    /// Receipts processed by the last training run (for incremental training).
    private var lastTrainedReceiptCount = 0
    /// Minimum approval receipts before the net activates (section 11.5).
    public static let warmupThreshold = 50
    /// Net confidence threshold for auto-approve modulation.
    public static let confidenceThreshold = 0.7

    public init() {
        self.store = TesseraLearningStore()
        self.state = store.load(
            TesseraLearnedPermissionStore.self,
            from: Self.storeFile,
            default: TesseraLearnedPermissionStore()
        )
        self.approverNetwork = store.load(
            TesseraApproverNetwork?.self,
            from: Self.networkFile,
            default: nil
        ) ?? nil
    }

    /// Test/internal initializer with a custom store directory.
    init(store: TesseraLearningStore) {
        self.store = store
        self.state = store.load(
            TesseraLearnedPermissionStore.self,
            from: Self.storeFile,
            default: TesseraLearnedPermissionStore()
        )
        self.approverNetwork = store.load(
            TesseraApproverNetwork?.self,
            from: Self.networkFile,
            default: nil
        ) ?? nil
    }

    // MARK: Config access

    public var config: TesseraPermissionConfig {
        lock.lock(); defer { lock.unlock() }
        return state.config
    }

    public func updateConfig(_ transform: (inout TesseraPermissionConfig) -> Void) {
        lock.lock(); defer { lock.unlock() }
        transform(&state.config)
        persistLocked()
    }

    // MARK: Classification

    /// Classify an action using the configured path-glob depth.
    public func classify(_ action: PendingAction) -> String {
        let depth = config.pathGlobDepth
        return TesseraActionClass.classify(action, pathGlobDepth: depth)
    }

    /// Whether a class is irreversible under the current denylist.
    public func isIrreversible(_ actionClass: String, risk: TesseraActionRisk) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return TesseraActionClass.isIrreversible(actionClass, risk: risk, denylist: state.denylist)
    }

    // MARK: Gate resolution (precedence, section 7 + Phase B/C)

    /// Apply the full learned-permission layer on top of the base safety
    /// decision. Steps 1-3 and 5 are handled by the base; this adds:
    /// - Step 4: irreversible guard (Phase A).
    /// - Step 6: ratchet promotion, modulated by the net (Phase A/C).
    /// - Step 6b: miscalibration tightening (Phase C).
    /// - Step 6c: dispositional floor blocks all learned promotion (Phase B).
    /// - YOLO: scoped auto-approve, smart-modulated by the net (Phase B/C).
    public func resolve(
        base: TesseraSafetyCheck,
        actionClass: String,
        risk: TesseraActionRisk,
        sandboxEnforceable: Bool,
        sessionID: String = ""
    ) -> TesseraGateResolution {
        lock.lock(); defer { lock.unlock() }

        // Step 4: irreversible guard. Never autoApprove.
        if TesseraActionClass.isIrreversible(actionClass, risk: risk, denylist: state.denylist) {
            let check: TesseraSafetyCheck = base == .reject ? .reject : .askUser
            return TesseraGateResolution(check: check, actionClass: actionClass, source: "rule")
        }

        // Step 5: base autoApprove passes through, unless miscalibration
        // tightened this class (section 12).
        if base == .autoApprove {
            if miscalibrationDetector.isTightened(actionClass) {
                return TesseraGateResolution(check: .askUser, actionClass: actionClass, source: "rule")
            }
            return TesseraGateResolution(check: .autoApprove, actionClass: actionClass, source: "rule")
        }

        guard base == .askUser else {
            return TesseraGateResolution(check: base, actionClass: actionClass, source: "rule")
        }

        // Step 6c: dispositional floor. Restricted floor blocks ALL learned
        // promotion (section 9). The user stays permanently needy.
        let floorBlocks = state.config.floor == .restricted

        // Step 6: learned promotion (ratchet + net modulation).
        if !floorBlocks,
           let entry = state.entries[actionClass],
           entry.granted, !entry.revoked,
           withinCeilingLocked(risk: risk, sandboxEnforceable: sandboxEnforceable),
           !miscalibrationDetector.isTightened(actionClass) {

            // Phase C: net modulation. The net can TIGHTEN a grant (re-prompt)
            // but never LOOSEN the rules. A low-confidence net re-prompts even
            // though the class is granted (section 11.4).
            if let net = approverNetwork {
                let conf = netConfidenceLocked(net, actionClass: actionClass, risk: risk,
                                              sandboxed: sandboxEnforceable, entry: entry)
                if conf >= Self.confidenceThreshold {
                    return TesseraGateResolution(check: .autoApprove, actionClass: actionClass,
                                                source: "net", netConfidence: conf)
                }
                // Low confidence: tighten the grant, re-prompt.
                return TesseraGateResolution(check: .askUser, actionClass: actionClass,
                                            source: "net", netConfidence: conf)
            }
            // No net (cold start): pure ratchet promotion.
            return TesseraGateResolution(check: .autoApprove, actionClass: actionClass, source: "ratchet")
        }

        // YOLO (section 10): scoped auto-approve for non-irreversible classes.
        // An explicit sessionID (from the live loop) wins; the published
        // session hint is the fallback for UI-driven checks.
        let yoloSessionID = sessionID.isEmpty ? currentSessionHint : sessionID
        if let yolo = activeYoloLocked(), yolo.sessionID == yoloSessionID {
            // Smart YOLO (section 11.7): if the net is warm, only auto-approve
            // actions the net is confident about. Uncertain ones still prompt.
            if let net = approverNetwork {
                let entry = state.entries[actionClass]
                let conf = netConfidenceLocked(net, actionClass: actionClass, risk: risk,
                                              sandboxed: sandboxEnforceable, entry: entry)
                if conf >= Self.confidenceThreshold {
                    return TesseraGateResolution(check: .autoApprove, actionClass: actionClass,
                                                source: "yolo", netConfidence: conf)
                }
                // Uncertain under YOLO: still prompt (the point of smart YOLO).
                return TesseraGateResolution(check: .askUser, actionClass: actionClass,
                                            source: "yolo", netConfidence: conf)
            }
            // Blanket YOLO (no net): approve everything non-irreversible.
            return TesseraGateResolution(check: .autoApprove, actionClass: actionClass, source: "yolo")
        }

        // Step 7: otherwise, the base decision stands.
        return TesseraGateResolution(check: base, actionClass: actionClass, source: "rule")
    }

    /// Compute the net's confidence for an action. Lock must be held.
    private func netConfidenceLocked(
        _ net: TesseraApproverNetwork,
        actionClass: String,
        risk: TesseraActionRisk,
        sandboxed: Bool,
        entry: TesseraLearnedPermission?
    ) -> Double {
        let features = TesseraApproverFeatures.extract(
            actionClass: actionClass,
            risk: risk,
            sandboxed: sandboxed,
            entry: entry,
            yoloActive: activeYoloLocked() != nil,
            recentDenialRate: 1.0 - (miscalibrationDetector.approvalRate(for: actionClass) ?? 1.0),
            secondsSinceLastDenial: nil,
            config: state.config
        )
        return net.predict(features)
    }

    /// The session id hint for YOLO scoping. Published by the agent loop via
    /// `setActiveSession`; `startYolo` also sets it. Lock-protected.
    private var currentSessionHint: String = ""

    /// Publish the loop's live session id (protocol surface).
    public func setActiveSession(_ sessionID: String) {
        lock.lock(); defer { lock.unlock() }
        currentSessionHint = sessionID
    }

    /// The last published session id (empty until a loop runs).
    public var activeSessionID: String {
        lock.lock(); defer { lock.unlock() }
        return currentSessionHint
    }

    private func withinCeilingLocked(risk: TesseraActionRisk, sandboxEnforceable: Bool) -> Bool {
        switch state.config.ceiling {
        case .containedLowRiskOnly:
            return sandboxEnforceable && risk == .low
        case .anyNonIrreversible:
            return true
        }
    }

    // MARK: Ratchet (section 6)

    /// Record an approval outcome, update the ratchet, feed the
    /// miscalibration detector, track YOLO stats, and emit an approval
    /// receipt (section 14). Returns the receipt.
    @discardableResult
    public func record(
        action: PendingAction,
        risk: TesseraActionRisk,
        sandboxed: Bool,
        decision: TesseraSafetyCheck,
        userChoice: TesseraUserChoice,
        source: String,
        sessionID: String
    ) -> TesseraLearningReceipt {
        lock.lock(); defer { lock.unlock() }

        let actionClass = TesseraActionClass.classify(action, pathGlobDepth: state.config.pathGlobDepth)
        let grantedBefore = state.entries[actionClass]?.granted ?? false
        let yoloActive = activeYoloLocked()?.sessionID == sessionID

        // Update ratchet state only on a real user choice.
        if userChoice == .approved || userChoice == .denied {
            recordLocked(
                actionClass: actionClass,
                approved: userChoice == .approved,
                sessionID: sessionID,
                risk: risk
            )
            // Feed the miscalibration detector (section 12).
            miscalibrationDetector.record(actionClass: actionClass, approved: userChoice == .approved)
        }

        // Track YOLO stats.
        if yoloActive, var yolo = yoloSession {
            yolo.actionCount += 1
            yoloSession = yolo
            yoloClasses.insert(actionClass)
            if userChoice == .denied { yoloDenials += 1 }
        }

        let grantedAfter = state.entries[actionClass]?.granted ?? false

        // Net confidence for the receipt (nil if cold).
        let netConf: Double?
        if let net = approverNetwork {
            let entry = state.entries[actionClass]
            netConf = netConfidenceLocked(net, actionClass: actionClass, risk: risk,
                                         sandboxed: sandboxed, entry: entry)
        } else {
            netConf = nil
        }

        var payload: [String: JSONValue] = [
            "actionClass": .string(actionClass),
            "toolName": .string(action.toolName),
            "risk": .string(risk.rawValue),
            "sandboxed": .bool(sandboxed),
            "decision": .string(decision.rawValue),
            "userChoice": .string(userChoice.rawValue),
            "source": .string(source),
            "grantedBefore": .bool(grantedBefore),
            "grantedAfter": .bool(grantedAfter),
            "yoloActive": .bool(yoloActive),
            "sessionID": .string(sessionID),
        ]
        if let netConf {
            payload["netConfidence"] = .number(netConf)
        }

        let receipt = TesseraLearningReceipt(
            kind: "approval",
            summary: "\(decision.rawValue) \(actionClass) (\(source))",
            payload: payload
        )
        appendReceiptLocked(receipt)
        persistLocked()
        return receipt
    }

    /// The asymmetric ratchet state machine (section 6). Lock must be held.
    private func recordLocked(actionClass: String, approved: Bool, sessionID: String, risk: TesseraActionRisk) {
        var entry = state.entries[actionClass] ?? TesseraLearnedPermission(
            actionClass: actionClass,
            irreversible: TesseraActionClass.isIrreversible(actionClass, risk: risk, denylist: state.denylist),
            riskAtFirstSeen: risk,
            lastSeen: Date()
        )

        // Freeze irreversible at first sight.
        entry.irreversible = entry.irreversible
            || TesseraActionClass.isIrreversible(actionClass, risk: risk, denylist: state.denylist)
        entry.lastSeen = Date()

        if approved {
            if sessionID != entry.lastSessionID {
                entry.distinctSessions += 1
                entry.lastSessionID = sessionID
            }
            entry.consecutiveApprovals += 1
            entry.totalApprovals += 1

            // Grant check: slow, multi-session, non-irreversible, non-revoked.
            if !entry.granted && !entry.revoked && !entry.irreversible
                && entry.consecutiveApprovals >= state.config.grantThresholdN
                && entry.distinctSessions >= state.config.sessionThresholdM
                && risk < .high {
                entry.granted = true
                entry.grantedAt = Date()
            }
        } else {
            // Asymmetric: one denial resets the consecutive run and revokes.
            entry.consecutiveApprovals = 0
            entry.totalDenials += 1
            if entry.granted {
                entry.granted = false
            }
        }

        state.entries[actionClass] = entry
    }

    // MARK: Recommendations (section 11.9, rule-based trigger)

    /// Classes worth recommending: not granted, not irreversible, not revoked,
    /// with a strong approval pattern. No ML required.
    public func recommendations() -> [TesseraRecommendation] {
        lock.lock(); defer { lock.unlock() }
        let floor = max(2, state.config.recommendationFloor)
        return state.entries.values
            .filter { entry in
                !entry.granted
                    && !entry.irreversible
                    && !entry.revoked
                    && entry.consecutiveApprovals >= floor
                    && entry.distinctSessions >= 1
            }
            .sorted { $0.consecutiveApprovals > $1.consecutiveApprovals }
            .map { TesseraRecommendation(
                actionClass: $0.actionClass,
                consecutiveApprovals: $0.consecutiveApprovals,
                distinctSessions: $0.distinctSessions
            ) }
    }

    /// Handle the user's response to a recommendation. A confirmation is a
    /// class-level label: stronger than accumulated approval. "Never" adds
    /// the class to the denylist (irreversible). Emits a receipt.
    @discardableResult
    public func confirmRecommendation(
        actionClass: String,
        choice: TesseraRecommendationChoice,
        sessionID: String
    ) -> TesseraLearningReceipt {
        lock.lock(); defer { lock.unlock() }

        switch choice {
        case .confirm:
            // Explicit user grant: stronger than accumulated.
            if var entry = state.entries[actionClass] {
                entry.granted = true
                entry.grantedAt = Date()
                state.entries[actionClass] = entry
            }
        case .notNow:
            break  // leave accumulating
        case .never:
            state.denylist.insert(actionClass)
            if var entry = state.entries[actionClass] {
                entry.irreversible = true
                entry.granted = false
                state.entries[actionClass] = entry
            }
        }

        let receipt = TesseraLearningReceipt(
            kind: "approval",
            summary: "recommendation \(choice.rawValue) \(actionClass)",
            payload: [
                "actionClass": .string(actionClass),
                "source": .string("recommendation"),
                "userChoice": .string(choice == .confirm ? "approved" : "denied"),
                "decision": .string(choice == .confirm ? "autoApprove" : "askUser"),
                "sessionID": .string(sessionID),
            ]
        )
        appendReceiptLocked(receipt)
        persistLocked()
        return receipt
    }

    // MARK: Audit and revocation (section 13)

    public func entries() -> [TesseraLearnedPermission] {
        lock.lock(); defer { lock.unlock() }
        return Array(state.entries.values).sorted { $0.actionClass < $1.actionClass }
    }

    public func entry(for actionClass: String) -> TesseraLearnedPermission? {
        lock.lock(); defer { lock.unlock() }
        return state.entries[actionClass]
    }

    /// Revoke a class: stops auto-approving, never re-grants until un-revoked.
    public func revoke(_ actionClass: String) {
        lock.lock(); defer { lock.unlock() }
        guard var entry = state.entries[actionClass] else { return }
        entry.revoked = true
        entry.revokedAt = Date()
        entry.granted = false
        state.entries[actionClass] = entry
        persistLocked()
    }

    /// Un-revoke: resumes accumulation from zero.
    public func unrevoke(_ actionClass: String) {
        lock.lock(); defer { lock.unlock() }
        guard var entry = state.entries[actionClass] else { return }
        entry.revoked = false
        entry.revokedAt = nil
        entry.consecutiveApprovals = 0
        state.entries[actionClass] = entry
        persistLocked()
    }

    /// Add a class to the manual denylist (irreversible, forever prompts).
    public func denylist(_ actionClass: String) {
        lock.lock(); defer { lock.unlock() }
        state.denylist.insert(actionClass)
        if var entry = state.entries[actionClass] {
            entry.irreversible = true
            entry.granted = false
            state.entries[actionClass] = entry
        }
        persistLocked()
    }

    /// Remove a class from the manual denylist.
    public func undenylist(_ actionClass: String) {
        lock.lock(); defer { lock.unlock() }
        state.denylist.remove(actionClass)
        persistLocked()
    }

    /// Clear every grant (entries kept for audit).
    public func resetAll() {
        lock.lock(); defer { lock.unlock() }
        for (key, var entry) in state.entries {
            entry.granted = false
            entry.grantedAt = nil
            entry.consecutiveApprovals = 0
            state.entries[key] = entry
        }
        persistLocked()
    }

    // MARK: Scoped YOLO (section 10)

    /// Start a scoped YOLO session. Bounded by time + goal + session.
    /// Irreversible classes still prompt even under YOLO.
    @discardableResult
    public func startYolo(
        goal: String? = nil,
        sessionID: String,
        reason: String = "",
        minutes: Int? = nil
    ) -> TesseraYoloSession {
        lock.lock(); defer { lock.unlock() }
        let mins = minutes ?? state.config.yoloDefaultMinutes
        let session = TesseraYoloSession(
            goal: goal,
            sessionID: sessionID,
            expiresAt: Date().addingTimeInterval(TimeInterval(mins * 60)),
            reason: reason
        )
        yoloSession = session
        yoloClasses = []
        yoloDenials = 0
        currentSessionHint = sessionID
        return session
    }

    /// The active YOLO session, or nil if inactive/expired/wrong session.
    public func activeYolo(for sessionID: String? = nil) -> TesseraYoloSession? {
        lock.lock(); defer { lock.unlock() }
        return activeYoloLocked(sessionID: sessionID)
    }

    private func activeYoloLocked(sessionID: String? = nil) -> TesseraYoloSession? {
        guard let yolo = yoloSession else { return nil }
        if yolo.isExpired { return nil }
        if let sid = sessionID, yolo.sessionID != sid { return nil }
        return yolo
    }

    /// End the YOLO session and return a summary of what ran autonomously.
    public func endYolo() -> TesseraYoloSummary? {
        lock.lock(); defer { lock.unlock() }
        guard let yolo = yoloSession else { return nil }
        let summary = TesseraYoloSummary(
            actionCount: yolo.actionCount,
            classes: yoloClasses,
            denials: yoloDenials,
            durationSeconds: Date().timeIntervalSince(yolo.startedAt)
        )
        yoloSession = nil
        yoloClasses = []
        yoloDenials = 0
        return summary
    }

    // MARK: Approver network training (section 11.5)

    /// Whether the net has enough data to activate.
    public var isNetWarm: Bool {
        lock.lock(); defer { lock.unlock() }
        return approverNetwork != nil
    }

    /// Train (or incrementally retrain) the approver network on the receipt
    /// stream. Runs in the idle window. No-ops until warmup threshold is met.
    /// Applies the collapse guard after training (section 11.6).
    @discardableResult
    public func trainApprover(denialWeight: Double = 5.0) -> Bool {
        lock.lock(); defer { lock.unlock() }

        let allReceipts = store.load([TesseraLearningReceipt].self, from: Self.receiptsFile, default: [])
        let approvalReceipts = allReceipts.filter { $0.kind == "approval" }

        // Cold start: not enough data.
        guard approvalReceipts.count >= Self.warmupThreshold else { return false }

        // Extract (features, label) pairs from receipts with a user choice.
        var features: [[Double]] = []
        var labels: [Double] = []
        for receipt in approvalReceipts {
            guard let choiceStr = receipt.payload["userChoice"]?.stringValue,
                  choiceStr == "approved" || choiceStr == "denied" else { continue }
            let actionClass = receipt.payload["actionClass"]?.stringValue ?? ""
            let riskStr = receipt.payload["risk"]?.stringValue ?? "medium"
            let risk = TesseraActionRisk(rawValue: riskStr) ?? .medium
            let sandboxed = receipt.payload["sandboxed"]?.boolValue ?? false
            let entry = state.entries[actionClass]

            let f = TesseraApproverFeatures.extract(
                actionClass: actionClass,
                risk: risk,
                sandboxed: sandboxed,
                entry: entry,
                yoloActive: receipt.payload["yoloActive"]?.boolValue ?? false,
                recentDenialRate: 0,
                secondsSinceLastDenial: nil,
                config: state.config
            )
            features.append(f)
            labels.append(choiceStr == "approved" ? 1.0 : 0.0)
        }

        guard features.count >= Self.warmupThreshold else { return false }

        // Hold out the most recent 20% as a calibration set (section 11.6).
        let holdoutK = max(10, features.count / 5)
        let trainFeatures = Array(features.dropLast(holdoutK))
        let trainLabels = Array(labels.dropLast(holdoutK))
        let holdoutFeatures = Array(features.suffix(holdoutK))
        let holdoutLabels = Array(labels.suffix(holdoutK))

        // Save previous weights for rollback.
        let previous = approverNetwork ?? TesseraApproverNetwork()
        previousNetwork = previous

        // Train (incremental: start from existing weights if available).
        var net = approverNetwork ?? TesseraApproverNetwork()
        net.train(features: trainFeatures, labels: trainLabels, denialWeight: denialWeight)

        // Collapse guard (section 11.6): check calibration on holdout.
        let passed = net.checkCollapseGuard(
            holdoutFeatures: holdoutFeatures,
            holdoutLabels: holdoutLabels,
            previous: previous
        )

        if passed {
            approverNetwork = net
            try? store.save(net, to: Self.networkFile)
        } else {
            // Rolled back. A never-trained previous net must stay COLD: a
            // random net masquerading as warm would modulate on noise.
            if previous.trainedOnReceipts > 0 {
                approverNetwork = previous
            } else {
                approverNetwork = nil
                try? store.save(TesseraApproverNetwork?.none, to: Self.networkFile)
            }
        }

        lastTrainedReceiptCount = approvalReceipts.count
        return passed
    }

    /// The current approver network (for inspection/testing).
    public var network: TesseraApproverNetwork? {
        lock.lock(); defer { lock.unlock() }
        return approverNetwork
    }

    /// Test hook: install a network directly, bypassing warmup and the
    /// collapse guard. Internal; visible to tests via @testable.
    func installNetworkForTesting(_ net: TesseraApproverNetwork) {
        lock.lock(); defer { lock.unlock() }
        approverNetwork = net
    }

    /// The miscalibration detector state (for inspection/testing).
    public var miscalibration: TesseraMiscalibrationDetector {
        lock.lock(); defer { lock.unlock() }
        return miscalibrationDetector
    }

    // MARK: Receipts

    /// Recent approval receipts, newest first.
    public func recentReceipts(limit: Int = 100) -> [TesseraLearningReceipt] {
        lock.lock(); defer { lock.unlock() }
        let all = store.load([TesseraLearningReceipt].self, from: Self.receiptsFile, default: [])
        return Array(all.suffix(limit)).reversed()
    }

    private func appendReceiptLocked(_ receipt: TesseraLearningReceipt) {
        var all = store.load([TesseraLearningReceipt].self, from: Self.receiptsFile, default: [])
        all.append(receipt)
        // Trim to cap.
        if all.count > Self.maxReceipts {
            all = Array(all.suffix(Self.maxReceipts))
        }
        try? store.save(all, to: Self.receiptsFile)
    }

    // MARK: Persistence

    private func persistLocked() {
        try? store.save(state, to: Self.storeFile)
    }

    // MARK: Purge (TesseraPurgeable)

    @discardableResult
    public func purgeTrainingData() throws -> Int {
        lock.lock(); defer { lock.unlock() }
        let count = state.entries.count
        state = TesseraLearnedPermissionStore()
        approverNetwork = nil
        previousNetwork = nil
        miscalibrationDetector.reset()
        yoloSession = nil
        yoloClasses = []
        yoloDenials = 0
        lastTrainedReceiptCount = 0
        persistLocked()
        try? store.delete(Self.receiptsFile)
        try? store.delete(Self.networkFile)
        return count
    }
}
