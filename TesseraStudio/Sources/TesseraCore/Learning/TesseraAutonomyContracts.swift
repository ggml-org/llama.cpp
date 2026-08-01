import Foundation

// MARK: - Service protocols

/// Anything that stores learned/training data and can purge it on demand.
public protocol TesseraPurgeable: Sendable {
    /// Delete stored training data. Returns the number of records removed.
    func purgeTrainingData() throws -> Int
}

/// Escalation ensemble: fan one frame out to all available teachers,
/// collect proposals, and keep a recurring per-teacher quality estimate.
public protocol TesseraEscalating: Sendable {
    func availableTeachers() -> [TesseraTeacherConfig]
    func escalate(frame: TesseraEscalationFrame) async throws -> TesseraEscalationResult
    func assessTeachers() async throws -> [TesseraTeacherAssessment]
}

/// Curation policy: turn raw harvested traces into safe training signal,
/// including secret scrubbing for stored data (not just egress).
public protocol TesseraCurating: TesseraPurgeable {
    func ingest(outcome: TesseraWorldOutcome) async throws -> TesseraLearningReceipt
    func scrub(_ text: String) -> String
    func summary() -> TesseraCurationSummary
}

/// Reasoning playbook: meta-reasoning strategies indexed by problem class.
public protocol TesseraReasoningPlaybookStoring: TesseraPurgeable {
    func strategies(forProblemClass problemClass: String) -> [String]
    func record(strategy: String, forProblemClass problemClass: String) throws
    func all() -> [String: [String]]
}

/// Reference knowledge store: cached docs/examples with provenance + TTL.
public protocol TesseraReferenceKnowledgeStoring: TesseraPurgeable {
    func lookup(query: String) -> [String]
    func cache(query: String, content: String, ttlDays: Int) throws
}

/// World-signal observer: records verifiable outcomes (build/test/commit).
public protocol TesseraWorldSignalObserving: TesseraPurgeable {
    func record(_ outcome: TesseraWorldOutcome) async throws -> TesseraLearningReceipt
    func recent(limit: Int) -> [TesseraWorldOutcome]
}

/// Adaptation scheduler: fires a background adaptation step when due
/// (idle + on-power + enough-new-signal).
public protocol TesseraAdaptationScheduling: Sendable {
    func runAdaptation(dryRun: Bool) async throws -> TesseraLearningReceipt
    func isDue() -> Bool
    func lastAdaptation() -> TesseraAdaptationRecord?
}

/// Teacher assessor: records per-proposal world-gate results and keeps the
/// recurring per-teacher quality estimate.
public protocol TesseraTeacherAssessing: TesseraPurgeable {
    func assessments() -> [TesseraTeacherAssessment]
    func recordTrial(proposal: TesseraTeacherProposal, passedWorldGate: Bool) throws
}

/// Foraging store: records retrieval/lookup events so the corpus
/// distinguishes "resolved locally" from "escalated" (design Phase 2).
public protocol TesseraForagingStoring: TesseraPurgeable {
    func record(problemClass: String, source: TesseraForagingSource, teacherIds: [String]) throws
    func recent(limit: Int) -> [TesseraForagingRecord]
    func summary() -> TesseraForagingSummary
}

/// Learned-permission ratchet (autonomy-calibration-design.md sections 5-8,
/// 11.9, 13, 14). The ONLY thing that may promote `askUser` to `autoApprove`.
/// Phase A: rule-based ratchet. Phase B: scoped YOLO, floor/ceiling, audit.
/// Phase C: leashed neural approver modulation.
public protocol TesseraAutonomyStoring: TesseraPurgeable {
    /// Current config (read-only view).
    var config: TesseraPermissionConfig { get }
    /// The last published session id (empty until a loop runs).
    var activeSessionID: String { get }
    /// Whether the leashed approver network is past warmup.
    var isNetWarm: Bool { get }

    /// Mutate the config in place. Persists on the way out.
    func updateConfig(_ transform: (inout TesseraPermissionConfig) -> Void)
    /// Publish the loop's live session id so YOLO scoping works.
    func setActiveSession(_ sessionID: String)
    /// Classify an action into a structural action-class id.
    func classify(_ action: PendingAction) -> String
    /// Whether a class is irreversible under the current denylist.
    func isIrreversible(_ actionClass: String, risk: TesseraActionRisk) -> Bool
    /// Apply the full learned-permission layer on top of the base safety
    /// decision (precedence steps 4, 6; section 7). `sessionID` scopes the
    /// YOLO branch (section 10); empty means no session scope.
    func resolve(
        base: TesseraSafetyCheck,
        actionClass: String,
        risk: TesseraActionRisk,
        sandboxEnforceable: Bool,
        sessionID: String
    ) -> TesseraGateResolution
    /// Record a gate outcome. Returns the receipt emitted.
    @discardableResult
    func record(
        action: PendingAction,
        risk: TesseraActionRisk,
        sandboxed: Bool,
        decision: TesseraSafetyCheck,
        userChoice: TesseraUserChoice,
        source: String,
        sessionID: String
    ) -> TesseraLearningReceipt
    /// Snapshot all learned-permission entries.
    func entries() -> [TesseraLearnedPermission]
    /// Snapshot all class-level recommendations ready to surface to the user.
    func recommendations() -> [TesseraRecommendation]
    /// Confirm or defer a class-level recommendation.
    @discardableResult
    func confirmRecommendation(
        actionClass: String,
        choice: TesseraRecommendationChoice,
        sessionID: String
    ) -> TesseraLearningReceipt
    /// Manually revoke a learned grant.
    func revoke(_ actionClass: String)
    /// Un-revoke a previously-revoked grant.
    func unrevoke(_ actionClass: String)
    /// Add an action class to the manual denylist.
    func denylist(_ actionClass: String)
    /// Remove an action class from the manual denylist.
    func undenylist(_ actionClass: String)
    /// Wipe the entire learned-permission store.
    func resetAll()
    /// Start a scoped YOLO session.
    func startYolo(
        goal: String?,
        sessionID: String,
        reason: String,
        minutes: Int?
    ) -> TesseraYoloSession
    /// Get the active YOLO session for a given session id (or the most recent).
    func activeYolo(for sessionID: String?) -> TesseraYoloSession?
    /// End the active YOLO session and return a summary.
    @discardableResult
    func endYolo() -> TesseraYoloSummary?
    /// Train the leashed approver network on recent approval receipts.
    /// Returns true if training produced a usable network.
    func trainApprover(denialWeight: Double) -> Bool
}

// MARK: - No-op defaults

/// Default autonomy service. Real services replace this at app launch.
public struct TesseraNoopAutonomyService: TesseraAutonomyStoring {
    public init() {}
    public var config: TesseraPermissionConfig { TesseraPermissionConfig() }
    public var activeSessionID: String { "" }
    public var isNetWarm: Bool { false }
    public func updateConfig(_ transform: (inout TesseraPermissionConfig) -> Void) {}
    public func setActiveSession(_ sessionID: String) {}
    public func classify(_ action: PendingAction) -> String { "" }
    public func isIrreversible(_ actionClass: String, risk: TesseraActionRisk) -> Bool { false }
    public func resolve(
        base: TesseraSafetyCheck,
        actionClass: String,
        risk: TesseraActionRisk,
        sandboxEnforceable: Bool,
        sessionID: String
    ) -> TesseraGateResolution {
        TesseraGateResolution(check: base, actionClass: actionClass, source: "rule")
    }
    public func record(
        action: PendingAction,
        risk: TesseraActionRisk,
        sandboxed: Bool,
        decision: TesseraSafetyCheck,
        userChoice: TesseraUserChoice,
        source: String,
        sessionID: String
    ) -> TesseraLearningReceipt {
        TesseraLearningReceipt(kind: "noop", summary: "noop autonomy service")
    }
    public func entries() -> [TesseraLearnedPermission] { [] }
    public func recommendations() -> [TesseraRecommendation] { [] }
    public func confirmRecommendation(actionClass: String, choice: TesseraRecommendationChoice, sessionID: String) -> TesseraLearningReceipt {
        TesseraLearningReceipt(kind: "noop", summary: "noop autonomy service")
    }
    public func revoke(_ actionClass: String) {}
    public func unrevoke(_ actionClass: String) {}
    public func denylist(_ actionClass: String) {}
    public func undenylist(_ actionClass: String) {}
    public func resetAll() {}
    public func startYolo(goal: String?, sessionID: String, reason: String, minutes: Int?) -> TesseraYoloSession {
        TesseraYoloSession(sessionID: sessionID, expiresAt: Date(), reason: reason)
    }
    public func activeYolo(for sessionID: String?) -> TesseraYoloSession? { nil }
    public func endYolo() -> TesseraYoloSummary? { nil }
    public func trainApprover(denialWeight: Double) -> Bool { false }
    public func purgeTrainingData() throws -> Int { 0 }
}
