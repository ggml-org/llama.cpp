import Foundation

// MARK: - No-op default services

/// Default no-op escalation service. Real services replace this at app launch.
public struct TesseraNoopEscalationService: TesseraEscalating {
    public init() {}
    public func availableTeachers() -> [TesseraTeacherConfig] { [] }
    public func escalate(frame: TesseraEscalationFrame) async throws -> TesseraEscalationResult {
        throw TesseraLearningError.notConfigured("escalation service")
    }
    public func assessTeachers() async throws -> [TesseraTeacherAssessment] { [] }
}

public struct TesseraNoopCurationService: TesseraCurating {
    public init() {}
    public func ingest(outcome: TesseraWorldOutcome) async throws -> TesseraLearningReceipt {
        throw TesseraLearningError.notConfigured("curation service")
    }
    public func scrub(_ text: String) -> String { text }
    public func summary() -> TesseraCurationSummary { TesseraCurationSummary() }
    public func purgeTrainingData() throws -> Int { 0 }
}

public struct TesseraNoopPlaybookStore: TesseraReasoningPlaybookStoring {
    public init() {}
    public func strategies(forProblemClass problemClass: String) -> [String] { [] }
    public func record(strategy: String, forProblemClass problemClass: String) throws {}
    public func all() -> [String: [String]] { [:] }
    public func purgeTrainingData() throws -> Int { 0 }
}

public struct TesseraNoopReferenceStore: TesseraReferenceKnowledgeStoring {
    public init() {}
    public func lookup(query: String) -> [String] { [] }
    public func cache(query: String, content: String, ttlDays: Int) throws {}
    public func purgeTrainingData() throws -> Int { 0 }
}

public struct TesseraNoopWorldSignalObserver: TesseraWorldSignalObserving {
    public init() {}
    public func record(_ outcome: TesseraWorldOutcome) async throws -> TesseraLearningReceipt {
        throw TesseraLearningError.notConfigured("world-signal observer")
    }
    public func recent(limit: Int) -> [TesseraWorldOutcome] { [] }
    public func purgeTrainingData() throws -> Int { 0 }
}

public struct TesseraNoopAdaptationScheduler: TesseraAdaptationScheduling {
    public init() {}
    public func runAdaptation(dryRun: Bool) async throws -> TesseraLearningReceipt {
        throw TesseraLearningError.notConfigured("adaptation scheduler")
    }
    public func isDue() -> Bool { false }
    public func lastAdaptation() -> TesseraAdaptationRecord? { nil }
}

public struct TesseraNoopTeacherAssessor: TesseraTeacherAssessing {
    public init() {}
    public func assessments() -> [TesseraTeacherAssessment] { [] }
    public func recordTrial(proposal: TesseraTeacherProposal, passedWorldGate: Bool) throws {}
    public func purgeTrainingData() throws -> Int { 0 }
}

public struct TesseraNoopForagingStore: TesseraForagingStoring {
    public init() {}
    public func record(problemClass: String, source: TesseraForagingSource, teacherIds: [String]) throws {}
    public func recent(limit: Int) -> [TesseraForagingRecord] { [] }
    public func summary() -> TesseraForagingSummary { TesseraForagingSummary() }
    public func purgeTrainingData() throws -> Int { 0 }
}

// MARK: - Composition root

/// Service locator for the learning subsystem. Tools resolve their
/// dependencies through `shared`; concrete services are installed at app
/// launch. Until then, no-op defaults keep everything compiling and make
/// every learning tool report "not configured" instead of crashing.
public final class TesseraLearningCenter: @unchecked Sendable {
    public static let shared = TesseraLearningCenter()

    private let lock = NSLock()
    private var _escalation: any TesseraEscalating = TesseraNoopEscalationService()
    private var _curation: any TesseraCurating = TesseraNoopCurationService()
    private var _playbook: any TesseraReasoningPlaybookStoring = TesseraNoopPlaybookStore()
    private var _reference: any TesseraReferenceKnowledgeStoring = TesseraNoopReferenceStore()
    private var _worldSignals: any TesseraWorldSignalObserving = TesseraNoopWorldSignalObserver()
    private var _scheduler: any TesseraAdaptationScheduling = TesseraNoopAdaptationScheduler()
    private var _assessor: any TesseraTeacherAssessing = TesseraNoopTeacherAssessor()
    private var _foraging: any TesseraForagingStoring = TesseraNoopForagingStore()
    private var _headRouting: any TesseraHeadRouting = TesseraNoopHeadRouting()
    private var _autonomy: any TesseraAutonomyStoring = TesseraNoopAutonomyService()
    // Optional: nil until installDefaults wires the drafter trainer.
    private var _training: TesseraTrainingOrchestrator?

    private init() {}

    // Accessors

    public var escalation: any TesseraEscalating {
        lock.lock(); defer { lock.unlock() }; return _escalation
    }
    public var curation: any TesseraCurating {
        lock.lock(); defer { lock.unlock() }; return _curation
    }
    public var playbook: any TesseraReasoningPlaybookStoring {
        lock.lock(); defer { lock.unlock() }; return _playbook
    }
    public var reference: any TesseraReferenceKnowledgeStoring {
        lock.lock(); defer { lock.unlock() }; return _reference
    }
    public var worldSignals: any TesseraWorldSignalObserving {
        lock.lock(); defer { lock.unlock() }; return _worldSignals
    }
    public var scheduler: any TesseraAdaptationScheduling {
        lock.lock(); defer { lock.unlock() }; return _scheduler
    }
    public var assessor: any TesseraTeacherAssessing {
        lock.lock(); defer { lock.unlock() }; return _assessor
    }
    public var foraging: any TesseraForagingStoring {
        lock.lock(); defer { lock.unlock() }; return _foraging
    }
    public var headRouting: any TesseraHeadRouting {
        lock.lock(); defer { lock.unlock() }; return _headRouting
    }
    public var autonomy: any TesseraAutonomyStoring {
        lock.lock(); defer { lock.unlock() }; return _autonomy
    }
    public var training: TesseraTrainingOrchestrator? {
        lock.lock(); defer { lock.unlock() }; return _training
    }

    /// True once a real escalation service with at least one teacher is
    /// installed and egress is enabled.
    public var isConfigured: Bool {
        !escalation.availableTeachers().isEmpty
    }

    // Installation

    public func install(escalation: any TesseraEscalating) {
        lock.lock(); defer { lock.unlock() }; _escalation = escalation
    }
    public func install(curation: any TesseraCurating) {
        lock.lock(); defer { lock.unlock() }; _curation = curation
    }
    public func install(playbook: any TesseraReasoningPlaybookStoring) {
        lock.lock(); defer { lock.unlock() }; _playbook = playbook
    }
    public func install(reference: any TesseraReferenceKnowledgeStoring) {
        lock.lock(); defer { lock.unlock() }; _reference = reference
    }
    public func install(worldSignals: any TesseraWorldSignalObserving) {
        lock.lock(); defer { lock.unlock() }; _worldSignals = worldSignals
    }
    public func install(scheduler: any TesseraAdaptationScheduling) {
        lock.lock(); defer { lock.unlock() }; _scheduler = scheduler
    }
    public func install(assessor: any TesseraTeacherAssessing) {
        lock.lock(); defer { lock.unlock() }; _assessor = assessor
    }
    public func install(foraging: any TesseraForagingStoring) {
        lock.lock(); defer { lock.unlock() }; _foraging = foraging
    }
    public func install(headRouting: any TesseraHeadRouting) {
        lock.lock(); defer { lock.unlock() }; _headRouting = headRouting
    }
    public func install(autonomy: any TesseraAutonomyStoring) {
        lock.lock(); defer { lock.unlock() }; _autonomy = autonomy
    }
    public func install(training: TesseraTrainingOrchestrator) {
        lock.lock(); defer { lock.unlock() }; _training = training
    }

    /// Purge stored training data across every purgeable store. Returns a
    /// receipt summarizing what was removed.
    public func purgeAll() throws -> TesseraLearningReceipt {
        var removed = 0
        removed += (try? curation.purgeTrainingData()) ?? 0
        removed += (try? playbook.purgeTrainingData()) ?? 0
        removed += (try? reference.purgeTrainingData()) ?? 0
        removed += (try? worldSignals.purgeTrainingData()) ?? 0
        removed += (try? assessor.purgeTrainingData()) ?? 0
        removed += (try? foraging.purgeTrainingData()) ?? 0
        removed += (try? training?.traceStore.purgeTrainingData()) ?? 0
        return TesseraLearningReceipt(
            kind: "purge",
            summary: "Purged \(removed) learning record(s).",
            payload: ["removed": .number(Double(removed))]
        )
    }
}
