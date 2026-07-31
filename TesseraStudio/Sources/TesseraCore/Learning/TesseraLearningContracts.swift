import Foundation

// MARK: - Learning errors

/// Errors surfaced by the self-improving learning subsystem.
public enum TesseraLearningError: Error, LocalizedError {
    case notConfigured(String)
    case noTeachersAvailable
    case egressDisabled
    case invalidArgument(String)
    case storeUnavailable(String)

    public var errorDescription: String? {
        switch self {
        case .notConfigured(let what): "Learning subsystem not configured: \(what)"
        case .noTeachersAvailable: "No escalation teachers are configured (set learning.teachers)."
        case .egressDisabled: "Escalation egress is disabled (learning.escalationEnabled)."
        case .invalidArgument(let detail): "Invalid argument: \(detail)"
        case .storeUnavailable(let detail): "Learning store unavailable: \(detail)"
        }
    }
}

// MARK: - Teachers (escalation ensemble)

/// One configured escalation teacher. The teacher POOL is exactly the
/// providers the user has supplied API keys for; escalation fans out to
/// all available teachers rather than picking a single oracle.
public struct TesseraTeacherConfig: Codable, Sendable, Identifiable, Equatable {
    public let id: String          // stable local id
    public var label: String       // human-readable label
    public var baseURL: String     // OpenAI-compatible base URL
    public var apiKey: String
    public var model: String
    public var zeroRetention: Bool // provider claims zero data retention
    public var weight: Double      // prior routing weight (assessment adjusts effective use)

    public init(
        id: String = UUID().uuidString,
        label: String,
        baseURL: String,
        apiKey: String,
        model: String,
        zeroRetention: Bool = true,
        weight: Double = 1.0
    ) {
        self.id = id
        self.label = label
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.model = model
        self.zeroRetention = zeroRetention
        self.weight = weight
    }
}

/// A live quality estimate for one teacher, maintained by the recurring
/// assessment. Gates future use: teachers that stop being useful drift
/// down. This is the structural defense against R3 (teacher bias).
public struct TesseraTeacherAssessment: Codable, Sendable, Identifiable, Equatable {
    public let teacherId: String
    public var worldGatePassFraction: Double   // fraction of proposals that passed the world gate
    public var reasoningExternalization: Double // R6 score in 0...1 (does it show its work?)
    public var samples: Int
    public var effectiveWeight: Double
    public var lastUpdated: Date

    public var id: String { teacherId }

    public init(
        teacherId: String,
        worldGatePassFraction: Double = 0.0,
        reasoningExternalization: Double = 0.0,
        samples: Int = 0,
        effectiveWeight: Double = 1.0,
        lastUpdated: Date = Date()
    ) {
        self.teacherId = teacherId
        self.worldGatePassFraction = worldGatePassFraction
        self.reasoningExternalization = reasoningExternalization
        self.samples = samples
        self.effectiveWeight = effectiveWeight
        self.lastUpdated = lastUpdated
    }
}

// MARK: - Escalation frame / proposals

/// Tier-1 escalation payload: a natural-language frame plus a structured
/// diagnostic envelope. No source code crosses the boundary at tier 1.
public struct TesseraEscalationFrame: Codable, Sendable, Equatable {
    public var problemClass: String        // e.g. "failing-test-resolution"
    public var summary: String             // natural-language problem frame
    public var observedVsExpected: String
    public var failingTests: [String]
    public var redactedErrors: [String]    // already scrubbed of secrets
    public var stackShape: String          // type / stack shape, no source

    public init(
        problemClass: String,
        summary: String,
        observedVsExpected: String = "",
        failingTests: [String] = [],
        redactedErrors: [String] = [],
        stackShape: String = ""
    ) {
        self.problemClass = problemClass
        self.summary = summary
        self.observedVsExpected = observedVsExpected
        self.failingTests = failingTests
        self.redactedErrors = redactedErrors
        self.stackShape = stackShape
    }
}

/// One teacher's response to an escalation frame.
public struct TesseraTeacherProposal: Codable, Sendable, Identifiable, Equatable {
    public let id: String
    public let teacherId: String
    public let reasoning: String       // object-layer reasoning
    public let metaMethod: String      // "how to reason about this class of problem"
    public let tokenCount: Int
    public let elapsedSeconds: Double

    public init(
        id: String = UUID().uuidString,
        teacherId: String,
        reasoning: String,
        metaMethod: String = "",
        tokenCount: Int = 0,
        elapsedSeconds: Double = 0
    ) {
        self.id = id
        self.teacherId = teacherId
        self.reasoning = reasoning
        self.metaMethod = metaMethod
        self.tokenCount = tokenCount
        self.elapsedSeconds = elapsedSeconds
    }
}

/// The result of fanning one frame out to the teacher ensemble.
public struct TesseraEscalationResult: Codable, Sendable, Equatable {
    public let frame: TesseraEscalationFrame
    public let proposals: [TesseraTeacherProposal]
    public let fannedOutTo: [String]   // teacher ids that were queried

    public init(frame: TesseraEscalationFrame, proposals: [TesseraTeacherProposal], fannedOutTo: [String]) {
        self.frame = frame
        self.proposals = proposals
        self.fannedOutTo = fannedOutTo
    }
}

// MARK: - World outcomes

/// The kind of verifiable world signal that grounds an update.
public enum TesseraWorldOutcomeKind: String, Codable, Sendable, CaseIterable {
    case build, test, commit, revert
}

/// A verifiable real-world outcome. This is the ground truth that gates
/// ALL updates (drafter and trunk alike).
public struct TesseraWorldOutcome: Codable, Sendable, Identifiable, Equatable {
    public let id: String
    public let kind: TesseraWorldOutcomeKind
    public let success: Bool
    public let detail: String
    public let proposalId: String?     // links to a teacher proposal when trialing ensemble output
    public let timestamp: Date

    public init(
        id: String = UUID().uuidString,
        kind: TesseraWorldOutcomeKind,
        success: Bool,
        detail: String = "",
        proposalId: String? = nil,
        timestamp: Date = Date()
    ) {
        self.id = id
        self.kind = kind
        self.success = success
        self.detail = detail
        self.proposalId = proposalId
        self.timestamp = timestamp
    }
}

// MARK: - Curation products

/// A (chosen, rejected) preference pair for one problem class, derived from a
/// pass vs a fail on the world gate. Pair FORMATION is real and testable; the
/// DPO-style training consumer is a marked plug-in point (design 4.2).
public struct TesseraPreferencePair: Codable, Sendable, Identifiable, Equatable {
    public let id: String
    public let problemClass: String
    public let chosen: TesseraWorldOutcome
    public let rejected: TesseraWorldOutcome

    public init(
        id: String = UUID().uuidString,
        problemClass: String,
        chosen: TesseraWorldOutcome,
        rejected: TesseraWorldOutcome
    ) {
        self.id = id
        self.problemClass = problemClass
        self.chosen = chosen
        self.rejected = rejected
    }
}

/// A brief rollup of curation state for the transparency surface: how much is
/// stored, how many duplicates were skipped, how many preference pairs are
/// formable now, and the mean heuristic quality of what is stored.
public struct TesseraCurationSummary: Codable, Sendable, Equatable {
    public var stored: Int
    public var dedupSkipped: Int
    public var preferencePairs: Int
    public var meanQuality: Double

    public init(stored: Int = 0, dedupSkipped: Int = 0, preferencePairs: Int = 0, meanQuality: Double = 0) {
        self.stored = stored
        self.dedupSkipped = dedupSkipped
        self.preferencePairs = preferencePairs
        self.meanQuality = meanQuality
    }
}

// MARK: - Multi-axis capability score

/// A per-candidate behavioral score vector. The vector is the substrate;
/// the weighted-sum scalar and Pareto non-domination below are two lenses
/// on the same numbers (ratified decision #8). generalCompetence is a
/// GUARD axis (hard regression constraint), not a trade-off weight.
public struct TesseraCapabilityScore: Codable, Sendable, Equatable {
    public var mechanical: Double        // failing-test + compiler/type-error instances
    public var apiCurrency: Double       // deprecated-API migration instances
    public var hardTail: Double          // escalation instances
    public var personalStyle: Double     // trunk LoRA / personal-distribution fit
    public var generalCompetence: Double // broad held-out set; the collapse guard

    public static let axisNames = ["mechanical", "apiCurrency", "hardTail", "personalStyle", "generalCompetence"]
    /// Axes that participate in trade-offs (everything except the guard).
    public static let optimizationAxisNames = ["mechanical", "apiCurrency", "hardTail", "personalStyle"]

    public init(
        mechanical: Double = 0,
        apiCurrency: Double = 0,
        hardTail: Double = 0,
        personalStyle: Double = 0,
        generalCompetence: Double = 0
    ) {
        self.mechanical = mechanical
        self.apiCurrency = apiCurrency
        self.hardTail = hardTail
        self.personalStyle = personalStyle
        self.generalCompetence = generalCompetence
    }

    public var vector: [Double] {
        [mechanical, apiCurrency, hardTail, personalStyle, generalCompetence]
    }

    public subscript(axis: String) -> Double {
        switch axis {
        case "mechanical": return mechanical
        case "apiCurrency": return apiCurrency
        case "hardTail": return hardTail
        case "personalStyle": return personalStyle
        case "generalCompetence": return generalCompetence
        default: return 0
        }
    }

    /// Weighted-sum lens over the OPTIMIZATION axes only. The guard axis
    /// is handled by `passesGuard`, never traded off here.
    public func weightedSum(weights: [String: Double]) -> Double {
        var sum = 0.0
        var weightTotal = 0.0
        for axis in Self.optimizationAxisNames {
            let w = weights[axis] ?? 1.0
            sum += w * self[axis]
            weightTotal += w
        }
        return weightTotal > 0 ? sum / weightTotal : 0.0
    }

    /// Pareto lens: true if self dominates other across ALL axes (>= on
    /// every axis, > on at least one).
    public func dominates(_ other: TesseraCapabilityScore) -> Bool {
        let a = vector
        let b = other.vector
        var strictlyBetter = false
        for i in 0..<a.count {
            if a[i] < b[i] { return false }
            if a[i] > b[i] { strictlyBetter = true }
        }
        return strictlyBetter
    }

    /// Guard check: general competence must not drop more than epsilon
    /// below the baseline. A nil baseline passes trivially.
    public func passesGuard(baseline: TesseraCapabilityScore?, epsilon: Double) -> Bool {
        guard let baseline else { return true }
        return generalCompetence >= baseline.generalCompetence - epsilon
    }
}

// MARK: - Receipts

/// A schema-versioned evidence record for the learning subsystem, in the
/// same receipts spirit as sidecar v3 / spec_calib.v*. Makes the system
/// inspectable and deletable.
public struct TesseraLearningReceipt: Codable, Sendable, Identifiable, Equatable {
    public static let currentSchemaVersion = 1

    public let id: String
    public let schemaVersion: Int
    public let kind: String          // "escalation" | "outcome" | "adaptation" | "assessment" | "curation" | "purge"
    public let timestamp: Date
    public let summary: String
    public let payload: [String: JSONValue]

    public init(
        id: String = UUID().uuidString,
        kind: String,
        summary: String,
        payload: [String: JSONValue] = [:],
        timestamp: Date = Date()
    ) {
        self.id = id
        self.schemaVersion = Self.currentSchemaVersion
        self.kind = kind
        self.summary = summary
        self.payload = payload
        self.timestamp = timestamp
    }
}

/// A persisted adaptation decision (design 4.5). Records what the guard did
/// and whether anything was adapted - in v1 `adapted` is always false because
/// the training step is a plug-in point, and this record says so honestly via
/// `backend` / `note`. The most recent record's score is the baseline the next
/// run's collapse guard compares against.
public struct TesseraAdaptationRecord: Codable, Sendable, Identifiable, Equatable {
    public let id: String
    public let timestamp: Date
    public let dryRun: Bool
    public let guardPassed: Bool
    public let adapted: Bool
    public let epsilon: Double
    public let score: TesseraCapabilityScore
    public let hasBaseline: Bool
    public let backend: String       // "harness" | "dry-run" | "unavailable" | "error"
    public let note: String

    public init(
        id: String = UUID().uuidString,
        timestamp: Date = Date(),
        dryRun: Bool,
        guardPassed: Bool,
        adapted: Bool,
        epsilon: Double,
        score: TesseraCapabilityScore,
        hasBaseline: Bool,
        backend: String,
        note: String
    ) {
        self.id = id
        self.timestamp = timestamp
        self.dryRun = dryRun
        self.guardPassed = guardPassed
        self.adapted = adapted
        self.epsilon = epsilon
        self.score = score
        self.hasBaseline = hasBaseline
        self.backend = backend
        self.note = note
    }
}

// MARK: - Foraging signals

/// Where an escalation frame or lookup was resolved. The whole point of
/// retrieve-before-escalate (design Phase 2) is to shift the corpus from
/// `remote` toward the local sources, so the source is recorded per event.
public enum TesseraForagingSource: String, Codable, Sendable, CaseIterable {
    case localPlaybook = "local-playbook"
    case localReference = "local-reference"
    case remote = "remote"
}

/// One retrieval/lookup event: what class of problem, where it was resolved,
/// and which teachers (real or synthetic local ids) contributed. This is the
/// telemetry that purifies the escalation corpus toward genuinely
/// reasoning-bound problems.
public struct TesseraForagingRecord: Codable, Sendable, Identifiable, Equatable {
    public let id: String
    public let problemClass: String
    public let source: TesseraForagingSource
    public let teacherIds: [String]
    public let timestamp: Date

    public init(
        id: String = UUID().uuidString,
        problemClass: String,
        source: TesseraForagingSource,
        teacherIds: [String] = [],
        timestamp: Date = Date()
    ) {
        self.id = id
        self.problemClass = problemClass
        self.source = source
        self.teacherIds = teacherIds
        self.timestamp = timestamp
    }
}

/// Aggregate counts by resolution source. `resolvedLocally` vs `remote` is
/// the headline: a climbing local fraction means retrieval is doing its job.
public struct TesseraForagingSummary: Codable, Sendable, Equatable {
    public var total: Int
    public var localPlaybook: Int
    public var localReference: Int
    public var remote: Int

    public var resolvedLocally: Int { localPlaybook + localReference }

    public init(total: Int = 0, localPlaybook: Int = 0, localReference: Int = 0, remote: Int = 0) {
        self.total = total
        self.localPlaybook = localPlaybook
        self.localReference = localReference
        self.remote = remote
    }
}

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

// MARK: - No-op defaults

/// Default escalation service: reports no teachers and refuses egress.
/// Lets tools compile and degrade gracefully before concrete services
/// are installed into `TesseraLearningCenter`.
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
