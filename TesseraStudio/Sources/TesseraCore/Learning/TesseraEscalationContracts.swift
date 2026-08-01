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