import Foundation

// MARK: - Autonomy data model (spec section 5)

/// The user's choice at an approval prompt.
public enum TesseraUserChoice: String, Codable, Sendable, CaseIterable {
    case approved
    case denied
    /// No prompt was shown (autoApprove or reject).
    case none
}

/// The user's response to a class-level recommendation (section 11.9).
public enum TesseraRecommendationChoice: String, Codable, Sendable, CaseIterable {
    /// Grant the class immediately (explicit user grant).
    case confirm
    /// Leave the class accumulating normally.
    case notNow
    /// Add the class to the manual denylist (irreversible).
    case never
}

/// Maximum autonomy learning may reach (section 9).
public enum AutonomyCeiling: String, Codable, Sendable, CaseIterable {
    /// Ratchet promotes only sandbox-contained low-risk classes. Conservative.
    case containedLowRiskOnly
    /// Ratchet promotes any granted non-irreversible class.
    case anyNonIrreversible
}

/// One learned-permission entry per action class (section 5).
public struct TesseraLearnedPermission: Codable, Sendable, Identifiable, Equatable {
    public var id: String { actionClass }
    public let actionClass: String
    /// Frozen at first sight (section 4). Once true, never false.
    public var irreversible: Bool
    public let riskAtFirstSeen: TesseraActionRisk
    /// Current consecutive-approval run; resets on any denial.
    public var consecutiveApprovals: Int
    /// Sessions with at least one approval.
    public var distinctSessions: Int
    public var totalApprovals: Int
    public var totalDenials: Int
    /// Crossed the grant threshold?
    public var granted: Bool
    public var grantedAt: Date?
    /// User manually revoked.
    public var revoked: Bool
    public var revokedAt: Date?
    public var lastSessionID: String?
    public var lastSeen: Date

    public init(
        actionClass: String,
        irreversible: Bool = false,
        riskAtFirstSeen: TesseraActionRisk = .medium,
        consecutiveApprovals: Int = 0,
        distinctSessions: Int = 0,
        totalApprovals: Int = 0,
        totalDenials: Int = 0,
        granted: Bool = false,
        grantedAt: Date? = nil,
        revoked: Bool = false,
        revokedAt: Date? = nil,
        lastSessionID: String? = nil,
        lastSeen: Date = Date()
    ) {
        self.actionClass = actionClass
        self.irreversible = irreversible
        self.riskAtFirstSeen = riskAtFirstSeen
        self.consecutiveApprovals = consecutiveApprovals
        self.distinctSessions = distinctSessions
        self.totalApprovals = totalApprovals
        self.totalDenials = totalDenials
        self.granted = granted
        self.grantedAt = grantedAt
        self.revoked = revoked
        self.revokedAt = revokedAt
        self.lastSessionID = lastSessionID
        self.lastSeen = lastSeen
    }
}

/// User-tunable autonomy configuration (section 5, section 16).
public struct TesseraPermissionConfig: Codable, Sendable, Equatable {
    /// Consecutive approvals required to grant (default 5).
    public var grantThresholdN: Int
    /// Distinct sessions required to grant (default 3).
    public var sessionThresholdM: Int
    /// Minimum approval requirement learning cannot reduce (section 9).
    public var floor: TesseraPermissionProfile
    /// Maximum autonomy learning cannot exceed (section 9).
    public var ceiling: AutonomyCeiling
    /// Path segments kept for path-glob classes (section 3).
    public var pathGlobDepth: Int
    /// Default scoped-YOLO duration in minutes (section 10).
    public var yoloDefaultMinutes: Int
    /// Consecutive approvals before a class is worth recommending (section 11.9).
    public var recommendationFloor: Int

    public init(
        grantThresholdN: Int = 5,
        sessionThresholdM: Int = 3,
        floor: TesseraPermissionProfile = .standard,
        ceiling: AutonomyCeiling = .containedLowRiskOnly,
        pathGlobDepth: Int = 1,
        yoloDefaultMinutes: Int = 30,
        recommendationFloor: Int = 3
    ) {
        self.grantThresholdN = grantThresholdN
        self.sessionThresholdM = sessionThresholdM
        self.floor = floor
        self.ceiling = ceiling
        self.pathGlobDepth = pathGlobDepth
        self.yoloDefaultMinutes = yoloDefaultMinutes
        // Default: max(2, N - 2).
        self.recommendationFloor = recommendationFloor
    }
}

/// The persisted learned-permission store (section 5).
struct TesseraLearnedPermissionStore: Codable, Sendable {
    static let currentSchemaVersion = 1
    var schemaVersion: Int
    var config: TesseraPermissionConfig
    var entries: [String: TesseraLearnedPermission]
    /// User's manual denylist (section 4, section 13).
    var denylist: Set<String>

    init(
        schemaVersion: Int = TesseraLearnedPermissionStore.currentSchemaVersion,
        config: TesseraPermissionConfig = TesseraPermissionConfig(),
        entries: [String: TesseraLearnedPermission] = [:],
        denylist: Set<String> = []
    ) {
        self.schemaVersion = schemaVersion
        self.config = config
        self.entries = entries
        self.denylist = denylist
    }
}

// MARK: - Recommendation (section 11.9)

/// A class-level recommendation surfaced to the user: "You keep approving
/// this; pre-approve the whole class?" A confirmation is a class-level
/// training label for the future approver network.
public struct TesseraRecommendation: Codable, Sendable, Identifiable, Equatable {
    public var id: String { actionClass }
    public let actionClass: String
    public let consecutiveApprovals: Int
    public let distinctSessions: Int
    public let message: String

    public init(actionClass: String, consecutiveApprovals: Int, distinctSessions: Int) {
        self.actionClass = actionClass
        self.consecutiveApprovals = consecutiveApprovals
        self.distinctSessions = distinctSessions
        let sessions = distinctSessions == 1 ? "1 session" : "\(distinctSessions) sessions"
        self.message = "You have approved `\(actionClass)` \(consecutiveApprovals) times across \(sessions). Pre-approve this class so Tessera stops asking?"
    }
}

// MARK: - Gate resolution

/// The resolved gate verdict plus provenance, returned by the autonomy-aware
/// gate check. `source` says WHY the verdict was reached.
public struct TesseraGateResolution: Sendable, Equatable {
    public let check: TesseraSafetyCheck
    public let actionClass: String
    /// "rule" | "ratchet" | "yolo" | "net" | "recommendation".
    public let source: String
    /// The approver network's confidence (nil until the net is warm).
    public let netConfidence: Double?

    public init(check: TesseraSafetyCheck, actionClass: String, source: String, netConfidence: Double? = nil) {
        self.check = check
        self.actionClass = actionClass
        self.source = source
        self.netConfidence = netConfidence
    }
}

// MARK: - Scoped YOLO (section 10)

/// A bounded, explicit, logged override. Not a settings toggle.
public struct TesseraYoloSession: Codable, Sendable, Identifiable, Equatable {
    public let id: String
    /// The task it is scoped to (user-stated).
    public let goal: String?
    /// Hard session bound.
    public let sessionID: String
    public let startedAt: Date
    /// Hard time bound.
    public let expiresAt: Date
    /// User's stated reason (audit).
    public let reason: String
    /// Actions auto-approved under it.
    public var actionCount: Int

    public init(
        id: String = UUID().uuidString,
        goal: String? = nil,
        sessionID: String,
        startedAt: Date = Date(),
        expiresAt: Date,
        reason: String = "",
        actionCount: Int = 0
    ) {
        self.id = id
        self.goal = goal
        self.sessionID = sessionID
        self.startedAt = startedAt
        self.expiresAt = expiresAt
        self.reason = reason
        self.actionCount = actionCount
    }

    public var isExpired: Bool { Date() >= expiresAt }
}

/// Summary emitted when a YOLO session ends (section 10).
public struct TesseraYoloSummary: Sendable, Equatable {
    public let actionCount: Int
    public let classes: Set<String>
    public let denials: Int
    public let durationSeconds: Double
}
