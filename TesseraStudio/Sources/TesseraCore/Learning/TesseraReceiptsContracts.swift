import Foundation

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