import Foundation

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