import Foundation

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