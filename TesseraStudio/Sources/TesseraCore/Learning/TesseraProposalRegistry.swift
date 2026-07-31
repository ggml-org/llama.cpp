import Foundation

/// File-backed proposal -> teacher attribution map. Every proposal the
/// escalation service creates is registered here so a later world outcome can
/// be attributed to the teacher that actually produced it (this is what fixes
/// the old "unknown" bucket in record_outcome).
///
/// Not installed into TesseraLearningCenter (that type is spine); shared via
/// `shared`, matching the center's own singleton pattern.
public final class TesseraProposalRegistry: TesseraPurgeable, @unchecked Sendable {
    public static let shared = TesseraProposalRegistry()

    private let store: TesseraLearningStore
    private let lock = NSLock()
    private static let file = "proposals.json"

    private init() {
        self.store = TesseraLearningStore()
    }

    /// Record which teacher produced a proposal. Best-effort: a save failure
    /// must not sink the escalation that is registering the proposal.
    public func register(_ proposal: TesseraTeacherProposal) {
        lock.lock(); defer { lock.unlock() }
        var map = store.load([String: String].self, from: Self.file, default: [:])
        map[proposal.id] = proposal.teacherId
        try? store.save(map, to: Self.file)
    }

    /// Resolve the teacher that produced a proposal, or nil when unknown
    /// (e.g. a proposal recorded before the registry existed).
    public func teacherId(forProposalId proposalId: String) -> String? {
        lock.lock(); defer { lock.unlock() }
        return store.load([String: String].self, from: Self.file, default: [:])[proposalId]
    }

    public func purgeTrainingData() throws -> Int {
        lock.lock(); defer { lock.unlock() }
        let count = store.load([String: String].self, from: Self.file, default: [:]).count
        try store.delete(Self.file)
        return count
    }
}
