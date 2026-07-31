import Foundation

/// The latest scored capability eval on record: the per-axis pass/fail
/// tallies (the lossless input the adapt harness re-reduces) plus the score
/// vector they produced. Written by the `evaluate` capability_eval path when
/// real results are supplied; read by the adaptation scheduler as its honest
/// input and as the source of the guard baseline. A nil latest record means
/// "no capability eval on record" - the scheduler stops rather than invents.
public struct TesseraCapabilityEvalRecord: Codable, Sendable, Equatable {
    public var tallies: [String: TesseraAxisTally]   // keyed by Swift axis name
    public var score: TesseraCapabilityScore
    public var weightedSum: Double
    public var backend: String                        // "harness" | "swift"
    public var timestamp: Date

    public init(
        tallies: [String: TesseraAxisTally],
        score: TesseraCapabilityScore,
        weightedSum: Double,
        backend: String,
        timestamp: Date = Date()
    ) {
        self.tallies = tallies
        self.score = score
        self.weightedSum = weightedSum
        self.backend = backend
        self.timestamp = timestamp
    }
}

/// File-backed store for the single latest capability-eval record. Mirrors
/// TesseraEvalInstanceStore's locking pattern; the record is optional because
/// "nothing scored yet" is a meaningful, honest state.
public final class TesseraCapabilityEvalStore: @unchecked Sendable {
    private let store: TesseraLearningStore
    private let lock = NSLock()
    private static let file = "capability-eval-latest.json"

    public init() {
        self.store = TesseraLearningStore()
    }

    public func recordLatest(_ record: TesseraCapabilityEvalRecord) throws {
        lock.lock(); defer { lock.unlock() }
        try store.save(record, to: Self.file)
    }

    public func latest() -> TesseraCapabilityEvalRecord? {
        lock.lock(); defer { lock.unlock() }
        return store.load(TesseraCapabilityEvalRecord?.self, from: Self.file, default: nil)
    }

    public func purge() throws {
        lock.lock(); defer { lock.unlock() }
        try store.delete(Self.file)
    }
}
