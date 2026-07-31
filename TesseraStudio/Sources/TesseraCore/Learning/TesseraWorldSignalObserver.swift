import Foundation

/// File-backed world-signal observer: a ring buffer of verifiable outcomes
/// capped at `capacity`, newest-last on disk (design 4.4). These outcomes
/// are the ground truth that gates every update.
public final class TesseraWorldSignalObserver: TesseraWorldSignalObserving, @unchecked Sendable {
    private let store: TesseraLearningStore
    private let lock = NSLock()
    private static let file = "world-signals.json"
    private static let capacity = 1000

    public init() {
        self.store = TesseraLearningStore()
    }

    public func record(_ outcome: TesseraWorldOutcome) async throws -> TesseraLearningReceipt {
        try recordSync(outcome)
    }

    // Synchronous so the NSLock is never held across an async suspension
    // point (NSLock is unavailable from asynchronous contexts).
    private func recordSync(_ outcome: TesseraWorldOutcome) throws -> TesseraLearningReceipt {
        lock.lock(); defer { lock.unlock() }
        var outcomes = store.load([TesseraWorldOutcome].self, from: Self.file, default: [])
        outcomes.append(outcome)
        if outcomes.count > Self.capacity {
            outcomes.removeFirst(outcomes.count - Self.capacity)
        }
        try store.save(outcomes, to: Self.file)

        return TesseraLearningReceipt(
            kind: "outcome",
            summary: "Recorded \(outcome.kind.rawValue) outcome (success=\(outcome.success)).",
            payload: [
                "outcomeId": .string(outcome.id),
                "kind": .string(outcome.kind.rawValue),
                "success": .bool(outcome.success),
            ]
        )
    }

    public func recent(limit: Int) -> [TesseraWorldOutcome] {
        lock.lock(); defer { lock.unlock() }
        let outcomes = store.load([TesseraWorldOutcome].self, from: Self.file, default: [])
        return Array(outcomes.reversed().prefix(limit))
    }

    public func purgeTrainingData() throws -> Int {
        lock.lock(); defer { lock.unlock() }
        let count = store.load([TesseraWorldOutcome].self, from: Self.file, default: []).count
        try store.delete(Self.file)
        return count
    }
}
