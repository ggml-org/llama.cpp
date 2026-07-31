import Foundation

/// File-backed foraging-signal store (design Phase 2). Records where each
/// escalation frame or docs lookup was resolved - local playbook, local
/// reference, or remote teacher - so the corpus can be watched shifting from
/// "escalated" toward "resolved locally". Append-only; summary counts are
/// computed on read.
public final class TesseraForagingStore: TesseraForagingStoring, @unchecked Sendable {
    private let store: TesseraLearningStore
    private let lock = NSLock()
    private static let file = "foraging.json"

    public init() {
        self.store = TesseraLearningStore()
    }

    public func record(problemClass: String, source: TesseraForagingSource, teacherIds: [String]) throws {
        lock.lock(); defer { lock.unlock() }
        var records = store.load([TesseraForagingRecord].self, from: Self.file, default: [])
        records.append(TesseraForagingRecord(problemClass: problemClass, source: source, teacherIds: teacherIds))
        try store.save(records, to: Self.file)
    }

    public func recent(limit: Int) -> [TesseraForagingRecord] {
        lock.lock(); defer { lock.unlock() }
        let records = store.load([TesseraForagingRecord].self, from: Self.file, default: [])
        return limit > 0 ? Array(records.suffix(limit)) : records
    }

    public func summary() -> TesseraForagingSummary {
        lock.lock(); defer { lock.unlock() }
        var summary = TesseraForagingSummary()
        for record in store.load([TesseraForagingRecord].self, from: Self.file, default: []) {
            summary.total += 1
            switch record.source {
            case .localPlaybook:  summary.localPlaybook += 1
            case .localReference: summary.localReference += 1
            case .remote:         summary.remote += 1
            }
        }
        return summary
    }

    public func purgeTrainingData() throws -> Int {
        lock.lock(); defer { lock.unlock() }
        let count = store.load([TesseraForagingRecord].self, from: Self.file, default: []).count
        try store.delete(Self.file)
        return count
    }
}
