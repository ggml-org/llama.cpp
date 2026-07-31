import Foundation

/// File-backed reasoning playbook: meta-reasoning strategies indexed by
/// problem class (design 4.3). Append-only with per-class dedup; editable
/// and deletable via purge.
public final class TesseraReasoningPlaybookStore: TesseraReasoningPlaybookStoring, @unchecked Sendable {
    private let store: TesseraLearningStore
    private let lock = NSLock()
    private static let file = "playbook.json"

    public init() {
        self.store = TesseraLearningStore()
    }

    public func strategies(forProblemClass problemClass: String) -> [String] {
        lock.lock(); defer { lock.unlock() }
        return store.load([String: [String]].self, from: Self.file, default: [:])[problemClass] ?? []
    }

    public func record(strategy: String, forProblemClass problemClass: String) throws {
        lock.lock(); defer { lock.unlock() }
        var map = store.load([String: [String]].self, from: Self.file, default: [:])
        var list = map[problemClass] ?? []
        if !list.contains(strategy) {
            list.append(strategy)
        }
        map[problemClass] = list
        try store.save(map, to: Self.file)
    }

    public func all() -> [String: [String]] {
        lock.lock(); defer { lock.unlock() }
        return store.load([String: [String]].self, from: Self.file, default: [:])
    }

    public func purgeTrainingData() throws -> Int {
        lock.lock(); defer { lock.unlock() }
        let map = store.load([String: [String]].self, from: Self.file, default: [:])
        let count = map.values.reduce(0) { $0 + $1.count }
        try store.delete(Self.file)
        return count
    }
}
