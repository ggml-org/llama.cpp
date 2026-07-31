import Foundation

/// File-backed reference knowledge store: cached docs/examples with a TTL
/// (design 4.3). Volatile by nature - entries expire and are never fused
/// into weights. Lookup is a case-insensitive substring match on the query.
public final class TesseraReferenceKnowledgeStore: TesseraReferenceKnowledgeStoring, @unchecked Sendable {
    private struct Entry: Codable, Sendable {
        let query: String
        let content: String
        let expiryDate: Date
    }

    private let store: TesseraLearningStore
    private let lock = NSLock()
    private static let file = "reference.json"

    public init() {
        self.store = TesseraLearningStore()
    }

    public func lookup(query: String) -> [String] {
        lock.lock(); defer { lock.unlock() }
        let entries = store.load([Entry].self, from: Self.file, default: [])
        let now = Date()
        let needle = query.lowercased()
        return entries
            .filter { entry in
                guard entry.expiryDate > now else { return false }
                let stored = entry.query.lowercased()
                return stored.contains(needle) || needle.contains(stored)
            }
            .map(\.content)
    }

    public func cache(query: String, content: String, ttlDays: Int) throws {
        lock.lock(); defer { lock.unlock() }
        var entries = store.load([Entry].self, from: Self.file, default: [])
        let expiry = Date().addingTimeInterval(TimeInterval(ttlDays) * 86_400)
        entries.append(Entry(query: query, content: content, expiryDate: expiry))
        try store.save(entries, to: Self.file)
    }

    public func purgeTrainingData() throws -> Int {
        lock.lock(); defer { lock.unlock() }
        let count = store.load([Entry].self, from: Self.file, default: []).count
        try store.delete(Self.file)
        return count
    }
}
