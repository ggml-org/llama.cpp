import Foundation

/// Concrete curation service: an append-only, file-backed record of curated
/// outcomes plus their receipts, and a secret scrubber for stored/egress
/// text (design 4.2). On top of ingest + scrub it provides the pure curation
/// analytics the loop needs next: content-hash dedup, a heuristic per-item
/// quality score, (chosen, rejected) preference-pair formation, and a
/// novelty/informativeness score to prioritize what to learn next.
///
/// Honesty ceiling: dedup, quality, pair FORMATION, and informativeness are
/// real and testable. Nothing here trains a model - the consumer of the
/// preference pairs is a marked plug-in point. Scrubbing is deliberately
/// conservative: it removes obvious secrets, it is not a guarantee that
/// nothing sensitive survives.
public final class TesseraCurationService: TesseraCurating, @unchecked Sendable {
    private let store: TesseraLearningStore
    private let lock = NSLock()
    private static let outcomesFile = "curation-outcomes.json"
    private static let receiptsFile = "curation-receipts.json"
    private static let dedupFile = "curation-dedup.json"

    /// Content-hash ring for dedup plus a lifetime count of skipped duplicates.
    private struct DedupState: Codable {
        var hashes: [String] = []
        var skipped: Int = 0
    }

    public init() {
        self.store = TesseraLearningStore()
    }

    public func ingest(outcome: TesseraWorldOutcome) async throws -> TesseraLearningReceipt {
        try ingestSync(outcome)
    }

    // Synchronous so the NSLock is never held across an async suspension
    // point (NSLock is unavailable from asynchronous contexts).
    private func ingestSync(_ outcome: TesseraWorldOutcome) throws -> TesseraLearningReceipt {
        lock.lock(); defer { lock.unlock() }

        // Content-hash dedup: a repeated outcome (same scrubbed kind + success
        // + detail) is recognized and skipped rather than stored twice. The
        // skip is recorded honestly in both the receipt and the lifetime count.
        var dedup = store.load(DedupState.self, from: Self.dedupFile, default: DedupState())
        let hash = contentHash(outcome)
        if dedup.hashes.contains(hash) {
            dedup.skipped += 1
            try store.save(dedup, to: Self.dedupFile)
            let receipt = TesseraLearningReceipt(
                kind: "curation",
                summary: "Skipped duplicate \(outcome.kind.rawValue) outcome (content already curated).",
                payload: [
                    "outcomeId": .string(outcome.id),
                    "kind": .string(outcome.kind.rawValue),
                    "deduplicated": .bool(true),
                ]
            )
            appendReceiptLocked(receipt)
            return receipt
        }
        dedup.hashes.append(hash)
        try store.save(dedup, to: Self.dedupFile)

        var outcomes = store.load([TesseraWorldOutcome].self, from: Self.outcomesFile, default: [])
        outcomes.append(outcome)
        try store.save(outcomes, to: Self.outcomesFile)

        let receipt = TesseraLearningReceipt(
            kind: "curation",
            summary: "Curated \(outcome.kind.rawValue) outcome (success=\(outcome.success), quality=\(String(format: "%.2f", qualityScore(outcome)))).",
            payload: [
                "outcomeId": .string(outcome.id),
                "kind": .string(outcome.kind.rawValue),
                "success": .bool(outcome.success),
                "quality": .number(qualityScore(outcome)),
                "contentHash": .string(hash),
            ]
        )
        appendReceiptLocked(receipt)
        return receipt
    }

    // Caller must hold `lock`.
    private func appendReceiptLocked(_ receipt: TesseraLearningReceipt) {
        var receipts = store.load([TesseraLearningReceipt].self, from: Self.receiptsFile, default: [])
        receipts.append(receipt)
        try? store.save(receipts, to: Self.receiptsFile)
    }

    public func scrub(_ text: String) -> String {
        // The shared versioned rule set: the curation stage's sensitivity
        // probe runs the very same patterns read-only (TesseraScrubRules).
        TesseraScrubRules.scrub(text)
    }

    // MARK: - Curation analytics (pure, testable)

    /// Stable content hash for dedup: FNV-1a over the scrubbed (kind, success,
    /// detail) tuple. Deterministic across launches (String.hashValue is not),
    /// so a repeated outcome is recognized on a later run. Scrubbing first
    /// means two outcomes that differ only in a secret value dedup together.
    public func contentHash(_ outcome: TesseraWorldOutcome) -> String {
        let material = "\(outcome.kind.rawValue)|\(outcome.success ? 1 : 0)|\(scrub(outcome.detail))"
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in material.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01b3
        }
        return String(hash, radix: 16)
    }

    /// Heuristic per-item quality in 0...1: how usable this outcome is as
    /// training signal. A clean success with enough detail to learn from
    /// scores highest; a bare outcome scores lower. Deliberately a heuristic,
    /// not a learned reward.
    public func qualityScore(_ outcome: TesseraWorldOutcome) -> Double {
        var q = outcome.success ? 0.6 : 0.3
        // Detail makes an outcome actionable; cap the bonus so verbosity alone
        // does not dominate.
        let words = outcome.detail.split(whereSeparator: { $0 == " " || $0 == "\n" }).count
        q += min(Double(words) / 20.0, 1.0) * 0.3
        // A commit is a strong verified signal; a revert a strong negative one.
        switch outcome.kind {
        case .commit: q += 0.1
        case .revert: q += 0.05
        case .test:   q += 0.05
        case .build:  break
        }
        return min(max(q, 0.0), 1.0)
    }

    /// Form (chosen, rejected) preference pairs per problem class from world
    /// outcomes: a passing outcome is "chosen", a failing one "rejected". v1
    /// keys the class by outcome KIND (build/test/commit/revert); a finer
    /// class label (e.g. the linked escalation frame's problemClass) is a
    /// plug-in point. Pairs are matched by recency (most recent pass with most
    /// recent fail), up to the smaller side's count.
    public func preferencePairs(from outcomes: [TesseraWorldOutcome]) -> [TesseraPreferencePair] {
        var passes: [String: [TesseraWorldOutcome]] = [:]
        var fails: [String: [TesseraWorldOutcome]] = [:]
        for outcome in outcomes {
            let cls = outcome.kind.rawValue
            if outcome.success { passes[cls, default: []].append(outcome) }
            else { fails[cls, default: []].append(outcome) }
        }

        var pairs: [TesseraPreferencePair] = []
        for cls in passes.keys.sorted() {
            guard let chosen = passes[cls], let rejected = fails[cls] else { continue }
            let n = min(chosen.count, rejected.count)
            for i in 0..<n {
                pairs.append(TesseraPreferencePair(
                    problemClass: cls,
                    chosen: chosen[chosen.count - 1 - i],
                    rejected: rejected[rejected.count - 1 - i]
                ))
            }
        }
        return pairs
    }

    /// Novelty / informativeness of a candidate outcome relative to what is
    /// already stored, in 0...1. Prioritizes what to learn next: unseen
    /// content outweighs a repeat, a rare outcome kind outweighs a common one,
    /// and a surprising result (a failure where we usually pass, or a success
    /// where we usually fail) outweighs an expected one.
    public func informativeness(of candidate: TesseraWorldOutcome, against stored: [TesseraWorldOutcome]) -> Double {
        // Content novelty: 1 if this exact scrubbed content is unseen, else 0.
        let seen = Set(stored.map { contentHash($0) })
        let novelty = seen.contains(contentHash(candidate)) ? 0.0 : 1.0

        // Kind rarity: 1 - frequency of this kind among stored outcomes.
        let rarity: Double
        if stored.isEmpty {
            rarity = 1.0
        } else {
            let sameKind = stored.filter { $0.kind == candidate.kind }.count
            rarity = 1.0 - Double(sameKind) / Double(stored.count)
        }

        // Surprise: a result that runs against the class's success rate.
        let classOutcomes = stored.filter { $0.kind == candidate.kind }
        let surprise: Double
        if classOutcomes.isEmpty {
            surprise = 0.5
        } else {
            let successRate = Double(classOutcomes.filter { $0.success }.count) / Double(classOutcomes.count)
            surprise = candidate.success ? (1.0 - successRate) : successRate
        }

        return min(max(novelty * 0.6 + rarity * 0.2 + surprise * 0.2, 0.0), 1.0)
    }

    public func summary() -> TesseraCurationSummary {
        lock.lock(); defer { lock.unlock() }
        let outcomes = store.load([TesseraWorldOutcome].self, from: Self.outcomesFile, default: [])
        let dedup = store.load(DedupState.self, from: Self.dedupFile, default: DedupState())
        let pairs = preferencePairs(from: outcomes)
        let meanQuality = outcomes.isEmpty
            ? 0.0
            : outcomes.map { qualityScore($0) }.reduce(0, +) / Double(outcomes.count)
        return TesseraCurationSummary(
            stored: outcomes.count,
            dedupSkipped: dedup.skipped,
            preferencePairs: pairs.count,
            meanQuality: meanQuality
        )
    }

    public func purgeTrainingData() throws -> Int {
        lock.lock(); defer { lock.unlock() }
        let count = store.load([TesseraWorldOutcome].self, from: Self.outcomesFile, default: []).count
        try store.delete(Self.outcomesFile)
        try store.delete(Self.receiptsFile)
        try store.delete(Self.dedupFile)
        return count
    }
}
