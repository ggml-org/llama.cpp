import Foundation

/// Concrete curation service: an append-only, file-backed record of curated
/// outcomes plus their receipts, and a secret scrubber for stored/egress
/// text (design 4.2). Scrubbing is deliberately conservative - it removes
/// obvious secrets, it is not a guarantee that nothing sensitive survives.
public final class TesseraCurationService: TesseraCurating, @unchecked Sendable {
    private let store: TesseraLearningStore
    private let lock = NSLock()
    private static let outcomesFile = "curation-outcomes.json"
    private static let receiptsFile = "curation-receipts.json"

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

        var outcomes = store.load([TesseraWorldOutcome].self, from: Self.outcomesFile, default: [])
        outcomes.append(outcome)
        try store.save(outcomes, to: Self.outcomesFile)

        let receipt = TesseraLearningReceipt(
            kind: "curation",
            summary: "Curated \(outcome.kind.rawValue) outcome (success=\(outcome.success)).",
            payload: [
                "outcomeId": .string(outcome.id),
                "kind": .string(outcome.kind.rawValue),
                "success": .bool(outcome.success),
            ]
        )
        var receipts = store.load([TesseraLearningReceipt].self, from: Self.receiptsFile, default: [])
        receipts.append(receipt)
        try store.save(receipts, to: Self.receiptsFile)
        return receipt
    }

    public func scrub(_ text: String) -> String {
        var out = text
        // PEM private-key blocks (spans lines).
        out = Self.replace(out,
            pattern: "-----BEGIN [A-Z ]*PRIVATE KEY-----[\\s\\S]*?-----END [A-Z ]*PRIVATE KEY-----",
            with: "[REDACTED PRIVATE KEY]")
        // Bearer tokens.
        out = Self.replace(out, pattern: "(?i)Bearer\\s+[A-Za-z0-9._\\-]+", with: "Bearer [REDACTED]")
        // OpenAI-style secret keys.
        out = Self.replace(out, pattern: "\\bsk-[A-Za-z0-9_\\-]{8,}", with: "sk-[REDACTED]")
        // KEY=VALUE lines whose name looks sensitive.
        out = Self.replace(out,
            pattern: "(?im)^(\\s*(?:export\\s+)?[A-Za-z0-9_]*(?:API_KEY|SECRET|TOKEN|PASSWORD)[A-Za-z0-9_]*\\s*[:=]\\s*).*$",
            with: "$1[REDACTED]")
        return out
    }

    public func purgeTrainingData() throws -> Int {
        lock.lock(); defer { lock.unlock() }
        let count = store.load([TesseraWorldOutcome].self, from: Self.outcomesFile, default: []).count
        try store.delete(Self.outcomesFile)
        try store.delete(Self.receiptsFile)
        return count
    }

    private static func replace(_ input: String, pattern: String, with template: String) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return input }
        let range = NSRange(input.startIndex..<input.endIndex, in: input)
        return regex.stringByReplacingMatches(in: input, options: [], range: range, withTemplate: template)
    }
}
