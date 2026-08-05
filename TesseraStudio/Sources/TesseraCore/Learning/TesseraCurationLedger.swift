import Foundation

/// One verdict in the curation ledger (runtime-traces spec section 12.4,
/// schema llama.tessera.curation.v1). The ledger is append-only: a session
/// re-analyzed under a newer scrubber version appends a fresh entry and the
/// latest entry wins on decode.
public struct TesseraCurationLedgerEntry: Codable, Sendable, Equatable {
    public struct Score: Codable, Sendable, Equatable {
        public var acceptance: Double
        public var tokens: Int
        public var repetition: Double

        public init(acceptance: Double, tokens: Int, repetition: Double) {
            self.acceptance = acceptance
            self.tokens = tokens
            self.repetition = repetition
        }
    }

    public let schema: String
    public let sid: String
    public let verdict: String
    public let reasons: [String]
    public let score: Score
    public let anonymizerRequiredVersion: String
    public let ts: String

    private enum CodingKeys: String, CodingKey {
        case schema
        case sid
        case verdict
        case reasons
        case score
        case anonymizerRequiredVersion = "anonymizer_required_version"
        case ts
    }

    public init(
        schema: String = TesseraCurationLedger.schema,
        sid: String,
        verdict: TesseraSessionVerdict,
        reasons: [String],
        score: Score,
        anonymizerRequiredVersion: String = TesseraScrubRules.requiredVersionStamp,
        ts: String = TesseraCurationLedger.timestamp(Date())
    ) {
        self.schema = schema
        self.sid = sid
        self.verdict = verdict.rawValue
        self.reasons = reasons
        self.score = score
        self.anonymizerRequiredVersion = anonymizerRequiredVersion
        self.ts = ts
    }

    /// Tolerant verdict decode: an unknown verdict string (a future schema
    /// addition) reads as nil so the session is re-analyzed, never crashed on.
    public var verdictValue: TesseraSessionVerdict? {
        TesseraSessionVerdict(rawValue: verdict)
    }
}

/// Append-only, file-backed verdict ledger at
/// <learningStoreDir>/curation-ledger.jsonl. Device-local analysis metadata:
/// it never leaves the machine - no manifest, batch, or egress artifact
/// references it (spec section 12.4).
public final class TesseraCurationLedger: @unchecked Sendable {
    public static let schema = "llama.tessera.curation.v1"
    public static let fileName = "curation-ledger.jsonl"

    private let fileURL: URL
    private let lock = NSLock()

    /// The learning data root (same location TesseraLearningStore uses);
    /// the ledger lives directly under it, next to traces/.
    public static func defaultDirectory() -> URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        return base.appendingPathComponent("TesseraStudio/learning", isDirectory: true)
    }

    public init(directory: URL = TesseraCurationLedger.defaultDirectory()) {
        self.fileURL = directory.appendingPathComponent(Self.fileName)
    }

    public var url: URL { fileURL }

    /// Append one verdict. The write is a single line append; the ledger is
    /// never rewritten, so an interrupted sweep can at most lose the entry
    /// being written.
    public func append(_ entry: TesseraCurationLedgerEntry) throws {
        lock.lock(); defer { lock.unlock() }
        let fm = FileManager.default
        try fm.createDirectory(
            at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        var data = try JSONEncoder().encode(entry)
        data.append(0x0A)
        if fm.fileExists(atPath: fileURL.path) {
            let handle = try FileHandle(forWritingTo: fileURL)
            defer { try? handle.close() }
            try handle.seekToEnd()
            try handle.write(contentsOf: data)
        } else {
            try data.write(to: fileURL)
        }
    }

    /// Every entry in append order. Tolerant: malformed lines and foreign
    /// schemas are skipped, never thrown - the ledger survives hand edits.
    public func entries() -> [TesseraCurationLedgerEntry] {
        lock.lock(); defer { lock.unlock() }
        guard let text = try? String(contentsOf: fileURL, encoding: .utf8) else { return [] }
        var out: [TesseraCurationLedgerEntry] = []
        text.enumerateLines { line, _ in
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty, let data = trimmed.data(using: .utf8) else { return }
            guard let entry = try? JSONDecoder().decode(TesseraCurationLedgerEntry.self, from: data),
                  entry.schema == Self.schema else { return }
            out.append(entry)
        }
        return out
    }

    /// Latest entry per sid (append-only, latest wins).
    public func latestVerdicts() -> [String: TesseraCurationLedgerEntry] {
        var latest: [String: TesseraCurationLedgerEntry] = [:]
        for entry in entries() {
            latest[entry.sid] = entry
        }
        return latest
    }

    public func verdict(for sid: String) -> TesseraSessionVerdict? {
        latestVerdicts()[sid]?.verdictValue
    }

    /// Sids whose LATEST verdict is the given one.
    public func sids(with verdict: TesseraSessionVerdict) -> Set<String> {
        Set(latestVerdicts().filter { $0.value.verdictValue == verdict }.map { $0.key })
    }

    public func quarantinedSids() -> Set<String> { sids(with: .quarantined) }
    public func promotedSids() -> Set<String> { sids(with: .promoted) }

    /// Quarantined sessions with their latest entry, newest first, for the
    /// dashboard quarantine list (date, token count, probe class - never the
    /// matched content).
    public func quarantinedSessions() -> [TesseraCurationLedgerEntry] {
        let latest = latestVerdicts()
        return latest.values
            .filter { $0.verdictValue == .quarantined }
            .sorted { $0.ts > $1.ts }
    }

    /// Display infos for the dashboard quarantine list, newest first.
    public func quarantinedSessionInfos() -> [TesseraQuarantinedSessionInfo] {
        quarantinedSessions().map { entry in
            TesseraQuarantinedSessionInfo(
                sid: entry.sid,
                date: Self.date(fromTimestamp: entry.ts),
                tokenCount: entry.score.tokens,
                probeClasses: TesseraProbeClass.classes(forLedgerReasons: entry.reasons))
        }
    }

    /// Append a user-initiated purge verdict (spec section 12.4). Latest
    /// wins, so the session leaves the quarantine list and a future sweep
    /// never re-analyzes it.
    public func markPurged(sid: String) throws {
        try append(TesseraCurationLedgerEntry(
            sid: sid,
            verdict: .purged,
            reasons: ["user-purge"],
            score: .init(acceptance: 0, tokens: 0, repetition: 0)))
    }

    /// The ledger that serves a trace store: the store directory is
    /// <learning>/traces and the ledger lives directly under <learning>.
    public static func forStore(_ store: TesseraTraceStore) -> TesseraCurationLedger {
        TesseraCurationLedger(directory: store.directoryURL.deletingLastPathComponent())
    }

    /// Curation state over the sessions currently present in the runtime
    /// store (spec section 10): promoted / quarantined totals plus sessions
    /// captured but not judged yet. Dropped and purged sessions count
    /// toward neither; judged sessions whose records were already trimmed
    /// do not appear at all.
    public func curationCounts(sessionSids: Set<String>) -> TesseraCurationCounts {
        let latest = latestVerdicts()
        var counts = TesseraCurationCounts()
        for sid in sessionSids {
            switch latest[sid]?.verdictValue {
            case .promoted:    counts.promoted += 1
            case .quarantined: counts.quarantined += 1
            case nil:          counts.pending += 1
            case .dropped, .purged: break
            }
        }
        return counts
    }

    /// Inverse of ``timestamp(_:)``; nil on foreign formats.
    public static func date(fromTimestamp ts: String) -> Date? {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss'Z'"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(identifier: "UTC")
        return formatter.date(from: ts)
    }

    /// Ledger ISO8601 UTC timestamps: 2026-08-04T22:30:00Z.
    public static func timestamp(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss'Z'"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(identifier: "UTC")
        return formatter.string(from: date)
    }
}

/// Curation state counts for the dashboard capture row (spec section 10).
public struct TesseraCurationCounts: Sendable, Equatable {
    public var promoted: Int
    public var quarantined: Int
    public var pending: Int

    public init(promoted: Int = 0, quarantined: Int = 0, pending: Int = 0) {
        self.promoted = promoted
        self.quarantined = quarantined
        self.pending = pending
    }
}

/// Display model for the dashboard quarantine list (spec section 10):
/// session date, token count, and the probe class that quarantined it -
/// never the matched content itself.
public struct TesseraQuarantinedSessionInfo: Sendable, Equatable, Identifiable {
    public let sid: String
    public let date: Date?
    public let tokenCount: Int
    public let probeClasses: [String]

    public var id: String { sid }

    public init(sid: String, date: Date?, tokenCount: Int, probeClasses: [String]) {
        self.sid = sid
        self.date = date
        self.tokenCount = tokenCount
        self.probeClasses = probeClasses
    }
}
