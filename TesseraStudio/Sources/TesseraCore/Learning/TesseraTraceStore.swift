import Foundation

/// One captured runtime session: every runtime record sharing a sid
/// (runtime-traces spec section 8). One provider generation call stamps
/// one sid, so a session is one turn's spec-decoding steps.
public struct TesseraRuntimeSessionSummary: Sendable, Equatable {
    public let sid: String
    public var records: Int
    public var accepted: Int
    public var drafted: Int

    public var acceptanceRate: Double? {
        drafted > 0 ? Double(accepted) / Double(drafted) : nil
    }

    init(sid: String, records: Int, accepted: Int, drafted: Int) {
        self.sid = sid
        self.records = records
        self.accepted = accepted
        self.drafted = drafted
    }
}

/// Store-wide runtime capture stats for the dashboard capture row.
public struct TesseraRuntimeCaptureSummary: Sendable, Equatable {
    public var totalRecords = 0
    public var totalBytes = 0
    /// Distinct sessions, oldest capture first.
    public var sessions: [TesseraRuntimeSessionSummary] = []

    public init() {}

    public var latestSession: TesseraRuntimeSessionSummary? { sessions.last }

    /// accepted/drafted across every captured step.
    public var acceptanceRate: Double? {
        let drafted = sessions.reduce(0) { $0 + $1.drafted }
        guard drafted > 0 else { return nil }
        let accepted = sessions.reduce(0) { $0 + $1.accepted }
        return Double(accepted) / Double(drafted)
    }
}

/// File-backed accumulator for llama.tessera.spec.v1 telemetry JSONL.
/// Traces accumulate across imatrix calibration runs; the orchestrator
/// reads them when enough data has gathered to form a training dataset.
///
/// Each completed imatrix run is copied in as a dated file
/// (traces-YYYYMMDD-HHMMSS.jsonl) under <learningStoreDir>/traces/. The
/// record count is cached and invalidated on append/purge so the training
/// gate can check it cheaply.
public final class TesseraTraceStore: @unchecked Sendable {
    private let directory: URL
    private let lock = NSLock()
    private var cachedRecordCount: Int?
    private var runtimeIndexCache: [RuntimeFileEntry]?
    private var s2sIndexCache: [S2SFileEntry]?

    /// Filename prefixes per provenance (runtime-traces spec section 8).
    /// All keep the traces- prefix, so totalRecords() counts every
    /// provenance and the training gate sees the combined total.
    public static let runtimeFilePrefix = "traces-runtime-"
    public static let replayFilePrefix = "traces-replay-"
    /// S2S utterance records (s2s design section 4.3). Codes are Tier B
    /// local-only, so the staging filter skips this prefix outright.
    public static let s2sFilePrefix = "traces-s2s-"

    /// Default rolling cap on the runtime share (spec section 8). The
    /// runtime trimmer never touches calibration or replay files.
    public static let runtimeBudgetBytesDefault = 200 * 1024 * 1024

    /// Default rolling cap on the s2s share. Same discipline as the runtime
    /// cap; the s2s trimmer never touches any other provenance.
    public static let s2sBudgetBytesDefault = 200 * 1024 * 1024

    public init(directory: URL = TesseraTraceStore.defaultDirectory()) {
        self.directory = directory
    }

    /// The store's traces/ directory. Exposed so siblings under the same
    /// learning root (the curation ledger, the curation stage state) resolve
    /// to matching locations in tests and in the app alike.
    public var directoryURL: URL { directory }

    public static func defaultDirectory() -> URL {
        TesseraLearningStore.defaultDirectory().appendingPathComponent("traces", isDirectory: true)
    }

    /// Copy a completed telemetry file into the store under a dated name.
    /// The source is left in place; a numeric suffix disambiguates two runs
    /// that land in the same second. Returns the stored file URL.
    @discardableResult
    public func appendRun(jsonlPath: URL) throws -> URL {
        lock.lock(); defer { lock.unlock() }
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let stem = Self.datedStem(Date())
        var name = "\(stem).jsonl"
        var n = 1
        while fm.fileExists(atPath: directory.appendingPathComponent(name).path) {
            name = "\(stem)-\(n).jsonl"
            n += 1
        }
        let dest = directory.appendingPathComponent(name)
        try fm.copyItem(at: jsonlPath, to: dest)
        cachedRecordCount = nil
        runtimeIndexCache = nil
        s2sIndexCache = nil
        return dest
    }

    /// Append runtime-captured telemetry records (runtime-traces spec
    /// section 8). Records already carry "provenance":"runtime" and one
    /// "sid" per provider generation call; they are written verbatim as
    /// traces-runtime-<date>.jsonl. Returns the stored file URL, or nil
    /// when there was nothing to write. After writing, retention and the
    /// runtime rolling cap are enforced; quarantined sessions (section 12)
    /// are exempt via exemptSids, wired by the curation stage.
    @discardableResult
    public func appendRuntime(records: [String], exemptSids: Set<String> = []) throws -> URL? {
        lock.lock(); defer { lock.unlock() }
        guard !records.isEmpty else { return nil }
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let stem = Self.datedStem(Date(), prefix: Self.runtimeFilePrefix)
        var name = "\(stem).jsonl"
        var n = 1
        while fm.fileExists(atPath: directory.appendingPathComponent(name).path) {
            name = "\(stem)-\(n).jsonl"
            n += 1
        }
        let dest = directory.appendingPathComponent(name)
        try (records.joined(separator: "\n") + "\n")
            .write(to: dest, atomically: true, encoding: .utf8)
        cachedRecordCount = nil
        runtimeIndexCache = nil
        s2sIndexCache = nil

        // Retention first (all provenances), then the runtime rolling cap.
        try trimExpiredUnlocked(
            retentionDays: TesseraSettings.learningDataRetentionDays,
            exemptSids: exemptSids, now: Date())
        try trimRuntimeUnlocked(
            budgetBytes: Self.runtimeBudgetBytesDefault, exemptSids: exemptSids)
        return dest
    }

    /// Append replay-derived telemetry records (runtime-traces spec section
    /// 12.2). Records are imatrix calibration output over a decoded session
    /// corpus, stamped by the curation stage with "provenance":"replay" and
    /// "replayed_from":"runtime"; they carry no sid (the sid is stripped at
    /// promotion). Written verbatim as traces-replay-<date>.jsonl. Date-based
    /// retention applies to replay files; the runtime rolling cap never does.
    /// Returns the stored file URL, or nil when there was nothing to write.
    @discardableResult
    public func appendReplay(records: [String], exemptSids: Set<String> = []) throws -> URL? {
        lock.lock(); defer { lock.unlock() }
        guard !records.isEmpty else { return nil }
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let stem = Self.datedStem(Date(), prefix: Self.replayFilePrefix)
        var name = "\(stem).jsonl"
        var n = 1
        while fm.fileExists(atPath: directory.appendingPathComponent(name).path) {
            name = "\(stem)-\(n).jsonl"
            n += 1
        }
        let dest = directory.appendingPathComponent(name)
        try (records.joined(separator: "\n") + "\n")
            .write(to: dest, atomically: true, encoding: .utf8)
        cachedRecordCount = nil
        runtimeIndexCache = nil
        s2sIndexCache = nil

        // Retention covers every provenance; quarantined sessions stay exempt.
        try trimExpiredUnlocked(
            retentionDays: TesseraSettings.learningDataRetentionDays,
            exemptSids: exemptSids, now: Date())
        return dest
    }

    /// Replay trace files, oldest-first.
    public func replayFiles() -> [URL] {
        lock.lock(); defer { lock.unlock() }
        return traceFilesUnlocked()
            .filter { $0.lastPathComponent.hasPrefix(Self.replayFilePrefix) }
    }

    /// Append captured S2S utterance records (s2s design section 4.3).
    /// Records are TesseraS2SRecord lines: they carry "provenance":"s2s"
    /// and one device-local sid per utterance, and are written verbatim as
    /// traces-s2s-<date>.jsonl. Capture is default-on with no opt-out
    /// (mandatory-collection doctrine); codes are Tier B local-only, so
    /// this share never touches dataset staging. Returns the stored file
    /// URL, or nil when there was nothing to write. After writing,
    /// retention and the s2s rolling cap are enforced; quarantined sessions
    /// are exempt via exemptSids, wired by the curation stage.
    @discardableResult
    public func appendS2S(records: [String], exemptSids: Set<String> = []) throws -> URL? {
        lock.lock(); defer { lock.unlock() }
        guard !records.isEmpty else { return nil }
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let stem = Self.datedStem(Date(), prefix: Self.s2sFilePrefix)
        var name = "\(stem).jsonl"
        var n = 1
        while fm.fileExists(atPath: directory.appendingPathComponent(name).path) {
            name = "\(stem)-\(n).jsonl"
            n += 1
        }
        let dest = directory.appendingPathComponent(name)
        try (records.joined(separator: "\n") + "\n")
            .write(to: dest, atomically: true, encoding: .utf8)
        cachedRecordCount = nil
        runtimeIndexCache = nil
        s2sIndexCache = nil

        // Retention first (all provenances), then the s2s rolling cap.
        try trimExpiredUnlocked(
            retentionDays: TesseraSettings.learningDataRetentionDays,
            exemptSids: exemptSids, now: Date())
        try trimS2SUnlocked(
            budgetBytes: Self.s2sBudgetBytesDefault, exemptSids: exemptSids)
        return dest
    }

    /// S2S trace files, oldest-first.
    public func s2sFiles() -> [URL] {
        lock.lock(); defer { lock.unlock() }
        return traceFilesUnlocked()
            .filter { $0.lastPathComponent.hasPrefix(Self.s2sFilePrefix) }
    }

    /// Stored trace files, oldest-first (the dated names sort chronologically).
    public func traceFiles() -> [URL] {
        lock.lock(); defer { lock.unlock() }
        return traceFilesUnlocked()
    }

    /// Total JSONL records across all files (non-empty lines), cached.
    public func totalRecords() -> Int {
        lock.lock(); defer { lock.unlock() }
        if let cached = cachedRecordCount { return cached }
        let total = traceFilesUnlocked().reduce(0) { $0 + Self.countRecords(in: $1) }
        cachedRecordCount = total
        return total
    }

    /// Delete all trace files. Returns the number of files removed.
    @discardableResult
    public func purge() throws -> Int {
        lock.lock(); defer { lock.unlock() }
        let files = traceFilesUnlocked()
        for file in files { try FileManager.default.removeItem(at: file) }
        cachedRecordCount = nil
        runtimeIndexCache = nil
        s2sIndexCache = nil
        return files.count
    }

    // MARK: - TesseraPurgeable

    /// Delete all trace files. Returns the number of records removed, per
    /// the purgeable contract (as opposed to `purge`, which counts files).
    public func purgeTrainingData() throws -> Int {
        lock.lock(); defer { lock.unlock() }
        let files = traceFilesUnlocked()
        let records = files.reduce(0) { $0 + Self.countRecords(in: $1) }
        for file in files { try FileManager.default.removeItem(at: file) }
        cachedRecordCount = nil
        runtimeIndexCache = nil
        s2sIndexCache = nil
        return records
    }

    // MARK: - Runtime capture (runtime-traces spec section 8)

    /// Runtime trace files, oldest-first.
    public func runtimeFiles() -> [URL] {
        lock.lock(); defer { lock.unlock() }
        return runtimeIndexUnlocked().map { $0.url }
    }

    /// Session-grouped runtime stats for the dashboard capture row.
    /// Sessions are ordered oldest capture first; a sid split across
    /// several files (a retried flush) merges into one session.
    public func runtimeSummary() -> TesseraRuntimeCaptureSummary {
        lock.lock(); defer { lock.unlock() }
        var summary = TesseraRuntimeCaptureSummary()
        var order: [String] = []
        var merged: [String: TesseraRuntimeSessionSummary] = [:]
        for entry in runtimeIndexUnlocked() {
            summary.totalBytes += entry.bytes
            summary.totalRecords += entry.records
            for (sid, t) in entry.sessionTotals {
                if var existing = merged[sid] {
                    existing.records += t.records
                    existing.accepted += t.accepted
                    existing.drafted += t.drafted
                    merged[sid] = existing
                } else {
                    order.append(sid)
                    merged[sid] = TesseraRuntimeSessionSummary(
                        sid: sid, records: t.records,
                        accepted: t.accepted, drafted: t.drafted)
                }
            }
        }
        summary.sessions = order.compactMap { merged[$0] }
        return summary
    }

    /// Rolling cap: trim runtime files oldest-first until the runtime share
    /// fits the budget. Calibration and replay files are never touched.
    /// Files holding a sid in exemptSids (quarantined sessions) are exempt;
    /// the share may stay over budget rather than remove them. Returns the
    /// number of files removed.
    @discardableResult
    public func trimRuntimeToBudget(budgetBytes: Int, exemptSids: Set<String> = []) throws -> Int {
        lock.lock(); defer { lock.unlock() }
        return try trimRuntimeUnlocked(budgetBytes: budgetBytes, exemptSids: exemptSids)
    }

    /// Date-based retention: remove files of ANY provenance older than
    /// retentionDays, except files holding a sid in exemptSids (quarantined
    /// sessions are exempt from automatic retention entirely; only
    /// user-initiated purge removes them). Returns the number of files
    /// removed.
    @discardableResult
    public func trimExpired(retentionDays: Int, exemptSids: Set<String> = [], now: Date = Date()) throws -> Int {
        lock.lock(); defer { lock.unlock() }
        return try trimExpiredUnlocked(retentionDays: retentionDays, exemptSids: exemptSids, now: now)
    }

    /// User-initiated purge of one session (spec sections 9 and 12.4):
    /// remove every runtime or s2s record carrying the sid, rewriting the
    /// affected files in place and deleting any file left empty.
    /// Quarantined sessions are exempt from automatic retention entirely;
    /// this is the ONLY path that removes them. Calibration and replay
    /// files are never touched (a promoted session loses its sid before
    /// replay, so no replay record can carry it). Returns the number of
    /// records removed.
    @discardableResult
    public func purgeSession(sid: String) throws -> Int {
        lock.lock(); defer { lock.unlock() }
        guard !sid.isEmpty else { return 0 }
        let fm = FileManager.default
        var removed = 0
        let targets = runtimeIndexUnlocked().map { ($0.url, $0.sids) }
            + s2sIndexUnlocked().map { ($0.url, $0.sids) }
        for (url, sids) in targets where sids.contains(sid) {
            guard let text = try? String(contentsOf: url, encoding: .utf8) else { continue }
            var kept: [String] = []
            text.enumerateLines { line, _ in
                guard !line.trimmingCharacters(in: .whitespaces).isEmpty else { return }
                if Self.lineSid(line) == sid {
                    removed += 1
                    return
                }
                kept.append(line)
            }
            if kept.isEmpty {
                try fm.removeItem(at: url)
            } else {
                try (kept.joined(separator: "\n") + "\n")
                    .write(to: url, atomically: true, encoding: .utf8)
            }
        }
        if removed > 0 {
            cachedRecordCount = nil
            runtimeIndexCache = nil
            s2sIndexCache = nil
        }
        return removed
    }

    // MARK: - Trimming (caller holds the lock)

    @discardableResult
    private func trimRuntimeUnlocked(budgetBytes: Int, exemptSids: Set<String>) throws -> Int {
        let entries = runtimeIndexUnlocked()
        var total = entries.reduce(0) { $0 + $1.bytes }
        guard total > budgetBytes else { return 0 }
        var removed = 0
        for entry in entries where total > budgetBytes {
            guard entry.sids.isDisjoint(with: exemptSids) else { continue }
            try FileManager.default.removeItem(at: entry.url)
            total -= entry.bytes
            removed += 1
        }
        if removed > 0 {
            cachedRecordCount = nil
            runtimeIndexCache = nil
        }
        return removed
    }

    @discardableResult
    private func trimExpiredUnlocked(retentionDays: Int, exemptSids: Set<String>, now: Date) throws -> Int {
        guard retentionDays > 0 else { return 0 }
        let cutoff = now.addingTimeInterval(-Double(retentionDays) * 86_400)
        // Sid-bearing provenances (runtime, s2s) can hold quarantined
        // sessions; their files are exempt from automatic retention.
        var sidsByFile: [URL: Set<String>] = Dictionary(
            uniqueKeysWithValues: runtimeIndexUnlocked().map { ($0.url, $0.sids) })
        for entry in s2sIndexUnlocked() { sidsByFile[entry.url] = entry.sids }
        var removed = 0
        for file in traceFilesUnlocked() {
            guard let created = try? file.resourceValues(forKeys: [.creationDateKey]),
                  let createdDate = created.creationDate, createdDate < cutoff else { continue }
            let sids = sidsByFile[file] ?? []
            guard sids.isDisjoint(with: exemptSids) else { continue }
            try FileManager.default.removeItem(at: file)
            removed += 1
        }
        if removed > 0 {
            cachedRecordCount = nil
            runtimeIndexCache = nil
            s2sIndexCache = nil
        }
        return removed
    }

    /// Rolling cap for the s2s share: trim s2s files oldest-first until the
    /// share fits the budget. Every other provenance is never touched.
    /// Files holding a sid in exemptSids (quarantined sessions) are exempt;
    /// the share may stay over budget rather than remove them. Returns the
    /// number of files removed.
    @discardableResult
    public func trimS2SToBudget(budgetBytes: Int, exemptSids: Set<String> = []) throws -> Int {
        lock.lock(); defer { lock.unlock() }
        return try trimS2SUnlocked(budgetBytes: budgetBytes, exemptSids: exemptSids)
    }

    @discardableResult
    private func trimS2SUnlocked(budgetBytes: Int, exemptSids: Set<String>) throws -> Int {
        let entries = s2sIndexUnlocked()
        var total = entries.reduce(0) { $0 + $1.bytes }
        guard total > budgetBytes else { return 0 }
        var removed = 0
        for entry in entries where total > budgetBytes {
            guard entry.sids.isDisjoint(with: exemptSids) else { continue }
            try FileManager.default.removeItem(at: entry.url)
            total -= entry.bytes
            removed += 1
        }
        if removed > 0 {
            cachedRecordCount = nil
            s2sIndexCache = nil
        }
        return removed
    }

    // MARK: - Runtime index (caller holds the lock)

    /// One parsed runtime file: byte size, line count, and per-sid totals.
    private struct RuntimeFileEntry {
        let url: URL
        let bytes: Int
        let records: Int
        let sids: Set<String>
        let sessionTotals: [String: (records: Int, accepted: Int, drafted: Int)]
    }

    private func runtimeIndexUnlocked() -> [RuntimeFileEntry] {
        if let cached = runtimeIndexCache { return cached }
        var entries: [RuntimeFileEntry] = []
        for file in traceFilesUnlocked()
        where file.lastPathComponent.hasPrefix(Self.runtimeFilePrefix) {
            guard let data = try? Data(contentsOf: file),
                  let text = String(data: data, encoding: .utf8) else { continue }
            var records = 0
            var sids = Set<String>()
            var totals: [String: (records: Int, accepted: Int, drafted: Int)] = [:]
            text.enumerateLines { line, _ in
                guard !line.trimmingCharacters(in: .whitespaces).isEmpty else { return }
                records += 1
                guard let parsed = Self.parseRuntimeLine(line), let sid = parsed.sid else { return }
                sids.insert(sid)
                var t = totals[sid] ?? (0, 0, 0)
                t.records += 1
                t.accepted += parsed.accepted
                t.drafted += parsed.drafted
                totals[sid] = t
            }
            entries.append(RuntimeFileEntry(
                url: file, bytes: data.count, records: records,
                sids: sids, sessionTotals: totals))
        }
        runtimeIndexCache = entries
        return entries
    }

    /// Extract (sid, accepted, drafted) from one telemetry record. A record
    /// without a sid still counts toward the file's line total but joins no
    /// session (the engine always stamps one; this stays honest if not).
    private static func parseRuntimeLine(_ line: String) -> (sid: String?, accepted: Int, drafted: Int)? {
        guard let data = line.data(using: .utf8),
              let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            return nil
        }
        let sid = (obj["sid"] as? String).flatMap { $0.isEmpty ? nil : $0 }
        let accepted = (obj["accepted"] as? NSNumber)?.intValue ?? 0
        let drafted = (obj["drafted"] as? NSNumber)?.intValue ?? 0
        return (sid, accepted, drafted)
    }

    /// The sid one JSONL record carries, or nil for unparseable lines and
    /// empty sids. Shared by every provenance whose records carry a sid
    /// (runtime and s2s).
    private static func lineSid(_ line: String) -> String? {
        guard let data = line.data(using: .utf8),
              let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            return nil
        }
        return (obj["sid"] as? String).flatMap { $0.isEmpty ? nil : $0 }
    }

    // MARK: - S2S index (caller holds the lock)

    /// One parsed s2s file: byte size and its sids (one per utterance).
    private struct S2SFileEntry {
        let url: URL
        let bytes: Int
        let sids: Set<String>
    }

    private func s2sIndexUnlocked() -> [S2SFileEntry] {
        if let cached = s2sIndexCache { return cached }
        var entries: [S2SFileEntry] = []
        for file in traceFilesUnlocked()
        where file.lastPathComponent.hasPrefix(Self.s2sFilePrefix) {
            guard let data = try? Data(contentsOf: file),
                  let text = String(data: data, encoding: .utf8) else { continue }
            var sids = Set<String>()
            text.enumerateLines { line, _ in
                guard !line.trimmingCharacters(in: .whitespaces).isEmpty else { return }
                if let sid = Self.lineSid(line) { sids.insert(sid) }
            }
            entries.append(S2SFileEntry(url: file, bytes: data.count, sids: sids))
        }
        s2sIndexCache = entries
        return entries
    }

    // MARK: - Helpers (caller holds the lock)

    private func traceFilesUnlocked() -> [URL] {
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        ) else { return [] }
        return entries
            .filter { $0.pathExtension == "jsonl" && $0.lastPathComponent.hasPrefix("traces-") }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    /// Number of non-empty JSONL records in a single trace file. Public so a
    /// producer can report honestly how many records it just appended.
    public static func recordCount(inFile file: URL) -> Int {
        countRecords(in: file)
    }

    private static func countRecords(in file: URL) -> Int {
        guard let text = try? String(contentsOf: file, encoding: .utf8) else { return 0 }
        var count = 0
        text.enumerateLines { line, _ in
            if !line.trimmingCharacters(in: .whitespaces).isEmpty { count += 1 }
        }
        return count
    }

    private static func datedStem(_ date: Date, prefix: String = "traces-") -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        return "\(prefix)\(formatter.string(from: date))"
    }
}
