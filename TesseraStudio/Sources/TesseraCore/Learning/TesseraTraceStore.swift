import Foundation

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

    public init(directory: URL = TesseraTraceStore.defaultDirectory()) {
        self.directory = directory
    }

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
        return dest
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
        return records
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

    private static func countRecords(in file: URL) -> Int {
        guard let text = try? String(contentsOf: file, encoding: .utf8) else { return 0 }
        var count = 0
        text.enumerateLines { line, _ in
            if !line.trimmingCharacters(in: .whitespaces).isEmpty { count += 1 }
        }
        return count
    }

    private static func datedStem(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        return "traces-\(formatter.string(from: date))"
    }
}
