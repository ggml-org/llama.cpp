import Foundation

/// Persists the last ``PleadTheFifthExecutor.WipeReport`` to a fixed
/// location in the user's home so the user has an audit trail of what
/// was destroyed and when. The file is plain JSON; it contains no
/// secrets. The user can delete it manually (per the design, "no trace"
/// is opt-in).
public struct WipeReportStore {
    public static let fileURL: URL = {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".tessera/last-wipe.json")
    }()

    public let fileURL: URL

    public init(fileURL: URL = WipeReportStore.fileURL) {
        self.fileURL = fileURL
    }

    /// Encode `report` as pretty-printed JSON and write it to ``fileURL``
    /// atomically. The parent directory is created on demand. Atomic
    /// write is critical: a half-written report is worse than no report.
    public func save(_ report: PleadTheFifthExecutor.WipeReport) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(report)
        try FileManager.default.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try data.write(to: fileURL, options: .atomic)
    }

    /// Read the last wipe report, if any. Returns nil when the file
    /// does not exist (no wipe has been performed on this machine).
    public func loadIfPresent() throws -> PleadTheFifthExecutor.WipeReport? {
        let fm = FileManager.default
        guard fm.fileExists(atPath: fileURL.path) else { return nil }
        let data = try Data(contentsOf: fileURL)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(PleadTheFifthExecutor.WipeReport.self, from: data)
    }

    /// Convenience static form for the common case.
    public static func save(_ report: PleadTheFifthExecutor.WipeReport) throws {
        try WipeReportStore().save(report)
    }
}
