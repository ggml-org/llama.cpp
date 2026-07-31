import Foundation

/// Small file-backed JSON store helper for the learning subsystem. Loads
/// and saves Codable values under the app's learning data dir with atomic
/// writes. Missing or corrupt files decode to a caller-supplied default, so
/// a half-written or hand-edited file never crashes a read.
///
/// Synchronization is the caller's responsibility: a read-modify-write must
/// be guarded by the owning service's lock, since load + save here are two
/// separate operations.
struct TesseraLearningStore: Sendable {
    let directory: URL

    init(directory: URL = TesseraLearningStore.defaultDirectory()) {
        self.directory = directory
    }

    static func defaultDirectory() -> URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        return base.appendingPathComponent("TesseraStudio/learning", isDirectory: true)
    }

    private func fileURL(_ name: String) -> URL {
        directory.appendingPathComponent(name)
    }

    func load<T: Codable>(_ type: T.Type, from name: String, default fallback: T) -> T {
        guard let data = try? Data(contentsOf: fileURL(name)) else { return fallback }
        return (try? JSONDecoder().decode(T.self, from: data)) ?? fallback
    }

    func save<T: Codable>(_ value: T, to name: String) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let data = try JSONEncoder().encode(value)
        try data.write(to: fileURL(name), options: .atomic)
    }

    func delete(_ name: String) throws {
        let url = fileURL(name)
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
    }
}
