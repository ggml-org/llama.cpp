import Foundation

/// Small helpers shared by the engine tools when they shell out to
/// `tessera-cli`. Kept in one place so each tool stays a thin argument
/// builder; the work of writing the config file and parsing the JSON
/// response lives here.
enum EngineToolSupport {

    /// Write a JSON config dictionary to a temp file and return its path.
    /// The file lives under the caller's working directory when supplied,
    /// otherwise under NSTemporaryDirectory, and is the path passed to
    /// `tessera-cli <subcommand> --config <path>`. The caller is
    /// responsible for cleanup; the engine tools unlink the file after
    /// the subprocess exits.
    static func writeConfigFile(
        config: [String: Any],
        workingDirectory: String? = nil
    ) throws -> String {
        let dir: String
        if let wd = workingDirectory, !wd.isEmpty {
            dir = wd
        } else {
            dir = NSTemporaryDirectory()
        }
        let filename = "tessera-config-\(UUID().uuidString).json"
        let path = (dir as NSString).appendingPathComponent(filename)
        let data = try JSONSerialization.data(
            withJSONObject: config,
            options: [.sortedKeys]
        )
        try data.write(to: URL(fileURLWithPath: path), options: .atomic)
        return path
    }

    /// Parse a JSON object from a subprocess's stdout. Returns nil on any
    /// parse failure so the caller can fall back to reporting the raw
    /// stdout string. Tolerant of leading/trailing whitespace (some
    /// binaries print a status line before the JSON).
    static func parseJSONObject(stdout: String) -> [String: Any]? {
        let trimmed = stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = trimmed.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data),
              let dict = obj as? [String: Any] else { return nil }
        return dict
    }

    /// Parse a JSON array from a subprocess's stdout.
    static func parseJSONArray(stdout: String) -> [Any]? {
        let trimmed = stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = trimmed.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data),
              let arr = obj as? [Any] else { return nil }
        return arr
    }
}
