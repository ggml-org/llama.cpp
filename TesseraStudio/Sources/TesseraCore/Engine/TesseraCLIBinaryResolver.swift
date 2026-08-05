import Foundation

/// Resolves the `tessera-cli` binary used as the subprocess fallback for the
/// engine tools. Mirrors the precedence used by ``TesseraTrainBinaryResolver``:
/// the user override wins as-is (intent is never silently second-guessed),
/// then the per-key settings override, then the known install locations, then
/// `$PATH`. Always returns `nil` (not a path) when nothing executable is
/// found - the tool surfaces a "binary not found" error to the user rather
/// than guessing a path that might not work.
public enum TesseraCLIBinaryResolver {

    /// Known install locations, in preference order. The build-cli/bin path
    /// is the developer-source build output (Worker 1's foundation branch
    /// produces it); build/bin is the legacy llama.cpp layout. The
    /// xcframework entry is unlikely to exist but is checked because the
    /// xcframework script can optionally stage the binary inside the
    /// framework's Resources directory.
    ///
    /// Mutable so tests can stage a synthetic directory and verify the
    /// auto-discover path without having to match a real $HOME layout.
    public static var knownLocations: [String] = [
        "\(NSHomeDirectory())/Developer/GitHub/tessera/build-cli/bin/tessera-cli",
        "\(NSHomeDirectory())/Developer/GitHub/tessera/build/bin/tessera-cli",
        "\(NSHomeDirectory())/Developer/GitHub/tessera/artifacts/tessera.xcframework/macos-arm64/tessera.framework/Resources/tessera-cli",
        "/opt/homebrew/bin/tessera-cli",
        "/usr/local/bin/tessera-cli",
    ]

    public enum ResolvedPath: Equatable, Sendable {
        case found(String)
        case notFound(searched: [String])
    }

    public enum ResolverError: Error, LocalizedError {
        case binaryNotFound(searched: [String])
        public var errorDescription: String? {
            switch self {
            case .binaryNotFound(let searched):
                return "tessera-cli binary not found; checked: \(searched.joined(separator: ", "))"
            }
        }
    }

    /// Resolve the binary path. The override is the caller's best guess
    /// (typically the user's settings value); if non-empty and executable
    /// it short-circuits the search. Otherwise the settings key is read
    /// and then the known locations are walked in order. Falls through to
    /// `which tessera-cli` via `$PATH` last.
    public static func resolve(
        override: String? = nil,
        settingsKey: String? = nil,
        isExecutable: (String) -> Bool = FileManager.default.isExecutableFile(atPath:)
    ) -> String? {
        if let path = firstExecutable(override, isExecutable: isExecutable) {
            return path
        }
        if let key = settingsKey,
           let stored = UserDefaults.standard.string(forKey: key),
           let path = firstExecutable(stored, isExecutable: isExecutable) {
            return path
        }
        if let path = knownLocations.first(where: isExecutable) {
            return path
        }
        return firstExecutable(pathSearch(), isExecutable: isExecutable)
    }

    /// Same precedence as `resolve`, but returns the list of locations that
    /// were tried so the Settings view can report the honest "not found"
    /// state with context. The override (if non-empty) is always included
    /// even when nothing matches, so the user sees what the tool attempted.
    public static func resolvedPathOrDiagnostic(
        override: String? = nil,
        settingsKey: String? = nil,
        isExecutable: (String) -> Bool = FileManager.default.isExecutableFile(atPath:),
        pathLookup: () -> String? = { pathSearch() }
    ) -> ResolvedPath {
        var searched: [String] = []
        let candidates: [String?] = [
            override,
            settingsKey.flatMap { UserDefaults.standard.string(forKey: $0) },
        ] + knownLocations + [pathLookup()]
        for c in candidates {
            guard let path = c, !path.isEmpty else { continue }
            searched.append(path)
            if isExecutable(path) { return .found(path) }
        }
        return .notFound(searched: searched)
    }

    // MARK: helpers

    private static func firstExecutable(
        _ candidate: String?,
        isExecutable: (String) -> Bool
    ) -> String? {
        guard let path = candidate?.trimmingCharacters(in: .whitespacesAndNewlines),
              !path.isEmpty, isExecutable(path) else { return nil }
        return path
    }

    /// Equivalent of `which tessera-cli`: walks `$PATH` directories and
    /// returns the first entry that names an executable `tessera-cli`. Kept
    /// here (instead of shelling out) so tests can stub it via the
    /// `pathLookup` parameter on `resolvedPathOrDiagnostic`. Public so it can
    /// be used as a default argument value.
    public static func pathSearch() -> String? {
        let env = ProcessInfo.processInfo.environment["PATH"] ?? ""
        let dirs = env.split(separator: ":").map(String.init)
        for dir in dirs {
            let candidate = (dir as NSString).appendingPathComponent("tessera-cli")
            if FileManager.default.isExecutableFile(atPath: candidate) {
                return candidate
            }
        }
        return nil
    }
}
