import Foundation

/// Resolves "where should Tessera put its data?" for every consumer in
/// the app: SwiftData's `ModelConfiguration`, the URL session cache,
/// the UserDefaults plist, and (once those land) the Postgres / Valkey
/// / DuckDB data directories.
///
/// While the encrypted volume is mounted, every data path is rooted at
/// the volume's mount point. When the volume is not mounted (first
/// launch before the migration runs, recovery flow after a wipe), the
/// redirector falls back to the original sandbox path so the rest of
/// the app continues to work; the data sitting at the fallback path is
/// either the pre-migration sandbox copy (about to be migrated) or
/// the empty state the user explicitly chose by unmounting.
///
/// Why an enum and not a struct: the redirector is purely a path
/// resolver. It holds no state, has no init, and is called from many
/// call sites (SwiftData container init, settings view, Postgres
/// sidecar boot, ...). An enum-of-functions keeps the API obvious:
/// every consumer just writes `TesseraDataRoot.appSupport()` and gets
/// the right URL.
public enum TesseraDataRoot {

    // MARK: - Public API

    /// Where the SwiftData store lives. Passed to
    /// `ModelConfiguration(url:)` as the storage URL.
    public static func appSupport() -> URL {
        insideVolume(
            subdirectory: "Library/Application Support/TesseraStudio",
            fallbackFile: "default.store"
        )
    }

    /// Where the SwiftData store *file* lives. Separate from
    /// ``appSupport()`` so the SwiftData container can pass the file
    /// directly to `ModelConfiguration(url:)`.
    public static func appSupportStoreFile() -> URL {
        appSupport().appendingPathComponent("default.store")
    }

    /// The cache directory. Used by `URLCache` and any custom caches
    /// Tessera keeps.
    public static func caches() -> URL {
        insideVolume(
            subdirectory: "Library/Caches/TesseraStudio",
            fallbackFile: nil
        )
    }

    /// The Preferences directory. Used by `UserDefaults` when the app
    /// writes its plist inside the volume (the migration step moves
    /// the existing `com.tessera.studio.mac.plist` here).
    public static func preferences() -> URL {
        insideVolume(
            subdirectory: "Library/Preferences",
            fallbackFile: nil
        )
    }

    /// Postgres data directory. Reserved for the post-Materials slice
    /// (not in this build); the path is wired up now so the redirector
    /// is the single source of truth for "where do data files go".
    public static func postgresDataDir() -> URL {
        insideVolume(subdirectory: "postgres/data", fallbackFile: nil)
    }

    /// Valkey data directory. Same as Postgres - reserved for the
    /// sidecar that ships in a later slice.
    public static func valkeyDataDir() -> URL {
        insideVolume(subdirectory: "valkey/data", fallbackFile: nil)
    }

    /// DuckDB file. One file, lives at the volume root for easy
    /// access during the Materials slice.
    public static func duckdbFile() -> URL {
        insideVolume(subdirectory: "duckdb", fallbackFile: "tessera.duckdb")
    }

    /// Whether the redirector is currently pointing at the encrypted
    /// volume. Views use this to decide whether to show the "your
    /// data is encrypted" badge or the "set up the encrypted volume"
    /// prompt.
    public static func isUsingEncryptedVolume() -> Bool {
        volumeRootIfMounted() != nil
    }

    // MARK: - Implementation

    /// The path the redirector uses when the volume is NOT mounted.
    /// In the production macOS app this is the sandbox's
    /// `Library/Application Support/TesseraStudio/`; tests can
    /// override it via ``setSandboxRoot(_:)`` to use a tmp directory.
    ///
    /// We resolve `FileManager.default.urls(for: .applicationSupportDirectory)`
    /// lazily (not as a stored property) so a test that mounts the
    /// volume after launch sees the new path on its next call.
    ///
    /// Public so the migrator's `.live` source can read the same
    /// paths the redirector would, so the override is honored by
    /// both the redirector and the migration flow.
    public static func sandboxRoot(for kind: SubdirectoryKind) -> URL {
        let override: URL?
        switch kind {
        case .appSupport: override = _appSupportSandboxOverride
        case .caches: override = _cachesSandboxOverride
        case .preferences: override = _preferencesSandboxOverride
        }
        if let override { return override }

        let fm = FileManager.default
        switch kind {
        case .appSupport:
            let base = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
                ?? URL(fileURLWithPath: NSTemporaryDirectory())
            return base.appendingPathComponent("TesseraStudio", isDirectory: true)
        case .caches:
            let base = fm.urls(for: .cachesDirectory, in: .userDomainMask).first
                ?? URL(fileURLWithPath: NSTemporaryDirectory())
            return base.appendingPathComponent("TesseraStudio", isDirectory: true)
        case .preferences:
            let base = fm.urls(for: .libraryDirectory, in: .userDomainMask).first
                ?? URL(fileURLWithPath: NSTemporaryDirectory())
            return base.appendingPathComponent("Preferences", isDirectory: true)
        }
    }

    /// Subdirectory under the user library. Public so tests can
    /// target the right override slot.
    public enum SubdirectoryKind: Sendable {
        case appSupport, caches, preferences
    }

    /// Compute a path inside the mounted volume, or fall back to the
    /// sandbox path if the volume is not currently mounted. The
    /// `subdirectory` is appended to the volume root; `fallbackFile`,
    /// if set, is appended to the sandbox root (so callers that want
    /// a file URL don't have to re-build the path).
    private static func insideVolume(subdirectory: String, fallbackFile: String?) -> URL {
        if let volumeRoot = volumeRootIfMounted() {
            return volumeRoot.appendingPathComponent(subdirectory, isDirectory: true)
        }
        let sandbox = sandboxRoot(for: sandboxKind(for: subdirectory))
        if let file = fallbackFile {
            return sandbox.appendingPathComponent(file, isDirectory: false)
        }
        return sandbox
    }

    private static func sandboxKind(for subdirectory: String) -> SubdirectoryKind {
        if subdirectory.hasPrefix("Library/Caches") { return .caches }
        if subdirectory.hasPrefix("Library/Preferences") { return .preferences }
        return .appSupport
    }

    // MARK: - Volume root

    /// Holder for the volume's mounted root. Set by the app at launch
    /// once the encrypted volume is mounted; cleared when the volume
    /// is unmounted. Held on a class so the redirector (an enum) can
    /// mutate the value via a small final-class box, mirroring the
    /// pattern used in ``TesseraEngineContext``.
    private final class MountRoot {
        var url: URL?
        init(_ url: URL?) { self.url = url }
    }

    private static let mountRootBox = MountRoot(nil)

    /// Set the mounted volume root. The app's launch sequence calls
    /// this once the volume is mounted; the unmount path clears it.
    /// Tests call it directly to simulate the mounted state without
    /// invoking hdiutil.
    public static func setMountedRoot(_ url: URL?) {
        mountRootBox.url = url
    }

    /// Internal accessor used by the app's volume wiring (see
    /// `TesseraAppLifecycle` for the production integration).
    public static func mountedRoot() -> URL? {
        mountRootBox.url
    }

    /// Convenience that returns the mounted root only when it's set.
    /// Used by ``insideVolume(subdirectory:fallbackFile:)`` to decide
    /// between the volume path and the sandbox path.
    private static func volumeRootIfMounted() -> URL? {
        mountRootBox.url
    }

    // MARK: - Test overrides

    /// Override the sandbox app-support root. Tests use this to point
    /// the redirector at a tmp directory so the migration can be
    /// exercised without writing to the user's real sandbox.
    public static func setSandboxRoot(_ url: URL, for kind: SubdirectoryKind) {
        switch kind {
        case .appSupport: _appSupportSandboxOverride = url
        case .caches: _cachesSandboxOverride = url
        case .preferences: _preferencesSandboxOverride = url
        }
    }

    private static var _appSupportSandboxOverride: URL?
    private static var _cachesSandboxOverride: URL?
    private static var _preferencesSandboxOverride: URL?
}
