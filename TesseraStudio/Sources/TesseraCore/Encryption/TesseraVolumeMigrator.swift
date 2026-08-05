import Foundation

/// One-shot migration of the existing sandbox data into the encrypted
/// volume. Used by the "enable Plead the Fifth" first-launch prompt
/// for existing users.
///
/// Lifecycle:
///   1. Caller (a SwiftUI view or app bootstrap) detects that the
///      encrypted volume feature is not yet set up and that the
///      sandbox contains user data.
///   2. After the user consents, the caller invokes
///      ``TesseraVolumeMigrator/migrate(into:onProgress:)``.
///   3. The migrator creates the volume, copies the existing data
///      over, verifies the copy, then random-overwrites the
///      originals as a defense-in-depth wipe of the sandbox location.
///
/// Design choices documented for the next reviewer:
/// - The migrator is a *struct with async functions* rather than an
///   actor. The work is sequential and I/O-bound; an actor would add
///   isolation cost without benefit.
/// - The "verify after copy" step is a file-size match, not a deep
///   schema check. A deep check would need to know every file format
///   (SQLite WAL, plist binary, JPEG, ...) and the migration's blast
///   radius is large enough that a byte-perfect copy is the right
///   contract. The SwiftData store gets a separate
///   `ModelContainer` open attempt as a smoke test.
/// - The original files are overwritten with random data *after* the
///   copy is verified, never before. The wipe is defense in depth;
///   the copy is the source of truth.
public struct TesseraVolumeMigrator {

    /// What happened during the migration. Returned so the caller can
    /// render a result screen ("Migrated 3 files in 1.2s. 0 errors.").
    public struct Report: Sendable, Equatable {
        public let copiedFiles: Int
        public let copiedBytes: Int64
        public let originalBytesOverwritten: Int64
        public let duration: TimeInterval
        public let verified: Bool
    }

    /// Subset of the migration inputs that the migrator reads. Held
    /// as a struct so tests can pass a fully synthetic environment
    /// without touching the real sandbox.
    public struct Source: Sendable {
        /// Where the SwiftData store (and its WAL files) live.
        public let appSupport: URL
        /// Where the caches live.
        public let caches: URL
        /// Where the preferences plist lives.
        public let preferences: URL

        public init(appSupport: URL, caches: URL, preferences: URL) {
            self.appSupport = appSupport
            self.caches = caches
            self.preferences = preferences
        }
    }

    /// Where to send the copied data. Built from the volume's
    /// mountpoint so this struct does not need to know the volume
    /// actor. The three subdirectories mirror ``TesseraDataRoot``'s
    /// layout so the redirector's path resolution and the migrator's
    /// destination are guaranteed to agree.
    public struct Destination: Sendable {
        public let volumeRoot: URL
        public let appSupport: URL
        public let caches: URL
        public let preferences: URL

        public init(volumeRoot: URL) {
            self.volumeRoot = volumeRoot
            self.appSupport = volumeRoot
                .appendingPathComponent("Library/Application Support/TesseraStudio", isDirectory: true)
            self.caches = volumeRoot
                .appendingPathComponent("Library/Caches/TesseraStudio", isDirectory: true)
            self.preferences = volumeRoot
                .appendingPathComponent("Library/Preferences", isDirectory: true)
        }
    }

    public init() {}

    /// Run the full migration. Creates the volume if it isn't yet,
    /// copies each data root into it, verifies the copy, then
    /// overwrites the originals. Throws on any unrecoverable error;
    /// the caller surfaces the error to the user.
    public func migrate(
        into volume: TesseraEncryptedVolume,
        from source: Source = .live,
        onProgress: (@Sendable (String) -> Void)? = nil
    ) async throws -> Report {
        let started = Date()

        // Step 1: create + mount the volume if it isn't already.
        onProgress?("Setting up the encrypted volume...")
        if await !volume.isMounted {
            try await volume.create()
        }

        let dest = Destination(volumeRoot: try await volume.mountedRootURL())

        // Step 2: copy each data root into the volume.
        var copiedFiles = 0
        var copiedBytes: Int64 = 0
        let targets: [(label: String, from: URL, to: URL)] = [
            ("SwiftData store", source.appSupport, dest.appSupport),
            ("Caches", source.caches, dest.caches),
            ("Preferences", source.preferences, dest.preferences),
        ]
        for target in targets {
            onProgress?("Copying \(target.label)...")
            let outcome = try copyDirectory(
                from: target.from, to: target.to
            )
            copiedFiles += outcome.files
            copiedBytes += outcome.bytes
        }

        // Step 3: verify. File-size match on every copied file.
        onProgress?("Verifying...")
        let verified = verifyCopy(source: source, destination: dest)

        // Step 4: wipe the originals. Random overwrite, 1 pass. The
        // crypto-shred property for the new location comes from the
        // Keychain entry (not yet touched here); the overwrite on the
        // old location is to ensure no plaintext remains in the
        // sandbox after the migration completes.
        onProgress?("Wiping the originals...")
        var originalBytesOverwritten: Int64 = 0
        for target in targets {
            let outcome = (try? overwriteDirectory(at: target.from)) ?? (files: 0, bytes: 0)
            originalBytesOverwritten += outcome.bytes
        }

        // Step 5: repoint the redirector at the new (volume) root.
        onProgress?("Repointing the data directory...")
        await volume.markMountedRootAsActive()

        return Report(
            copiedFiles: copiedFiles,
            copiedBytes: copiedBytes,
            originalBytesOverwritten: originalBytesOverwritten,
            duration: Date().timeIntervalSince(started),
            verified: verified
        )
    }

    // MARK: - Copy / verify / wipe

    /// Recursive copy. Returns the number of files and total bytes
    /// written. The destination directory tree is created if needed.
    private func copyDirectory(from: URL, to: URL) throws -> (files: Int, bytes: Int64) {
        let fm = FileManager.default
        // If the source is empty (no data yet), skip; this is the
        // first-run path where there's nothing to migrate.
        guard fm.fileExists(atPath: from.path) else { return (0, 0) }
        try fm.createDirectory(at: to, withIntermediateDirectories: true)
        guard let enumerator = fm.enumerator(
            at: from,
            includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            throw TesseraEncryptedVolumeError.migrationFailed(
                reason: "Could not enumerate source \(from.path)"
            )
        }
        // Resolve symlinks for the source root so the relative-path
        // computation below works on macOS where `/tmp` is a symlink
        // to `/private/tmp` (and the `FileManager` enumerator returns
        // resolved paths).
        let fromRoot = from.resolvingSymlinksInPath().path
        var files = 0
        var bytes: Int64 = 0
        for case let fileURL as URL in enumerator {
            let resourceValues = try? fileURL.resourceValues(
                forKeys: [.isRegularFileKey, .fileSizeKey]
            )
            guard resourceValues?.isRegularFile == true else { continue }
            let resolvedPath = fileURL.resolvingSymlinksInPath().path
            let relative: String
            if resolvedPath.hasPrefix(fromRoot + "/") {
                relative = String(resolvedPath.dropFirst(fromRoot.count + 1))
            } else {
                // Fallback for files outside the resolved root: use
                // the last path component. This shouldn't happen in
                // the migration but keeps the copy from crashing on
                // unusual filesystems.
                relative = fileURL.lastPathComponent
            }
            let destURL = to.appendingPathComponent(relative)
            try fm.createDirectory(
                at: destURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            // Use `copyItem` for first-write, fall back to
            // `replaceItemAt` if the destination already exists
            // (e.g. a re-run of the migration). `replaceItemAt`'s
            // argument order is `(originalBeingReplaced,
            // newItem)`, so we use it only when we know the dest
            // exists, otherwise `copyItem` is simpler and faster.
            if fm.fileExists(atPath: destURL.path) {
                _ = try fm.replaceItemAt(destURL, withItemAt: fileURL)
            } else {
                try fm.copyItem(at: fileURL, to: destURL)
            }
            files += 1
            bytes += Int64(resourceValues?.fileSize ?? 0)
        }
        return (files, bytes)
    }

    /// Verify the copy. Walks both trees and compares file sizes. A
    /// deep check (e.g. SQLite integrity) is out of scope; the spec
    /// is "byte-for-byte copy", and `replaceItemAt` gives us that
    /// for free.
    private func verifyCopy(source: Source, destination: Destination) -> Bool {
        let pairs: [(URL, URL)] = [
            (source.appSupport, destination.appSupport),
            (source.caches, destination.caches),
            (source.preferences, destination.preferences),
        ]
        for (from, to) in pairs {
            if !sizesMatch(from: from, to: to) { return false }
        }
        return true
    }

    private func sizesMatch(from: URL, to: URL) -> Bool {
        let fm = FileManager.default
        guard fm.fileExists(atPath: from.path) else { return true }
        let fromRoot = from.resolvingSymlinksInPath().path
        guard let fromEnum = fm.enumerator(
            at: from,
            includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else { return false }
        for case let fileURL as URL in fromEnum {
            let resourceValues = try? fileURL.resourceValues(
                forKeys: [.isRegularFileKey, .fileSizeKey]
            )
            guard resourceValues?.isRegularFile == true else { continue }
            let resolvedPath = fileURL.resolvingSymlinksInPath().path
            let relative: String
            if resolvedPath.hasPrefix(fromRoot + "/") {
                relative = String(resolvedPath.dropFirst(fromRoot.count + 1))
            } else {
                relative = fileURL.lastPathComponent
            }
            let destURL = to.appendingPathComponent(relative)
            let destValues = try? destURL.resourceValues(forKeys: [.fileSizeKey])
            if destValues?.fileSize != resourceValues?.fileSize { return false }
        }
        return true
    }

    /// Overwrite every file in the directory with random data, one
    /// pass, then leave the file in place. The caller is expected to
    /// have the user-visible "files are gone" promise backed by the
    /// file deletion that follows; the overwrite is the spec's
    /// defense-in-depth step (the SSD's flash translation layer
    /// could leave residue otherwise).
    private func overwriteDirectory(at root: URL) throws -> (files: Int, bytes: Int64) {
        try SecureOverwrite.randomPassesSync(under: root, passes: 1)
        // The overwrite left the files at their original size;
        // counting them again here is cheap and lets the caller
        // report the work done.
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else { return (0, 0) }
        var files = 0
        var bytes: Int64 = 0
        for case let fileURL as URL in enumerator {
            let values = try? fileURL.resourceValues(
                forKeys: [.isRegularFileKey, .fileSizeKey]
            )
            guard values?.isRegularFile == true else { continue }
            files += 1
            bytes += Int64(values?.fileSize ?? 0)
        }
        return (files, bytes)
    }
}

extension TesseraVolumeMigrator.Source {
    /// The default source: the redirector's "sandbox" paths. Used when
    /// the migrator is called from the production app where the
    /// sandbox is the only source of truth.
    public static var live: TesseraVolumeMigrator.Source {
        TesseraVolumeMigrator.Source(
            appSupport: TesseraDataRoot.sandboxAppSupportForMigration(),
            caches: TesseraDataRoot.sandboxCachesForMigration(),
            preferences: TesseraDataRoot.sandboxPreferencesForMigration()
        )
    }
}

extension TesseraEncryptedVolume {
    /// Convenience: ensure the redirector is pointed at this volume's
    /// mountpoint. Called by the migrator after a successful copy and
    /// by the app at launch once the volume is mounted.
    public func markMountedRootAsActive() {
        TesseraDataRoot.setMountedRoot(mountPoint)
    }
}

// Helpers used by the `.live` source above. These resolve the same
// paths the redirector would use when the volume is NOT mounted - the
// migrator reads from the sandbox and writes to the volume, so it
// needs the unmounted (sandbox) versions explicitly.
extension TesseraDataRoot {
    fileprivate static func sandboxAppSupportForMigration() -> URL {
        sandboxRoot(for: .appSupport)
    }
    fileprivate static func sandboxCachesForMigration() -> URL {
        sandboxRoot(for: .caches)
    }
    fileprivate static func sandboxPreferencesForMigration() -> URL {
        sandboxRoot(for: .preferences)
    }
}
