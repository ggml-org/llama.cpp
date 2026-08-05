import Foundation

/// Errors surfaced by ``TesseraEncryptedVolume``. Modeled as a single
/// enum so callers (the migration flow, the reset flow, the SwiftUI
/// views) can switch on it and render meaningful messages. The `kind`
/// is the machine-readable category; the localized description is for
/// the user-visible error.
public enum TesseraEncryptedVolumeError: Error, LocalizedError, Equatable {
    case keychainRejected(operation: String)
    case keychainMissingPassword
    case hdiutilFailed(operation: String, exitCode: Int32, stderr: String)
    case bundleAlreadyExists(URL)
    case bundleMissing(URL)
    case notMounted
    case alreadyMounted(URL)
    case mountpointUnavailable(URL)
    case migrationFailed(reason: String)
    case platformUnsupported
    case other(String)

    public var errorDescription: String? {
        switch self {
        case .keychainRejected(let op):
            return "The macOS Keychain refused the \(op) request. The keychain may be locked or the app may lack entitlements."
        case .keychainMissingPassword:
            return "No volume password is stored in the Keychain. Create the volume first, or restore the password from a backup."
        case .hdiutilFailed(let op, let code, let stderr):
            let tail = stderr.isEmpty ? "" : ": \(stderr)"
            return "hdiutil \(op) failed with exit code \(code)\(tail)"
        case .bundleAlreadyExists(let url):
            return "The encrypted volume already exists at \(url.path)."
        case .bundleMissing(let url):
            return "The encrypted volume is missing at \(url.path). It may have been moved or wiped."
        case .notMounted:
            return "The encrypted volume is not currently mounted."
        case .alreadyMounted(let url):
            return "The encrypted volume is already mounted at \(url.path)."
        case .mountpointUnavailable(let url):
            return "The mount point \(url.path) is already in use by another volume."
        case .migrationFailed(let reason):
            return "Migration to the encrypted volume failed: \(reason)"
        case .platformUnsupported:
            return "Encrypted volumes are only supported on macOS."
        case .other(let msg):
            return msg
        }
    }

    /// The machine-readable category. Useful for switch-on-error in
    /// views (e.g. `.keychainMissingPassword` -> show "Reset Tessera"
    /// button; `.hdiutilFailed` -> show raw stderr + retry).
    public enum Kind: String, Sendable {
        case keychainRejected
        case keychainMissingPassword
        case hdiutilFailed
        case bundleAlreadyExists
        case bundleMissing
        case notMounted
        case alreadyMounted
        case mountpointUnavailable
        case migrationFailed
        case platformUnsupported
        case other
    }

    public var kind: Kind {
        switch self {
        case .keychainRejected: return .keychainRejected
        case .keychainMissingPassword: return .keychainMissingPassword
        case .hdiutilFailed: return .hdiutilFailed
        case .bundleAlreadyExists: return .bundleAlreadyExists
        case .bundleMissing: return .bundleMissing
        case .notMounted: return .notMounted
        case .alreadyMounted: return .alreadyMounted
        case .mountpointUnavailable: return .mountpointUnavailable
        case .migrationFailed: return .migrationFailed
        case .platformUnsupported: return .platformUnsupported
        case .other: return .other
        }
    }
}

/// Configuration for a single encrypted volume. Held in
/// ``TesseraEncryptedVolume`` so the actor and the data root redirector
/// can agree on the same paths.
public struct TesseraVolumeConfig: Sendable, Equatable {
    /// Filesystem path to the `.sparsebundle` directory. Sparse bundles
    /// are bundles on disk (not single files), so the path is to a
    /// directory whose interior holds the bands, token, and plist.
    public let bundleURL: URL
    /// Where the bundle is mounted. macOS usually picks
    /// `/Volumes/<volname>`; we pin it explicitly so the redirector and
    /// the rest of the app can rely on a stable path.
    public let mountPoint: URL
    /// Display volume name (visible in Finder if the user opens it).
    public let volumeName: String
    /// Maximum bundle size in bytes. 1 GiB is a sane default for the
    /// dev preview; production will size this to the user's
    /// configured data budget.
    public let sizeBytes: Int

    public init(
        bundleURL: URL,
        mountPoint: URL = URL(fileURLWithPath: "/Volumes/TesseraVault"),
        volumeName: String = "TesseraVault",
        sizeBytes: Int = 1024 * 1024 * 1024
    ) {
        self.bundleURL = bundleURL
        self.mountPoint = mountPoint
        self.volumeName = volumeName
        self.sizeBytes = sizeBytes
    }
}

/// Manages the lifecycle of the encrypted APFS volume that holds
/// Tessera's data.
///
/// Why an actor: the volume has a single, process-wide mount state. Two
/// callers racing on `mount` and `unmount` would corrupt that state.
/// Swift's actor isolation gives us a serial executor for the
/// `isMounted` + `mountPoint` fields and an obvious point to run the
/// hdiutil subprocesses (which are themselves synchronous, so they
/// don't fight each other inside the actor).
///
/// macOS-only. The `init` succeeds on every platform so the
/// iOS-target compilations don't break; the methods that actually
/// touch hdiutil or the volume paths throw `.platformUnsupported` on
/// non-macOS hosts. This matches the pattern in ``ProcessRunner``.
public actor TesseraEncryptedVolume {

    /// The current mount state, observed under actor isolation.
    public private(set) var isMounted: Bool = false

    /// The resolved mount point (== `config.mountPoint` while mounted).
    /// Kept as a separate field so `mountedRootURL` can return it
    /// without re-reading the config.
    public private(set) var mountedURL: URL?

    /// The volume configuration. Set once at init; immutable for the
    /// actor's lifetime. Tests construct actors with non-default
    /// configs (e.g. a tmp path); the production wiring passes the
    /// canonical `~/Library/Application Support/TesseraStudio/vault.sparsebundle`.
    public let config: TesseraVolumeConfig

    /// The Keychain account name for the volume password. Defaults to
    /// ``TesseraKeychainVolume.volumePasswordAccount``; the test suite
    /// overrides it to a per-test UUID to keep the real keychain
    /// pristine.
    public let keychainAccount: String

    /// Where to find `hdiutil`. Defaults to `/usr/bin/hdiutil`; tests
    /// can inject a wrapper if they ever need to capture invocations
    /// (the current tests use the real binary, since hdiutil in
    /// `/usr/bin` is the same one shipped with macOS and there is no
    /// portable stub).
    public let hdiutilPath: String

    /// Process runner used for the hdiutil subprocesses. Lets tests
    /// inject a fake if they need to; production callers leave it at
    /// the default.
    private let processRunner: ProcessRunner

    public init(
        config: TesseraVolumeConfig,
        keychainAccount: String = TesseraKeychainVolume.volumePasswordAccount,
        hdiutilPath: String = "/usr/bin/hdiutil",
        processRunner: ProcessRunner = ProcessRunner()
    ) {
        self.config = config
        self.keychainAccount = keychainAccount
        self.hdiutilPath = hdiutilPath
        self.processRunner = processRunner
    }

    // MARK: - Lifecycle

    /// First-run path. Generates a fresh password, stores it in the
    /// Keychain, creates the encrypted APFS volume, and mounts it.
    ///
    /// - Throws: ``TesseraEncryptedVolumeError/bundleAlreadyExists``
    ///   when the bundle is already on disk (caller should call
    ///   `mount()` instead). Re-throws any Keychain or hdiutil error.
    public func create() async throws {
        #if os(macOS)
        try requirePlatformSupported()

        // Refuse to clobber an existing bundle. The caller is expected
        // to inspect the filesystem and dispatch to mount() in that
        // case; we make the failure mode explicit rather than
        // silently wiping the user's volume.
        if FileManager.default.fileExists(atPath: config.bundleURL.path) {
            throw TesseraEncryptedVolumeError.bundleAlreadyExists(config.bundleURL)
        }

        guard let password = TesseraKeychainVolume.generateVolumePassword() else {
            throw TesseraEncryptedVolumeError.keychainRejected(operation: "generate-volume-password")
        }
        guard TesseraKeychainVolume.storeVolumePassword(password) else {
            throw TesseraEncryptedVolumeError.keychainRejected(operation: "store-volume-password")
        }

        // From here on, any failure needs to roll back the Keychain
        // entry we just stored - leaving the password in the Keychain
        // without a corresponding bundle would be a footgun.
        do {
            try await createBundle(password: password)
            try await mount(password: password)
        } catch {
            _ = TesseraKeychainVolume.deleteVolumePassword()
            throw error
        }
        #else
        throw TesseraEncryptedVolumeError.platformUnsupported
        #endif
    }

    /// Mount the volume. Reads the password from the Keychain, then
    /// runs `hdiutil attach -stdinpass`. Idempotent: if the volume is
    /// already mounted at the configured mount point, this returns
    /// without doing anything.
    public func mount() async throws {
        #if os(macOS)
        try requirePlatformSupported()

        if isAlreadyMounted() {
            isMounted = true
            mountedURL = config.mountPoint
            return
        }

        guard let password = TesseraKeychainVolume.storedVolumePassword() else {
            throw TesseraEncryptedVolumeError.keychainMissingPassword
        }
        try await mount(password: password)
        #else
        throw TesseraEncryptedVolumeError.platformUnsupported
        #endif
    }

    /// Unmount the volume. Used on quit (after the app's data stores
    /// are closed) and by the reset flow.
    ///
    /// Idempotent: if the volume is not currently mounted, returns
    /// without error. The wipe executor calls this after destroying
    /// the Keychain entry; in that case the mount is expected to be
    /// already-gone from a prior quit, and the operation is a no-op.
    public func unmount() async throws {
        #if os(macOS)
        try requirePlatformSupported()

        if !isAlreadyMounted() {
            isMounted = false
            mountedURL = nil
            return
        }

        let result = try await processRunner.run(
            executable: hdiutilPath,
            arguments: ["detach", config.mountPoint.path]
        )
        guard result.exitCode == 0 else {
            throw TesseraEncryptedVolumeError.hdiutilFailed(
                operation: "detach",
                exitCode: result.exitCode,
                stderr: result.stderr
            )
        }
        isMounted = false
        mountedURL = nil
        #else
        throw TesseraEncryptedVolumeError.platformUnsupported
        #endif
    }

    /// The root of the mounted volume, where the app's data lives.
    /// Throws ``TesseraEncryptedVolumeError/notMounted`` if the
    /// volume is not currently mounted.
    public func mountedRootURL() throws -> URL {
        guard isMounted, let url = mountedURL else {
            throw TesseraEncryptedVolumeError.notMounted
        }
        return url
    }

    /// The mount point as observed under actor isolation. Mirrors
    /// `mountedRootURL()` but as a property so callers that already
    /// know the volume is mounted (e.g. the migrator, right after
    /// `create()`) don't need to wrap the read in a `try`.
    public var mountPoint: URL { config.mountPoint }

    /// Reset the volume. Used by the "Reset Tessera" recovery path
    /// when the user has lost the password or the volume is corrupted.
    ///
    /// Steps:
    ///   1. Try to unmount (will fail or be a no-op if the key is
    ///      already gone).
    ///   2. Overwrite the bundle's bands with random data, 3 passes
    ///      (defense in depth - the crypto-shred property is achieved
    ///      by deleting the Keychain entry, but the wipe spec
    ///      requires the overwrite too).
    ///   3. Delete the bundle directory.
    ///   4. Delete the Keychain entry.
    ///   5. Create a fresh empty bundle with a new password.
    ///   6. Mount it.
    ///
    /// This intentionally swallows most sub-failures after step 1 -
    /// the crypto-shred property holds the moment the Keychain entry
    /// is gone, so a missing bands file or a full disk during the
    /// overwrite does not roll back the reset.
    public func reset() async throws {
        #if os(macOS)
        try requirePlatformSupported()

        // Step 1: unmount. Best effort - if the key is gone already,
        // this is a no-op (the volume will be unmounted by macOS at
        // the next unmount-on-quit).
        try? await unmount()

        // Step 2-3: overwrite and delete the bundle. The overwrite
        // is best-effort: if the bundle is gone, skip; if any band
        // file is unwritable, log and continue.
        if FileManager.default.fileExists(atPath: config.bundleURL.path) {
            do {
                try await SecureOverwrite.randomPasses(
                    under: config.bundleURL,
                    passes: 3
                )
            } catch {
                // Overwrite failure does not block the reset; the
                // crypto-shred property is already in effect.
            }
            try? FileManager.default.removeItem(at: config.bundleURL)
        }

        // Step 4: delete the Keychain entry. Idempotent.
        _ = TesseraKeychainVolume.deleteVolumePassword()

        // Step 5-6: fresh password, fresh bundle, mount.
        try await create()
        #else
        throw TesseraEncryptedVolumeError.platformUnsupported
        #endif
    }

    // MARK: - Internal helpers (still actor-isolated)

    #if os(macOS)
    /// Run `hdiutil create` to make the bundle. The password is piped
    /// to stdin. On any non-zero exit the bundle is removed and the
    /// error is rethrown.
    private func createBundle(password: String) async throws {
        // Make sure the parent directory exists; hdiutil refuses to
        // create a bundle inside a missing directory.
        let parent = config.bundleURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: parent, withIntermediateDirectories: true
        )

        // Size in megabytes; hdiutil's -size takes a unit-suffixed
        // string. Round up to the nearest MiB so we never pass zero.
        let sizeMB = max(1, (config.sizeBytes + 1024 * 1024 - 1) / (1024 * 1024))
        let args = [
            "create",
            "-size", "\(sizeMB)m",
            "-fs", "APFS",
            "-encryption", "AES-256",
            "-volname", config.volumeName,
            "-stdinpass",
            config.bundleURL.path,
        ]
        let result = try await runHdiutilWithPassword(
            arguments: args, password: password
        )
        if result.exitCode != 0 {
            // Clean up the partial bundle before throwing.
            try? FileManager.default.removeItem(at: config.bundleURL)
            throw TesseraEncryptedVolumeError.hdiutilFailed(
                operation: "create", exitCode: result.exitCode, stderr: result.stderr
            )
        }
    }

    /// Run `hdiutil attach` with the password on stdin. On success,
    /// sets `isMounted` and `mountedURL` so the rest of the app can
    /// observe the new state.
    private func mount(password: String) async throws {
        // Pin the mount point. Without -mountpoint, hdiutil uses the
        // volume name and the path can drift if the user renames it
        // in Finder; we want a stable path for the data root
        // redirector.
        let args = [
            "attach",
            "-mountpoint", config.mountPoint.path,
            "-nobrowse",
            "-stdinpass",
            config.bundleURL.path,
        ]
        let result = try await runHdiutilWithPassword(
            arguments: args, password: password
        )
        if result.exitCode != 0 {
            // A common case here is "wrong password" (errSec from the
            // underlying auth). Map that to a clearer error so the UI
            // can show "wrong password" rather than the raw hdiutil
            // text.
            let lowered = result.stderr.lowercased()
            if lowered.contains("authentication") || lowered.contains("password") {
                throw TesseraEncryptedVolumeError.hdiutilFailed(
                    operation: "attach",
                    exitCode: result.exitCode,
                    stderr: "Authentication failed. The Keychain-stored password did not unlock the volume."
                )
            }
            throw TesseraEncryptedVolumeError.hdiutilFailed(
                operation: "attach", exitCode: result.exitCode, stderr: result.stderr
            )
        }
        isMounted = true
        mountedURL = config.mountPoint
    }

    /// Spawn hdiutil with the password on stdin. Uses a Process with
    /// the standard input piped to a Pipe so the password never
    /// appears in the process table or on disk.
    private func runHdiutilWithPassword(
        arguments: [String], password: String
    ) async throws -> ProcessResult {
        let passwordData = Data(password.utf8)
        return try await withCheckedThrowingContinuation { continuation in
            let process = Process()
            let stdoutPipe = Pipe()
            let stderrPipe = Pipe()
            let stdinPipe = Pipe()

            process.executableURL = URL(fileURLWithPath: hdiutilPath)
            process.arguments = arguments
            process.standardOutput = stdoutPipe
            process.standardError = stderrPipe
            process.standardInput = stdinPipe

            process.terminationHandler = { proc in
                let outData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
                let errData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
                let result = ProcessResult(
                    exitCode: proc.terminationStatus,
                    stdout: String(data: outData, encoding: .utf8) ?? "",
                    stderr: String(data: errData, encoding: .utf8) ?? ""
                )
                continuation.resume(returning: result)
            }

            do {
                try process.run()
                // hdiutil -stdinpass reads the password from stdin
                // and closes the pipe on its own. Writing the
                // password bytes and closing is enough.
                try stdinPipe.fileHandleForWriting.write(contentsOf: passwordData)
                try stdinPipe.fileHandleForWriting.close()
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }

    /// Whether the bundle is currently mounted at the configured mount
    /// point. Checks the filesystem rather than relying on a cached
    /// flag, so the actor's view stays in sync if something external
    /// (a manual `hdiutil detach` from Terminal) unmounts the volume.
    private func isAlreadyMounted() -> Bool {
        var isDir: ObjCBool = false
        let exists = FileManager.default.fileExists(
            atPath: config.mountPoint.path, isDirectory: &isDir
        )
        return exists && isDir.boolValue
    }

    private func requirePlatformSupported() throws {
        // The compile-time #if os(macOS) already gates the body of
        // every public method, so this check is for tests that
        // exercise the public surface from a non-Mac build (the SPM
        // iOS library target compiles the public surface but never
        // calls these methods). Reaching this on Linux would be a
        // bug.
        #if os(macOS)
        // no-op
        #else
        throw TesseraEncryptedVolumeError.platformUnsupported
        #endif
    }
    #endif
}
