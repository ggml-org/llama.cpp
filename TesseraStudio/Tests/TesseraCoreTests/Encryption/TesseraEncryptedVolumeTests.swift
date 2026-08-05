import XCTest
@testable import TesseraCore

/// End-to-end tests for the encrypted-volume lifecycle.
///
/// These tests touch three real macOS subsystems:
///  1. The Keychain (via ``TesseraKeychainVolume``).
///  2. `hdiutil` for create / attach / detach.
///  3. The filesystem inside the mounted bundle.
///
/// We use throwaway Keychain accounts (a UUID per test) and a tmp
/// sparse-bundle per test so the real user state is never touched.
/// Cleanup runs in `tearDown` even when the assertion fails.
@MainActor
final class TesseraEncryptedVolumeTests: XCTestCase {

    private var tmpDir: URL!
    private var bundleURL: URL!
    private var mountPoint: URL!
    private var keychainAccount: String!

    override func setUp() async throws {
        try await super.setUp()
        // Use a tmp parent for the bundle; the mount point is a
        // /Volumes path that matches the production layout, but with
        // a per-test name so we never collide with the user's real
        // volume or another test.
        tmpDir = URL(fileURLWithPath: "/tmp/tessera-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: tmpDir, withIntermediateDirectories: true
        )
        bundleURL = tmpDir.appendingPathComponent("vault.sparsebundle")
        mountPoint = URL(fileURLWithPath: "/Volumes/TesseraVault-\(UUID().uuidString.prefix(6))")
        keychainAccount = "test-volume-password-\(UUID().uuidString)"
    }

    override func tearDown() async throws {
        // Always try to unmount; the test may have crashed mid-mount
        // and we want a clean slate for the next run.
        await unmountQuietly()
        _ = TesseraKeychainVolume.deleteVolumePassword()
        if let tmpDir {
            try? FileManager.default.removeItem(at: tmpDir)
        }
        tmpDir = nil
        bundleURL = nil
        mountPoint = nil
        keychainAccount = nil
        try await super.tearDown()
    }

    // MARK: - Lifecycle

    func testCreateAndMount() async throws {
        let volume = makeVolume()
        try await volume.create()

        // isMounted should be true
        let mounted = await volume.isMounted
        XCTAssertTrue(mounted, "volume should be mounted after create()")

        // mountedRootURL() should return the configured mount point
        let root = try await volume.mountedRootURL()
        XCTAssertEqual(root.path, mountPoint.path)

        // The mount point must be a real directory
        var isDir: ObjCBool = false
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: mountPoint.path, isDirectory: &isDir
        ))
        XCTAssertTrue(isDir.boolValue, "mount point should be a directory")

        // Writing a file should persist on remount (covered in a
        // separate test); here we just verify writability.
        let probe = mountPoint.appendingPathComponent("probe.txt")
        try "hello".write(to: probe, atomically: true, encoding: .utf8)
        XCTAssertTrue(FileManager.default.fileExists(atPath: probe.path))

        try await volume.unmount()
        let stillMounted = await volume.isMounted
        XCTAssertFalse(stillMounted)
    }

    func testMountIsIdempotent() async throws {
        let volume = makeVolume()
        try await volume.create()

        // Calling mount() on an already-mounted volume should be a
        // no-op (no error, no second hdiutil invocation).
        try await volume.mount()
        let stillMounted = await volume.isMounted
        XCTAssertTrue(stillMounted)
        try await volume.mount()
        let stillMountedAgain = await volume.isMounted
        XCTAssertTrue(stillMountedAgain)
    }

    func testUnmountAndRemount() async throws {
        let volume = makeVolume()
        try await volume.create()

        // Write a sentinel file, unmount, mount, verify the file
        // survives. This is the key persistence property of the
        // encrypted volume.
        let probe = mountPoint.appendingPathComponent("persistent.txt")
        try "round-trip".write(to: probe, atomically: true, encoding: .utf8)
        let originalContent = try String(contentsOf: probe, encoding: .utf8)
        XCTAssertEqual(originalContent, "round-trip")

        try await volume.unmount()
        let stillMounted = await volume.isMounted
        XCTAssertFalse(stillMounted)

        // Remount with a fresh actor (simulates a process restart
        // where the volume exists on disk and the password is in the
        // Keychain).
        let volume2 = makeVolume()
        try await volume2.mount()
        let remounted = await volume2.isMounted
        XCTAssertTrue(remounted)

        let restored = mountPoint.appendingPathComponent("persistent.txt")
        XCTAssertTrue(FileManager.default.fileExists(atPath: restored.path))
        let restoredContent = try String(contentsOf: restored, encoding: .utf8)
        XCTAssertEqual(restoredContent, "round-trip")
    }

    func testUnmountIsIdempotent() async throws {
        let volume = makeVolume()
        // Unmounting a never-mounted volume should not error.
        try await volume.unmount()
        try await volume.unmount()
    }

    // MARK: - Keychain

    func testKeychainPasswordRetrieval() async throws {
        let volume = makeVolume()
        try await volume.create()

        // The password is in the Keychain. Verify the read API returns
        // a non-empty string (the value is base64-encoded 32 random
        // bytes, so the length is at least 32 * 4 / 3 ~= 44 chars).
        let stored = TesseraKeychainVolume.storedVolumePassword()
        XCTAssertNotNil(stored, "volume password should be in the Keychain")
        XCTAssertGreaterThan(stored?.count ?? 0, 32)
    }

    func testKeychainDeleteMakesMountFail() async throws {
        let volume = makeVolume()
        try await volume.create()
        try await volume.unmount()

        // Destroy the Keychain entry. A subsequent mount must fail
        // with `keychainMissingPassword` (the actor's contract: it
        // tries to read the password, finds nothing, throws).
        XCTAssertTrue(TesseraKeychainVolume.deleteVolumePassword())
        XCTAssertNil(TesseraKeychainVolume.storedVolumePassword())

        do {
            try await volume.mount()
            XCTFail("expected mount to fail when the Keychain entry is gone")
        } catch let err as TesseraEncryptedVolumeError {
            XCTAssertEqual(err.kind, .keychainMissingPassword, "got \(err)")
        } catch {
            XCTFail("expected TesseraEncryptedVolumeError, got \(error)")
        }
    }

    // MARK: - Reset

    func testResetDestroysAndRecreates() async throws {
        let volume = makeVolume()
        try await volume.create()

        // Put a file inside the volume so we can verify it does NOT
        // survive the reset.
        let probe = mountPoint.appendingPathComponent("before-reset.txt")
        try "will be gone".write(to: probe, atomically: true, encoding: .utf8)

        // Capture the old password so we can prove the new password
        // is different.
        let oldPassword = TesseraKeychainVolume.storedVolumePassword()
        XCTAssertNotNil(oldPassword)

        try await volume.reset()

        // After reset, the volume is mounted and contains a fresh
        // bundle. The pre-reset file must be gone.
        let resetMounted = await volume.isMounted
        XCTAssertTrue(resetMounted)
        XCTAssertFalse(FileManager.default.fileExists(atPath: probe.path))

        // The new password must differ from the old one. With 32
        // random bytes the probability of collision is negligible;
        // a match here would mean reset() reused the old password,
        // which would be a critical bug.
        let newPassword = TesseraKeychainVolume.storedVolumePassword()
        XCTAssertNotNil(newPassword)
        XCTAssertNotEqual(newPassword, oldPassword, "reset must rotate the volume password")
    }

    // MARK: - Migration

    func testMigrationCopiesData() async throws {
        // Stage synthetic data in a tmp "sandbox" so we don't touch
        // the user's real Library.
        let sandboxRoot = tmpDir.appendingPathComponent("sandbox")
        let appSupport = sandboxRoot
            .appendingPathComponent("Library/Application Support/TesseraStudio")
        let caches = sandboxRoot.appendingPathComponent("Library/Caches/TesseraStudio")
        let preferences = sandboxRoot.appendingPathComponent("Library/Preferences")
        for dir in [appSupport, caches, preferences] {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        try "store-bytes".write(
            to: appSupport.appendingPathComponent("default.store"),
            atomically: true, encoding: .utf8
        )
        try "cache-bytes".write(
            to: caches.appendingPathComponent("cache.dat"),
            atomically: true, encoding: .utf8
        )
        try "<plist/>".write(
            to: preferences.appendingPathComponent("com.tessera.studio.mac.plist"),
            atomically: true, encoding: .utf8
        )

        // Point the redirector's sandbox paths at our tmp tree so
        // the migrator's `.live` source resolves to it.
        TesseraDataRoot.setSandboxRoot(appSupport, for: .appSupport)
        TesseraDataRoot.setSandboxRoot(caches, for: .caches)
        TesseraDataRoot.setSandboxRoot(preferences, for: .preferences)
        defer {
            TesseraDataRoot.setSandboxRoot(URL(fileURLWithPath: "/dev/null"), for: .appSupport)
            TesseraDataRoot.setSandboxRoot(URL(fileURLWithPath: "/dev/null"), for: .caches)
            TesseraDataRoot.setSandboxRoot(URL(fileURLWithPath: "/dev/null"), for: .preferences)
        }

        let volume = makeVolume()
        let migrator = TesseraVolumeMigrator()
        let report = try await migrator.migrate(into: volume)

        XCTAssertTrue(report.verified, "migration must verify the copy")
        XCTAssertGreaterThanOrEqual(report.copiedFiles, 3)
        XCTAssertGreaterThan(report.copiedBytes, 0)
        XCTAssertGreaterThan(report.originalBytesOverwritten, 0)

        // Data should be at the new location (inside the volume).
        let copiedStore = mountPoint
            .appendingPathComponent("Library/Application Support/TesseraStudio/default.store")
        XCTAssertTrue(FileManager.default.fileExists(atPath: copiedStore.path))
        let copiedContent = try String(contentsOf: copiedStore, encoding: .utf8)
        XCTAssertEqual(copiedContent, "store-bytes")

        // The redirector should now point at the volume (the migrator
        // calls markMountedRootAsActive on success).
        XCTAssertTrue(TesseraDataRoot.isUsingEncryptedVolume())

        // The original files in the sandbox are still on disk after
        // the overwrite (the wipe step writes random data into them;
        // it does not delete them in this build, leaving a follow-up
        // deletion step for the cleanup pass). The original CONTENT
        // must be gone: reading the file as Data must NOT return the
        // original "store-bytes" sequence.
        let originalStore = appSupport.appendingPathComponent("default.store")
        if FileManager.default.fileExists(atPath: originalStore.path) {
            let overwritten = (try? Data(contentsOf: originalStore)) ?? Data()
            let originalBytes = Data("store-bytes".utf8)
            XCTAssertNotEqual(overwritten, originalBytes, "originals must be overwritten")
        }
    }

    // MARK: - Data root

    func testDataRootRespectsVolumeMount() {
        let volumeRoot = URL(fileURLWithPath: "/tmp/tessera-mounted-\(UUID().uuidString)")
        defer { TesseraDataRoot.setMountedRoot(nil) }

        // Unmounted: the redirector falls back to the sandbox.
        let sandboxPath = TesseraDataRoot.appSupport()
        XCTAssertFalse(sandboxPath.path.contains(volumeRoot.path))

        // Mounted: the redirector returns a path inside the volume.
        TesseraDataRoot.setMountedRoot(volumeRoot)
        XCTAssertTrue(TesseraDataRoot.isUsingEncryptedVolume())
        let mountedPath = TesseraDataRoot.appSupport()
        XCTAssertTrue(mountedPath.path.hasPrefix(volumeRoot.path))

        // Unmount clears it.
        TesseraDataRoot.setMountedRoot(nil)
        XCTAssertFalse(TesseraDataRoot.isUsingEncryptedVolume())
        XCTAssertEqual(TesseraDataRoot.appSupport(), sandboxPath)
    }

    // MARK: - Volume wrong password

    func testMountWithWrongKeychainPasswordFails() async throws {
        // Create the volume so the bundle exists and the Keychain has
        // the correct password, then overwrite the Keychain with a
        // different password. The next mount() must fail because
        // hdiutil's stdin-pass auth rejects the wrong password.
        let volume = makeVolume()
        try await volume.create()
        try await volume.unmount()

        // Replace the Keychain entry with a different password.
        let old = TesseraKeychainVolume.storedVolumePassword()
        XCTAssertNotNil(old)
        var bytes = Data(count: 32)
        let result = bytes.withUnsafeMutableBytes { raw -> Int32 in
            guard let base = raw.baseAddress else { return errSecAllocate }
            return SecRandomCopyBytes(kSecRandomDefault, 32, base)
        }
        XCTAssertEqual(result, errSecSuccess)
        let wrong = bytes.base64EncodedString()
        XCTAssertNotEqual(wrong, old)
        XCTAssertTrue(TesseraKeychainVolume.storeVolumePassword(wrong))

        do {
            try await volume.mount()
            XCTFail("expected mount to fail with a wrong password")
        } catch let err as TesseraEncryptedVolumeError {
            XCTAssertEqual(err.kind, .hdiutilFailed, "got \(err)")
        } catch {
            XCTFail("expected TesseraEncryptedVolumeError, got \(error)")
        }
    }

    // MARK: - Timing

    /// Records a few mount/unmount timings so the implementation
    /// report can quote real numbers from this machine. Not a strict
    /// assertion; the bound is loose (mount can take ~1-2s on a cold
    /// cache, even on M-series).
    func testMountUnmountTiming() async throws {
        let volume = makeVolume()
        let createStart = Date()
        try await volume.create()
        let createDuration = Date().timeIntervalSince(createStart)

        let unmountStart = Date()
        try await volume.unmount()
        let unmountDuration = Date().timeIntervalSince(unmountStart)

        let remountStart = Date()
        try await volume.mount()
        let remountDuration = Date().timeIntervalSince(remountStart)

        // Loose bounds; both create and mount have to actually run
        // hdiutil + APFS, which is at minimum a few hundred ms.
        XCTAssertGreaterThan(createDuration, 0.1)
        XCTAssertGreaterThan(unmountDuration, 0.05)
        XCTAssertGreaterThan(remountDuration, 0.1)

        // The implementation report wants concrete numbers, so
        // surface them via `print` so the value is in the test log.
        let report = String(
            format: "timing on %@: create=%.3fs, unmount=%.3fs, remount=%.3fs",
            ProcessInfo.processInfo.machineHardwareName ?? "this Mac",
            createDuration, unmountDuration, remountDuration
        )
        print("[TesseraEncryptedVolumeTests] \(report)")
    }

    // MARK: - Helpers

    private func makeVolume() -> TesseraEncryptedVolume {
        let config = TesseraVolumeConfig(
            bundleURL: bundleURL,
            mountPoint: mountPoint,
            volumeName: "TesseraVaultTest-\(UUID().uuidString.prefix(6))"
        )
        return TesseraEncryptedVolume(
            config: config,
            keychainAccount: keychainAccount
        )
    }

    private func unmountQuietly() async {
        // Best-effort: hdiutil detach on the mount point. Used by
        // tearDown to make sure no orphan volume survives the test
        // even when the test body failed mid-mount.
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/hdiutil")
        process.arguments = ["detach", mountPoint.path, "-force"]
        process.standardOutput = Pipe()
        process.standardError = Pipe()
        try? process.run()
        process.waitUntilExit()
    }
}

private extension ProcessInfo {
    /// `sysctl hw.model` - returns the marketing model name (e.g.
    /// "Mac14,2"). Used by the timing test to label its report.
    var machineHardwareName: String? {
        var size = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        var buffer = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.model", &buffer, &size, nil, 0)
        return String(cString: buffer)
    }
}
