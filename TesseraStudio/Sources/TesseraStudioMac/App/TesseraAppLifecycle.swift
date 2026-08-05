import Foundation
import TesseraCore

/// App-launch helpers for the encrypted-volume foundation. Keeps the
/// `TesseraStudioMacApp.init` body small and gives the SwiftUI
/// settings view a single bootstrap entry point.
@MainActor
enum TesseraAppLifecycle {

    /// Mount the encrypted volume on launch if one is configured
    /// (Keychain has a password + the bundle exists on disk).
    ///
    /// The "subsequent launch" path from the design spec (Section
    /// 6.5): read the password from the Keychain, run hdiutil
    /// attach, and repoint the data root redirector. If anything
    /// fails (key missing, bundle missing, wrong password), the
    /// app falls through and uses the sandbox location - the user
    /// is then shown the Settings view's "Set up" prompt or a
    /// hard error depending on the failure mode.
    ///
    /// This function is intentionally silent on failure. The
    /// follow-up wiring (Phase 2's wipe-executor worker) is
    /// responsible for showing the user-visible error UX; for v1
    /// the failure is logged and the app continues with the
    /// sandbox layout, which is the "fail-open" interim behavior
    /// for the dev preview. Hard-fail-closed is gated on the
    /// "Plea the Fifth" feature flag being enabled.
    static func bootstrapEncryptedVolume() {
        guard TesseraKeychainVolume.hasVolumePassword() else {
            // No password -> no volume. App starts with sandbox data
            // paths; the user will be prompted to set up the volume
            // on the first launch with the feature enabled.
            return
        }
        let config = defaultVolumeConfig()
        guard FileManager.default.fileExists(atPath: config.bundleURL.path) else {
            // Keychain has a password but the bundle is missing on
            // disk. The user is in the "password only" state (e.g.
            // they restored from a Time Machine backup that included
            // the Keychain but not the bundle). Reset the Keychain
            // and let the Settings view prompt for a fresh setup.
            _ = TesseraKeychainVolume.deleteVolumePassword()
            return
        }
        let volume = TesseraEncryptedVolume(config: config)
        Task {
            do {
                try await volume.mount()
                TesseraDataRoot.setMountedRoot(config.mountPoint)
            } catch {
                // Log and continue. The Settings view's "Set up"
                // flow handles the user-visible recovery.
                print("[TesseraAppLifecycle] mount failed: \(error.localizedDescription)")
            }
        }
    }

    /// The production volume config. Lives in
    /// `~/Library/Application Support/TesseraStudio/vault.sparsebundle`
    /// so the bundle sits next to the SwiftData store it protects.
    static func defaultVolumeConfig() -> TesseraVolumeConfig {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first ?? URL(fileURLWithPath: NSTemporaryDirectory())
        let bundle = appSupport
            .appendingPathComponent("TesseraStudio", isDirectory: true)
            .appendingPathComponent("vault.sparsebundle")
        return TesseraVolumeConfig(
            bundleURL: bundle,
            mountPoint: URL(fileURLWithPath: "/Volumes/TesseraVault"),
            volumeName: "TesseraVault"
        )
    }
}
