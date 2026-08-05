import SwiftUI
#if canImport(AppKit)
import AppKit
#endif
import TesseraCore

/// Settings UI for the encrypted-volume foundation of "Plea the Fifth".
///
/// Minimal first cut: surfaces the volume state, lets the user
/// trigger the migration flow, and exposes the "Reset Tessera"
/// recovery path. The full settings surface (covert trigger, hot-key
/// remap, coercion mode toggle) is Phase 2's work; this view is just
/// enough that the encrypted-volume feature is reachable from
/// Settings without writing the whole UX in one pass.
///
/// Layout follows the macOS HIG: a Form with grouped sections, an
/// action button for the irreversible operations gated behind a
/// confirmation step, and a status row at the top.
@available(macOS 14, *)
public struct PleadTheFifthSettingsView: View {
    @State private var state: VolumeState = .unknown
    @State private var status: String = ""
    @State private var isWorking = false
    @State private var showResetConfirm = false
    @State private var resetPhrase = ""

    public init() {}

    public var body: some View {
        Form {
            Section("Encrypted volume") {
                LabeledContent("Status") {
                    Text(stateLabel).foregroundStyle(stateColor)
                }
                LabeledContent("Mount point") {
                    Text(TesseraDataRoot.isUsingEncryptedVolume()
                         ? TesseraDataRoot.mountedRoot()?.path ?? "—"
                         : "—")
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                if !status.isEmpty {
                    Text(status).font(.caption).foregroundStyle(.secondary)
                }
            }
            Section {
                Button {
                    Task { await runSetup() }
                } label: {
                    Text(state == .unmounted ? "Set up encrypted volume" : "Re-mount volume")
                }
                .disabled(isWorking)

                if state == .mounted {
                    Button("Unmount") {
                        Task { await runUnmount() }
                    }
                    .disabled(isWorking)
                }
            } footer: {
                Text("Tessera stores all of its data in an encrypted APFS volume. The volume password is held in your Mac's Keychain. We recommend you also save it in 1Password / Bitwarden / Apple Passwords.")
                    .font(.caption)
            }
            Section {
                Button("Reset Tessera…", role: .destructive) {
                    showResetConfirm = true
                }
                .disabled(isWorking || state == .unknown)
            } header: {
                Text("Recovery")
            } footer: {
                Text("Reset Tessera destroys the encrypted volume and starts fresh. Existing data is unrecoverable. Use this when the volume is corrupted or the password is lost.")
                    .font(.caption)
            }
        }
        .formStyle(.grouped)
        .task { await refresh() }
        .alert("Reset Tessera", isPresented: $showResetConfirm) {
            TextField("Type \"reset everything\" to confirm", text: $resetPhrase)
            Button("Reset", role: .destructive) {
                Task { await runReset() }
            }
            Button("Cancel", role: .cancel) { resetPhrase = "" }
        } message: {
            Text("This will destroy the encrypted volume and create a new one. The existing data is unrecoverable.")
        }
    }

    private var stateLabel: String {
        switch state {
        case .unknown: return "Checking…"
        case .unmounted: return "Not set up"
        case .mounted: return "Mounted"
        case .error(let msg): return "Error: \(msg)"
        }
    }

    private var stateColor: Color {
        switch state {
        case .mounted: return .green
        case .unmounted: return .secondary
        case .error: return .red
        case .unknown: return .secondary
        }
    }

    @MainActor
    private func refresh() async {
        let hasPassword = TesseraKeychainVolume.hasVolumePassword()
        state = hasPassword ? .mounted : .unmounted
    }

    @MainActor
    private func runSetup() async {
        isWorking = true
        defer { isWorking = false }
        do {
            let cfg = defaultConfig()
            let volume = TesseraEncryptedVolume(config: cfg)
            if state == .unmounted {
                try await volume.create()
            } else {
                try await volume.mount()
            }
            TesseraDataRoot.setMountedRoot(cfg.mountPoint)
            status = "Volume ready."
            state = .mounted
        } catch {
            status = error.localizedDescription
            state = .error(error.localizedDescription)
        }
    }

    @MainActor
    private func runUnmount() async {
        isWorking = true
        defer { isWorking = false }
        let cfg = defaultConfig()
        let volume = TesseraEncryptedVolume(config: cfg)
        do {
            try await volume.unmount()
            TesseraDataRoot.setMountedRoot(nil)
            status = "Volume unmounted."
        } catch {
            status = error.localizedDescription
        }
    }

    @MainActor
    private func runReset() async {
        defer { resetPhrase = "" }
        guard resetPhrase.trimmingCharacters(in: .whitespaces).lowercased() == "reset everything" else {
            status = "Reset cancelled: phrase did not match."
            return
        }
        isWorking = true
        defer { isWorking = false }
        let cfg = defaultConfig()
        let volume = TesseraEncryptedVolume(config: cfg)
        do {
            try await volume.reset()
            TesseraDataRoot.setMountedRoot(cfg.mountPoint)
            status = "Volume reset."
            state = .mounted
        } catch {
            status = error.localizedDescription
            state = .error(error.localizedDescription)
        }
    }

    private func defaultConfig() -> TesseraVolumeConfig {
        let appSupport = (FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first ?? URL(fileURLWithPath: NSTemporaryDirectory()))
            .appendingPathComponent("TesseraStudio", isDirectory: true)
        let bundle = appSupport.appendingPathComponent("vault.sparsebundle")
        return TesseraVolumeConfig(bundleURL: bundle)
    }

    private enum VolumeState: Equatable {
        case unknown, unmounted, mounted
        case error(String)
    }
}
