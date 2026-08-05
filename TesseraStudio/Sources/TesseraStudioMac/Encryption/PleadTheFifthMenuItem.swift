#if canImport(AppKit)
import AppKit
import SwiftUI
import TesseraCore

/// The macOS menu bar item for "Plea the Fifth".
///
/// Owns an `NSStatusItem` with a submenu:
/// - "Plead the Fifth..." opens a custom NSPanel confirmation dialog
///   that requires the typed phrase `destroy everything` (case-
///   insensitive, paste blocked).
/// - "Plead the Fifth (covert)" appears when a covert trigger phrase
///   is configured, and shows the current phrase in its submenu.
/// - "Last wipe report..." opens the last wipe report JSON.
///
/// In coercion mode (``PleadTheFifthSettings/coercionMode``) the
/// destructive menu items are hidden, but the status item icon stays
/// in the menu bar as a neutral lock - the user knows the mode is
/// active; the adversary does not (see design section 9.5).
@MainActor
public final class PleadTheFifthMenuItem: NSObject {

    private let statusItem: NSStatusItem
    private let volume: PleadTheFifthVolume
    private let executor: PleadTheFifthExecutor
    private let store: WipeReportStore

    /// Hook called after the report has been written so the host app
    /// can exit. Defaults to `exit(0)`. The hook is dispatched
    /// asynchronously on the main queue so the calling task can
    /// unwind cleanly.
    private let exitAfterWipe: @MainActor () -> Void

    /// Sub-views.
    private var confirmationPanel: ConfirmationPanel?
    private var reportWindow: ReportWindow?
    private var coercionObserver: NSObjectProtocol?

    public init(
        volume: PleadTheFifthVolume,
        sidecarController: SidecarController = NoOpSidecarController(),
        store: WipeReportStore = WipeReportStore(),
        exitAfterWipe: @escaping @MainActor () -> Void = { exit(0) }
    ) {
        self.volume = volume
        self.executor = PleadTheFifthExecutor(
            volume: volume,
            sidecarController: sidecarController
        )
        self.store = store
        self.exitAfterWipe = exitAfterWipe
        self.statusItem = NSStatusBar.system.statusItem(
            withLength: NSStatusItem.variableLength
        )
        super.init()
        applyIcon()
        rebuildMenu()
        observeCoercionMode()
    }

    deinit {
        if let observer = coercionObserver {
            NotificationCenter.default.removeObserver(observer)
        }
    }

    // MARK: - public

    /// Tear down. The status item is removed from the menu bar.
    public func uninstall() {
        if let observer = coercionObserver {
            NotificationCenter.default.removeObserver(observer)
        }
        coercionObserver = nil
        confirmationPanel?.close()
        reportWindow?.close()
        NSStatusBar.system.removeStatusItem(statusItem)
    }

    // MARK: - menu

    private func applyIcon() {
        // SF Symbols; "lock" is neutral and looks like a dozen other
        // macOS menu bar items. SF Symbol rendering on the menu bar
        // is template (monochrome) by default - the user gets the
        // standard dark/light treatment.
        if let button = statusItem.button {
            button.image = NSImage(systemSymbolName: "lock.fill",
                                   accessibilityDescription: "Tessera")
        }
    }

    private func rebuildMenu() {
        let menu = NSMenu()
        menu.autoenablesItems = false

        let coerced = PleadTheFifthSettings.coercionMode
        let hasCovert = PleadTheFifthSettings.covertTriggerConfigured

        if !coerced {
            // Primary action.
            let primary = NSMenuItem(
                title: "Plead the Fifth\u{2026}",
                action: #selector(openConfirmation),
                keyEquivalent: ""
            )
            primary.target = self
            primary.isEnabled = true
            menu.addItem(primary)

            // Covert trigger (visible only if configured).
            if hasCovert, let phrase = PleadTheFifthSettings.covertTriggerPhrase.nilIfEmpty {
                let covert = NSMenuItem(
                    title: "Covert trigger (advanced)",
                    action: nil,
                    keyEquivalent: ""
                )
                let sub = NSMenu()
                let info = NSMenuItem(
                    title: "Phrase: \(phrase)",
                    action: nil,
                    keyEquivalent: ""
                )
                info.isEnabled = false
                sub.addItem(info)
                let test = NSMenuItem(
                    title: "Test",
                    action: #selector(testCovertTrigger),
                    keyEquivalent: ""
                )
                test.target = self
                sub.addItem(test)
                covert.submenu = sub
                menu.addItem(covert)
            }

            menu.addItem(.separator())

            let report = NSMenuItem(
                title: "Last wipe report\u{2026}",
                action: #selector(openReport),
                keyEquivalent: ""
            )
            report.target = self
            menu.addItem(report)
        } else {
            // Coercion mode: only the lock icon stays visible, the
            // destructive controls are hidden. We do still expose a
            // tiny informational entry so the user (who knows the
            // mode is on) has a confirmation.
            let info = NSMenuItem(
                title: "Plea the Fifth: coercion mode",
                action: nil,
                keyEquivalent: ""
            )
            info.isEnabled = false
            menu.addItem(info)
        }

        statusItem.menu = menu
    }

    // MARK: - actions

    @objc private func openConfirmation() {
        if confirmationPanel == nil {
            confirmationPanel = ConfirmationPanel { [weak self] result in
                guard let self = self else { return }
                Task { @MainActor in
                    await self.handleConfirmation(result: result)
                }
            }
        }
        confirmationPanel?.present()
    }

    @objc private func openReport() {
        if reportWindow == nil {
            reportWindow = ReportWindow()
        }
        reportWindow?.present()
    }

    @objc private func testCovertTrigger() {
        // The "Test" button in the covert submenu: post a non-
        // destructive alert so the user knows the phrase is
        // configured. We intentionally do NOT fire the wipe.
        let alert = NSAlert()
        alert.messageText = "Covert trigger is armed"
        alert.informativeText = """
            The current phrase is set. Type it inside any Tessera text
            input (followed by at least 4 more characters) and the wipe
            will fire. The 5-second cooldown prevents double-fires.
            """
        alert.alertStyle = .informational
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }

    private func handleConfirmation(result: ConfirmationPanel.Result) async {
        switch result {
        case .cancelled, .rateLimited:
            return
        case .confirmed:
            await runWipe(trigger: .menu)
        }
    }

    // MARK: - wipe dispatch

    private func runWipe(trigger: PleadTheFifthExecutor.TriggerSource) async {
        do {
            let report = try await executor.destroyAll(trigger: trigger)
            // Save the report BEFORE we ask the host to exit, so the
            // audit trail is on disk no matter what happens after.
            try? store.save(report)
            // Defer the actual exit by 100ms so the save's I/O has
            // a chance to fully flush and any UI / log line has time
            // to render.
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [exitAfterWipe] in
                exitAfterWipe()
            }
        } catch {
            // The only throw from destroyAll is "already running".
            // The user can't trigger this from a UI element, so we
            // just log and continue.
            NSLog("Plea the Fifth: executor refused - %@", "\(error)")
        }
    }

    // MARK: - coercion mode

    private func observeCoercionMode() {
        // UserDefaults is KVO-observable for the standard suite but
        // there is no public `dataChanged` key. The reliable signal
        // is the in-process notification `UserDefaults.didChangeNotification`
        // (we don't need cross-process; the menu bar updates only
        // when Tessera writes the value itself).
        coercionObserver = NotificationCenter.default.addObserver(
            forName: UserDefaults.didChangeNotification,
            object: UserDefaults.standard,
            queue: .main
        ) { [weak self] _ in
            self?.rebuildMenu()
        }
        // Settings view posts this when the user clicks "View last
        // wipe report". The menu item owns the window, so it listens
        // and presents.
        NotificationCenter.default.addObserver(
            forName: .openLastWipeReport,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.openReport()
        }
    }
}

private extension String {
    var nilIfEmpty: String? { isEmpty ? nil : self }
}
#endif
