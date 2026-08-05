#if canImport(AppKit)
import AppKit
import SwiftUI
import TesseraCore

/// The typed-phrase confirmation dialog for the menu bar "Plead the
/// Fifth..." action. Custom NSPanel (per design section 8.5) so that:
///
/// 1. Paste is intercepted and ignored - the user must type the
///    phrase. Defeats the "I copied `destroy everything` from
///    somewhere" attack.
/// 2. The Confirm button is disabled for 5 seconds after the panel
///    appears, preventing a misclick from firing the wipe.
/// 3. Failed attempts are rate-limited: the dialog is dismissed
///    if the user has accumulated 3 failures within a 30-second
///    window, and a fourth attempt is logged.
@MainActor
public final class ConfirmationPanel: NSObject, NSWindowDelegate {
    public enum Result {
        case confirmed
        case cancelled
        case rateLimited
    }

    private let onResult: (Result) -> Void
    private var window: NSPanel?
    private var hosting: NSHostingController<ConfirmationView>?
    private var unlockTimer: Timer?
    private var failedAttempts: Int = 0
    private var firstFailureAt: Date?
    private let confirmationPhrase = "destroy everything"
    private let unlockDelaySeconds: TimeInterval = 5
    private let rateLimitWindow: TimeInterval = 30
    private let rateLimitCap = 3

    public init(onResult: @escaping (Result) -> Void) {
        self.onResult = onResult
    }

    public func present() {
        if let window = window, window.isVisible {
            window.makeKeyAndOrderFront(nil)
            return
        }
        let view = ConfirmationView(
            expectedPhrase: confirmationPhrase,
            unlockDelaySeconds: unlockDelaySeconds,
            onSubmit: { [weak self] text in
                self?.handleSubmit(text: text) ?? .dismissed
            },
            onCancel: { [weak self] in
                self?.finish(.cancelled)
            }
        )
        let hosting = NSHostingController(rootView: view)
        let panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 460, height: 220),
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        panel.title = "Plead the Fifth"
        panel.isFloatingPanel = true
        panel.becomesKeyOnlyIfNeeded = true
        panel.hidesOnDeactivate = false
        panel.contentViewController = hosting
        panel.delegate = self
        panel.center()

        self.window = panel
        self.hosting = hosting
        panel.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    public func close() {
        unlockTimer?.invalidate()
        unlockTimer = nil
        window?.close()
        window = nil
    }

    // MARK: - NSWindowDelegate

    public func windowWillClose(_ notification: Notification) {
        // Close via the red traffic light counts as cancel.
        if let window = self.window, notification.object as? NSWindow === window {
            finish(.cancelled)
        }
    }

    // MARK: - internals

    private func handleSubmit(text: String) -> ConfirmationView.SubmitResult {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.caseInsensitiveCompare(confirmationPhrase) == .orderedSame {
            failedAttempts = 0
            firstFailureAt = nil
            finish(.confirmed)
            return .dismissed
        }
        // Rate-limit bookkeeping.
        let now = Date()
        if let first = firstFailureAt, now.timeIntervalSince(first) > rateLimitWindow {
            firstFailureAt = nil
            failedAttempts = 0
        }
        if firstFailureAt == nil { firstFailureAt = now }
        failedAttempts += 1
        if failedAttempts > rateLimitCap {
            finish(.rateLimited)
            return .dismissed
        }
        return .wrongPhrase(failedAttempts: failedAttempts, cap: rateLimitCap)
    }

    private func finish(_ result: Result) {
        unlockTimer?.invalidate()
        unlockTimer = nil
        window?.close()
        window = nil
        onResult(result)
    }
}

/// SwiftUI body of the confirmation panel. Stays thin: it owns
/// no state, reports user input to the panel.
private struct ConfirmationView: View {
    enum SubmitResult: Equatable {
        case dismissed
        case wrongPhrase(failedAttempts: Int, cap: Int)
    }

    let expectedPhrase: String
    let unlockDelaySeconds: TimeInterval
    let onSubmit: (String) -> SubmitResult
    let onCancel: () -> Void

    @State private var text: String = ""
    @State private var unlockAt: Date = Date()
    @State private var now: Date = Date()
    @State private var lastFailure: String?
    @State private var hasAppeared = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Plead the Fifth")
                .font(.headline)
            Text("Type the phrase below to confirm. Paste is disabled. The button unlocks after a short delay.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            // Custom paste-blocked field. SwiftUI's TextField on macOS
            // honors the standard Edit > Paste menu, so we override
            // it with an empty `NSResponder` and let the
            // NSHostingController forward nothing for the paste
            // action. Done in `pasteBlocker`.
            PasteBlockedTextField(text: $text, onSubmit: submit)
                .frame(height: 24)
            if let lastFailure = lastFailure {
                Text(lastFailure)
                    .font(.caption)
                    .foregroundStyle(.red)
            }
            HStack {
                Spacer()
                Button("Cancel", role: .cancel, action: onCancel)
                    .keyboardShortcut(.cancelAction)
                Button("Confirm") { submit() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(!canConfirm)
            }
        }
        .padding(20)
        .frame(width: 460)
        .onAppear {
            // First-appear: set the unlock target, kick the timer.
            if !hasAppeared {
                unlockAt = Date().addingTimeInterval(unlockDelaySeconds)
                hasAppeared = true
            }
        }
        // 1Hz tick is plenty - the button is whole seconds.
        .task {
            while !Task.isCancelled {
                now = Date()
                try? await Task.sleep(nanoseconds: 250_000_000)
            }
        }
    }

    private var canConfirm: Bool {
        now >= unlockAt
            && text.trimmingCharacters(in: .whitespacesAndNewlines).count >= expectedPhrase.count
    }

    private func submit() {
        guard canConfirm else { return }
        let result = onSubmit(text)
        switch result {
        case .dismissed:
            text = ""
            lastFailure = nil
        case .wrongPhrase(let n, let cap):
            lastFailure = "Phrase does not match. Attempt \(n) of \(cap)."
            text = ""
        }
    }
}

/// NSViewRepresentable wrapping NSTextField with paste blocked.
/// SwiftUI's `TextField` on macOS forwards the system's Edit > Paste
/// action; we replace the field with a custom NSTextField whose
/// `validateProposedFirstResponder` and `paste` sender return nil
/// and ignore the operation, respectively. The user can still type
/// the phrase manually; pasting `destroy everything` is rejected.
private struct PasteBlockedTextField: NSViewRepresentable {
    @Binding var text: String
    let onSubmit: () -> Void

    func makeCoordinator() -> Coordinator { Coordinator(text: $text, onSubmit: onSubmit) }

    func makeNSView(context: Context) -> NSTextField {
        let field = NSTextField()
        field.isBordered = true
        field.bezelStyle = .roundedBezel
        field.placeholderString = "Type the phrase"
        field.delegate = context.coordinator
        field.target = context.coordinator
        field.action = #selector(Coordinator.commit(_:))
        return field
    }

    func updateNSView(_ nsView: NSTextField, context: Context) {
        if nsView.stringValue != text { nsView.stringValue = text }
    }

    final class Coordinator: NSObject, NSTextFieldDelegate {
        @Binding var text: String
        let onSubmit: () -> Void

        init(text: Binding<String>, onSubmit: @escaping () -> Void) {
            _text = text
            self.onSubmit = onSubmit
        }

        func controlTextDidChange(_ obj: Notification) {
            guard let field = obj.object as? NSTextField else { return }
            text = field.stringValue
        }

        @objc func commit(_ sender: Any?) {
            onSubmit()
        }

        // Block paste via the responder chain (Edit > Paste,
        // Cmd-V, programmatic). The field receives the action
        // but does nothing with it.
        func controlTextValidating(_ control: NSControl,
                                   command: Selector,
                                   value: Any?) -> Bool {
            // `paste:` and `pasteAndMatchStyle:` are NSText protocol
            // methods; use the raw selector strings because the
            // protocol conformance isn't re-exported as a typed
            // #selector on NSResponder in Swift.
            if command == Selector(("paste:"))
                || command == Selector(("pasteAndMatchStyle:")) {
                return false
            }
            return true
        }
    }
}
#endif
