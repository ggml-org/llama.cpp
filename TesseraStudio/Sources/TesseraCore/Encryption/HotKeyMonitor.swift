#if canImport(AppKit)
import AppKit
import Foundation

/// macOS-only global hot-key for the "Plea the Fifth" command.
///
/// Uses `NSEvent.addGlobalMonitorForEvents(matching:handler:)` which
/// requires the user to grant Tessera Input Monitoring permission in
/// System Settings -> Privacy & Security -> Input Monitoring. The
/// monitor only fires while Tessera is running; this is the
/// documented limitation of `addGlobalMonitorForEvents` and we accept
/// it for v1.
///
/// The default chord is `Cmd+Shift+Backspace` (keyCode 51). The
/// choice is the design's: the chord is unclaimed, unambiguous, and
/// can be pressed in one motion under duress.
public final class HotKeyMonitor: @unchecked Sendable {
    public struct Chord: Equatable, Sendable {
        public let keyCode: UInt16
        public let command: Bool
        public let shift: Bool
        public init(keyCode: UInt16, command: Bool, shift: Bool) {
            self.keyCode = keyCode
            self.command = command
            self.shift = shift
        }

        public static let defaultChord = Chord(
            keyCode: 51,        // backspace / delete
            command: true,
            shift: true
        )

        public var displayString: String {
            // macOS convention: ⌘⇧⌫
            var parts: [String] = []
            if command { parts.append("⌘") }
            if shift { parts.append("⇧") }
            parts.append("⌫")
            return parts.joined()
        }
    }

    public typealias Handler = @MainActor @Sendable () -> Void

    private var monitor: Any?
    private let chord: Chord
    private let handler: Handler

    public init(chord: Chord = .defaultChord, handler: @escaping Handler) {
        self.chord = chord
        self.handler = handler
    }

    /// Register the global monitor. Safe to call once per instance.
    /// The returned handle is opaque; callers normally let the
    /// monitor live for the lifetime of the app.
    @discardableResult
    public func start() -> Bool {
        guard monitor == nil else { return true }
        let mask: NSEvent.EventTypeMask = .keyDown
        let chord = self.chord
        let handler = self.handler
        let m = NSEvent.addGlobalMonitorForEvents(matching: mask) { event in
            // The monitor callback receives a non-optional NSEvent.
            let mods = event.modifierFlags
            // Reject any extra modifiers the user did not configure,
            // so the chord stays specific and we do not steal
            // Cmd+Shift+Backspace+Option or similar.
            if mods.contains(.option) || mods.contains(.control) { return }
            let cmd = mods.contains(.command)
            let sh  = mods.contains(.shift)
            guard event.keyCode == chord.keyCode,
                  cmd == chord.command,
                  sh == chord.shift else { return }
            DispatchQueue.main.async { handler() }
        }
        if m == nil {
            // The most common reason is "user has not granted Input
            // Monitoring permission yet." We return false so the
            // caller (the menu bar wiring) can surface a one-time
            // hint to the user.
            return false
        }
        self.monitor = m
        return true
    }

    public func stop() {
        if let m = monitor {
            NSEvent.removeMonitor(m)
            monitor = nil
        }
    }

    deinit {
        if let m = monitor { NSEvent.removeMonitor(m) }
    }

    /// Pure helper exposed for tests. Returns true if `event` matches
    /// `chord` under the same rules the live monitor uses.
    public static func matches(event: NSEvent, chord: Chord) -> Bool {
        let mods = event.modifierFlags
        if mods.contains(.option) || mods.contains(.control) { return false }
        return event.keyCode == chord.keyCode
            && mods.contains(.command) == chord.command
            && mods.contains(.shift) == chord.shift
    }
}
#endif
