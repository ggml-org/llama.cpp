import SwiftUI
import AppKit
import TesseraCore

// MARK: - AgentCursorOverlay

/// A SwiftUI overlay that visualizes the agent cursor
/// (per spec §6.5). The agent cursor is a small robot icon
/// at the agent's edit location with a subtle blue
/// background; the cursor blinks at the standard 530ms
/// rate when the agent is active. The user cursor is the
/// standard system text caret (no special treatment).
///
/// **Two cursors, no contention.** The user and the
/// agent have separate cursors in the same document.
/// Both can be active at the same time; the user can
/// click anywhere without affecting the agent's cursor.
/// The overlay reads its position from
/// `EditorCursorState.agentCursor` (the `TextCursor`
/// data model) and converts the offset to a screen
/// position via the text view's `layoutManager`.
///
/// **State.** The overlay is a passive view: it reads
/// `EditorCursorState` from the environment (provided
/// by the host window) and renders the agent cursor at
/// the position the host computes. The host is
/// responsible for mapping the AST offset to a screen
/// position; the overlay only knows how to draw.
public struct AgentCursorOverlay: View {
    public let state: EditorCursorState
    public let theme: EditorTheme
    public let screenPositionProvider: (TextCursor) -> CGPoint?

    public init(
        state: EditorCursorState,
        theme: EditorTheme = .light,
        screenPositionProvider: @escaping (TextCursor) -> CGPoint?
    ) {
        self.state = state
        self.theme = theme
        self.screenPositionProvider = screenPositionProvider
    }

    public var body: some View {
        ZStack(alignment: .topLeading) {
            if let agentCursor = state.agentCursor,
               let position = screenPositionProvider(agentCursor) {
                AgentCursorGlyph(
                    isActive: state.agentCursorActive,
                    colorHex: theme.agentCursorColorHex
                )
                .position(x: position.x, y: position.y)
                .allowsHitTesting(false)
            }
        }
    }
}

// MARK: - AgentCursorGlyph

/// The visual representation of the agent cursor. A
/// small robot icon with a subtle blue background that
/// blinks when the agent is active. The blink uses the
/// `cursorBlink` animation primitive (530ms cycle, 50/50
/// on/off; static under Reduce Motion).
private struct AgentCursorGlyph: View {
    let isActive: Bool
    let colorHex: String

    var body: some View {
        ZStack {
            // Background bar (subtle blue)
            RoundedRectangle(cornerRadius: 2)
                .fill(Color(hex: colorHex)?.opacity(0.15) ?? Color.blue.opacity(0.15))
                .frame(width: 14, height: 18)
            // Robot icon
            Image(systemName: "cpu")
                .font(.system(size: 9, weight: .semibold))
                .foregroundStyle(Color(hex: colorHex) ?? .blue)
        }
        .cursorBlink(isActive: isActive)
    }
}

// MARK: - Color(hex:) helper (SwiftUI Color)

private extension Color {
    init?(hex: String) {
        var s = hex
        if s.hasPrefix("#") { s.removeFirst() }
        guard s.count == 6 || s.count == 8 else { return nil }
        var value: UInt64 = 0
        guard Scanner(string: s).scanHexInt64(&value) else { return nil }
        let r, g, b, a: Double
        if s.count == 6 {
            r = Double((value >> 16) & 0xFF) / 255
            g = Double((value >> 8) & 0xFF) / 255
            b = Double(value & 0xFF) / 255
            a = 1
        } else {
            r = Double((value >> 24) & 0xFF) / 255
            g = Double((value >> 16) & 0xFF) / 255
            b = Double((value >> 8) & 0xFF) / 255
            a = Double(value & 0xFF) / 255
        }
        self = Color(red: r, green: g, blue: b, opacity: a)
    }
}
