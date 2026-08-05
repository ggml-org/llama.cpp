import SwiftUI

// MARK: - AnimationPrimitives

/// The seven animation primitives the editor uses (per
/// `docs/tessera-productivity-design.md` §8). Each primitive
/// is a SwiftUI `Animation` value or a small `ViewModifier`
/// that wraps one; the editor's view layer composes them
/// into the block-level transitions, the text-appear cadence,
/// the agent-cursor blink, the thinking-pulse animation,
/// and the "Hold your horses" banner slide-in.
///
/// **Reduce Motion.** Every primitive respects the system
/// `accessibilityReduceMotion` setting. The fallback is
/// spelled out per the spec:
///
/// | Primitive | Fallback |
/// |---|---|
/// | Block slide-in | Crossfade only (no slide) |
/// | Block replace | Crossfade only |
/// | Block delete collapse | Instant removal (no animation) |
/// | Text appear | Whole text appears at once (no per-char cadence) |
/// | Cursor blink | Static caret (no blink) |
/// | Thinking pulse | Static dot (no animation) |
/// | Agent paused banner | Instant appearance (no slide) |
///
/// **Interruptibility.** All animations are SwiftUI
/// `withAnimation` / `Animation`-based, so a new
/// `withAnimation` call automatically interrupts the
/// previous one. The agent can cancel a slide-in by
/// triggering a replacement mid-animation.
///
/// **Cadence.** The text-appear cadence is a 60ms-per-char
/// default (range 30-100ms, user setting). It's driven by
/// an `AsyncStream` so the agent's mutation stream flows
/// through the editor's view layer at the chosen rate.
public enum AnimationPrimitives {

    // MARK: - Durations (constants from spec §8)

    public static let blockSlideInDuration: TimeInterval = 0.25
    public static let blockReplaceDuration: TimeInterval = 0.30
    public static let blockDeleteDuration: TimeInterval = 0.20
    public static let textAppearPerChar: TimeInterval = 0.06
    public static let textAppearPerCharMin: TimeInterval = 0.03
    public static let textAppearPerCharMax: TimeInterval = 0.10
    public static let cursorBlinkDuration: TimeInterval = 0.53
    public static let thinkingPulseDurationSeconds: TimeInterval = 1.00
    public static let agentPausedBannerDuration: TimeInterval = 0.20

    // MARK: - Block slide-in

    /// Block slide-in (spec row 1). 250ms, .easeOut, falls
    /// back to a crossfade when Reduce Motion is on.
    public static func blockSlideIn(
        reduceMotion: Bool = Self.isReduceMotion
    ) -> Animation {
        if reduceMotion { return .easeInOut(duration: 0.15) }
        return .easeOut(duration: blockSlideInDuration)
    }

    // MARK: - Block replace

    /// Block replace (spec row 2). 300ms, .easeInOut,
    /// crossfade fallback.
    public static func blockReplace(
        reduceMotion: Bool = Self.isReduceMotion
    ) -> Animation {
        if reduceMotion { return .easeInOut(duration: 0.15) }
        return .easeInOut(duration: blockReplaceDuration)
    }

    // MARK: - Block delete collapse

    /// Block delete collapse (spec row 3). 200ms, .easeIn,
    /// instant removal fallback.
    public static func blockDelete(
        reduceMotion: Bool = Self.isReduceMotion
    ) -> Animation? {
        if reduceMotion { return nil }
        return .easeIn(duration: blockDeleteDuration)
    }

    // MARK: - Text appear (per-char cadence)

    /// The per-character delay for the text-appear cadence
    /// (spec row 4). Defaults to 60ms; the user can configure
    /// 30-100ms. Returns `nil` (whole text at once) when
    /// Reduce Motion is on.
    public static func textAppearDelay(
        reduceMotion: Bool = Self.isReduceMotion,
        perChar: TimeInterval = textAppearPerChar
    ) -> TimeInterval? {
        if reduceMotion { return nil }
        return perChar.clamped(to: textAppearPerCharMin...textAppearPerCharMax)
    }

    // MARK: - Cursor blink

    /// Cursor blink cycle (spec row 5). 530ms cycle, 50/50
    /// on/off. Falls back to a static caret (no animation)
    /// when Reduce Motion is on.
    public static func cursorBlink(
        reduceMotion: Bool = Self.isReduceMotion
    ) -> TimeInterval? {
        if reduceMotion { return nil }
        return cursorBlinkDuration
    }

    // MARK: - Thinking pulse

    /// Thinking pulse cycle (spec row 6). 1000ms cycle,
    /// spring(response: 0.5, dampingFraction: 0.7). Falls
    /// back to a static dot when Reduce Motion is on.
    public static func thinkingPulseAnimation(
        reduceMotion: Bool = Self.isReduceMotion
    ) -> Animation? {
        if reduceMotion { return nil }
        return Animation.spring(response: 0.5, dampingFraction: 0.7)
    }

    public static let thinkingPulseDuration: TimeInterval = thinkingPulseDurationSeconds

    // MARK: - Agent paused banner

    /// Agent paused banner slide-in (spec row 7). 200ms,
    /// .easeOut. Falls back to an instant appearance when
    /// Reduce Motion is on.
    public static func agentPausedBanner(
        reduceMotion: Bool = Self.isReduceMotion
    ) -> Animation {
        if reduceMotion { return .linear(duration: 0.001) }
        return .easeOut(duration: agentPausedBannerDuration)
    }

    // MARK: - Reduce Motion detection

    /// True iff the system Reduce Motion accessibility
    /// setting is on. The static accessor is fine for the
    /// editor's view layer; production can inject a
    /// different value (e.g., for unit tests of the
    /// fallback paths).
    public static var isReduceMotion: Bool {
        #if canImport(UIKit)
        return UIAccessibility.isReduceMotionEnabled
        #else
        return NSWorkspace.shared.accessibilityDisplayShouldReduceMotion
        #endif
    }
}

// MARK: - View modifiers

/// A view modifier that applies the block slide-in animation
/// when the block first appears. The slide-in is a combined
/// opacity + slight Y-offset transition (the spec calls for
/// "slide" but the exact transform is the spec's call; this
/// is the standard iOS/macOS block-appear pattern).
public struct BlockSlideInModifier: ViewModifier {
    public let isActive: Bool
    public init(isActive: Bool = true) { self.isActive = isActive }
    public func body(content: Content) -> some View {
        content
            .opacity(isActive ? 1 : 0)
            .offset(y: isActive ? 0 : 8)
    }
}

public extension View {
    /// Apply the block slide-in animation. Call inside a
    /// `withAnimation { isActive = true }` block; the
    /// animation type is `AnimationPrimitives.blockSlideIn`.
    func blockSlideIn(isActive: Bool) -> some View {
        modifier(BlockSlideInModifier(isActive: isActive))
    }
}

/// A view modifier that runs the thinking pulse (oscillating
/// scale + opacity). Used by the chat panel's "agent is
/// working" status dot. The modifier is a no-op (renders the
/// static dot) when Reduce Motion is on.
public struct ThinkingPulseModifier: ViewModifier {
    @State private var phase: Double = 0
    public let isActive: Bool
    public init(isActive: Bool) { self.isActive = isActive }
    public func body(content: Content) -> some View {
        Group {
            if isActive {
                content
                    .scaleEffect(0.8 + 0.2 * phase)
                    .opacity(0.4 + 0.6 * phase)
                    .onAppear {
                        guard let animation = AnimationPrimitives.thinkingPulseAnimation() else { return }
                        withAnimation(animation.repeatForever(autoreverses: true)) {
                            phase = 1
                        }
                    }
            } else {
                content
            }
        }
    }
}

public extension View {
    /// Apply the thinking-pulse animation. No-op under Reduce
    /// Motion.
    func thinkingPulse(isActive: Bool) -> some View {
        modifier(ThinkingPulseModifier(isActive: isActive))
    }
}

/// A view modifier for the agent-cursor blink. The cursor's
/// "active" state drives a 50/50 on/off cycle at 530ms. The
/// modifier is a no-op (static cursor) under Reduce Motion.
public struct CursorBlinkModifier: ViewModifier {
    @State private var on: Bool = true
    public let isActive: Bool
    public init(isActive: Bool) { self.isActive = isActive }
    public func body(content: Content) -> some View {
        Group {
            if isActive, let cycle = AnimationPrimitives.cursorBlink() {
                content
                    .opacity(on ? 1 : 0)
                    .onAppear {
                        withAnimation(.easeInOut(duration: cycle / 2).repeatForever(autoreverses: true)) {
                            on.toggle()
                        }
                    }
            } else {
                content.opacity(1)
            }
        }
    }
}

public extension View {
    /// Apply the cursor-blink animation. No-op under Reduce Motion.
    func cursorBlink(isActive: Bool) -> some View {
        modifier(CursorBlinkModifier(isActive: isActive))
    }
}

// MARK: - TextAppearCadence

/// A small helper that streams a string one character at a
/// time, at the configured per-character delay. The view
/// layer consumes the stream and animates the text view's
/// contents accordingly.
///
/// The cadence is interruptible: stopping the `Task` (via
/// task cancellation) halts the stream immediately. The
/// editor uses this to cancel a text-appear when the user
/// starts editing the block.
public struct TextAppearCadence: Sendable {
    public let perChar: TimeInterval
    public init(perChar: TimeInterval = AnimationPrimitives.textAppearPerChar) {
        self.perChar = perChar
    }

    /// Stream `text` character-by-character. Yields each
    /// character as a `String` of length 1, plus the index
    /// of the character in the source. The consumer can
    /// build a `String` from the prefix of length
    /// `index + 1` and feed it to the text view on each
    /// iteration.
    public func stream(_ text: String) -> AsyncStream<(index: Int, character: Character)> {
        let delay = perChar
        return AsyncStream { continuation in
            let task = Task {
                for (idx, ch) in text.enumerated() {
                    if Task.isCancelled { break }
                    continuation.yield((index: idx, character: ch))
                    if delay > 0 {
                        try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                    }
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
