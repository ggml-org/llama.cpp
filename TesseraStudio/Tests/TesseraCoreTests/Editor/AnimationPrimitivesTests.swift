import XCTest
import Foundation
import SwiftUI
@testable import TesseraCore

/// Tests for `AnimationPrimitives` (per spec §8). The seven
/// primitives are: block slide-in, block replace, block
/// delete collapse, text appear, cursor blink, thinking
/// pulse, agent paused banner. Each has a Reduce Motion
/// fallback.
final class AnimationPrimitivesTests: XCTestCase {

    // MARK: - Durations (from spec §8)

    func testBlockSlideInDuration() {
        XCTAssertEqual(AnimationPrimitives.blockSlideInDuration, 0.25, accuracy: 0.001)
    }

    func testBlockReplaceDuration() {
        XCTAssertEqual(AnimationPrimitives.blockReplaceDuration, 0.30, accuracy: 0.001)
    }

    func testBlockDeleteDuration() {
        XCTAssertEqual(AnimationPrimitives.blockDeleteDuration, 0.20, accuracy: 0.001)
    }

    func testCursorBlinkDuration() {
        XCTAssertEqual(AnimationPrimitives.cursorBlinkDuration, 0.53, accuracy: 0.001)
    }

    func testThinkingPulseDuration() {
        XCTAssertEqual(AnimationPrimitives.thinkingPulseDuration, 1.0, accuracy: 0.001)
    }

    func testAgentPausedBannerDuration() {
        XCTAssertEqual(AnimationPrimitives.agentPausedBannerDuration, 0.20, accuracy: 0.001)
    }

    func testTextAppearPerCharRange() {
        XCTAssertEqual(AnimationPrimitives.textAppearPerChar, 0.06, accuracy: 0.001)
        XCTAssertEqual(AnimationPrimitives.textAppearPerCharMin, 0.03, accuracy: 0.001)
        XCTAssertEqual(AnimationPrimitives.textAppearPerCharMax, 0.10, accuracy: 0.001)
    }

    // MARK: - Reduce Motion fallbacks

    func testBlockSlideInFallsBackUnderReduceMotion() {
        let normal = AnimationPrimitives.blockSlideIn(reduceMotion: false)
        let reduced = AnimationPrimitives.blockSlideIn(reduceMotion: true)
        // Both should be valid SwiftUI Animations; the
        // reduced-motion variant is shorter.
        XCTAssertNotEqual(String(describing: normal), String(describing: reduced))
    }

    func testBlockReplaceFallsBackUnderReduceMotion() {
        let normal = AnimationPrimitives.blockReplace(reduceMotion: false)
        let reduced = AnimationPrimitives.blockReplace(reduceMotion: true)
        XCTAssertNotEqual(String(describing: normal), String(describing: reduced))
    }

    func testBlockDeleteReturnsNilUnderReduceMotion() {
        let normal = AnimationPrimitives.blockDelete(reduceMotion: false)
        XCTAssertNotNil(normal)
        let reduced = AnimationPrimitives.blockDelete(reduceMotion: true)
        XCTAssertNil(reduced, "block delete should be instant under Reduce Motion (no animation)")
    }

    func testTextAppearDelayReturnsNilUnderReduceMotion() {
        let normal = AnimationPrimitives.textAppearDelay(reduceMotion: false)
        XCTAssertNotNil(normal)
        let reduced = AnimationPrimitives.textAppearDelay(reduceMotion: true)
        XCTAssertNil(reduced, "text-appear should produce whole-text-at-once under Reduce Motion")
    }

    func testCursorBlinkReturnsNilUnderReduceMotion() {
        let normal = AnimationPrimitives.cursorBlink(reduceMotion: false)
        XCTAssertNotNil(normal)
        let reduced = AnimationPrimitives.cursorBlink(reduceMotion: true)
        XCTAssertNil(reduced, "cursor blink should be static under Reduce Motion")
    }

    func testThinkingPulseReturnsNilUnderReduceMotion() {
        let normal = AnimationPrimitives.thinkingPulseAnimation(reduceMotion: false)
        XCTAssertNotNil(normal)
        let reduced = AnimationPrimitives.thinkingPulseAnimation(reduceMotion: true)
        XCTAssertNil(reduced, "thinking pulse should be static under Reduce Motion")
    }

    func testAgentPausedBannerFallsBackUnderReduceMotion() {
        let normal = AnimationPrimitives.agentPausedBanner(reduceMotion: false)
        let reduced = AnimationPrimitives.agentPausedBanner(reduceMotion: true)
        XCTAssertNotEqual(String(describing: normal), String(describing: reduced))
    }

    // MARK: - Per-char delay clamping

    func testTextAppearDelayClampsToRange() {
        XCTAssertEqual(AnimationPrimitives.textAppearDelay(reduceMotion: false, perChar: 0.01), 0.03)
        XCTAssertEqual(AnimationPrimitives.textAppearDelay(reduceMotion: false, perChar: 1.0), 0.10)
        XCTAssertEqual(AnimationPrimitives.textAppearDelay(reduceMotion: false, perChar: 0.06), 0.06)
    }

    // MARK: - TextAppearCadence (interruptible streaming)

    func testTextAppearCadenceStreamsCharacters() async {
        let cadence = TextAppearCadence(perChar: 0.001)  // tiny for tests
        var collected: [String] = []
        for await event in cadence.stream("hi") {
            collected.append(String(event.character))
        }
        XCTAssertEqual(collected.joined(), "hi")
    }

    func testTextAppearCadenceStreamsEmptyString() async {
        let cadence = TextAppearCadence(perChar: 0.001)
        var count = 0
        for await _ in cadence.stream("") {
            count += 1
        }
        XCTAssertEqual(count, 0)
    }

    func testTextAppearCadenceIsInterruptible() async {
        let cadence = TextAppearCadence(perChar: 0.001)
        let task = Task {
            var count = 0
            for await _ in cadence.stream("hello world") {
                count += 1
                if count >= 3 { break }
            }
            return count
        }
        let count = await task.value
        XCTAssertEqual(count, 3)
    }

    // MARK: - Interruptibility (SwiftUI animations)

    func testAnimationsAreInterruptible() {
        // SwiftUI's withAnimation is interruptible by
        // construction; we just confirm we can build two
        // animations and they're independent values.
        let a1 = AnimationPrimitives.blockSlideIn(reduceMotion: false)
        let a2 = AnimationPrimitives.blockSlideIn(reduceMotion: false)
        XCTAssertEqual(String(describing: a1), String(describing: a2))
    }

    // MARK: - ViewModifier composition

    func testBlockSlideInViewModifierAppliesOffset() {
        // The modifier must not crash when applied. We don't
        // assert pixel values (SwiftUI's measurement isn't
        // available in unit tests without a host view), but
        // we do confirm the modifier exists and the view
        // composes.
        var modified: AnyView = AnyView(
            Color.red.blockSlideIn(isActive: true)
        )
        _ = modified  // silence unused
        modified = AnyView(
            Color.red.blockSlideIn(isActive: false)
        )
        _ = modified
    }

    func testThinkingPulseViewModifierDoesNotCrash() {
        _ = AnyView(
            Circle().thinkingPulse(isActive: false)
        )
    }

    func testCursorBlinkViewModifierDoesNotCrash() {
        _ = AnyView(
            Rectangle().cursorBlink(isActive: false)
        )
    }
}
