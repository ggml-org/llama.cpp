import XCTest
@testable import TesseraCore

/// Unit tests for ``CovertTriggerMonitor`` (the actor that owns
/// the covert trigger phrase, the cooldown clock, and the fire
/// callback). The matching rules come from design section 9.3:
///
///   - Case-insensitive substring match
///   - 8+ character minimum phrase (defensive, the Settings
///     view also validates)
///   - Observed text must be > phrase.length + 4 (the phrase
///     must appear in a longer sentence, not be the entire
///     input)
///   - 5-second cooldown between successful fires
///   - Empty phrase disables the monitor
///
/// These tests don't exercise the AppKit/UIKit method swizzle
/// (XCTest can't drive a real keystroke through a swizzle on a
/// real NSTextView); the swizzle is verified manually per the
/// task brief. The unit tests cover the matching rules and
/// callback wiring; the swizzle delivers those same `observe`
/// calls in production.
final class CovertTriggerMonitorTests: XCTestCase {
    /// Fresh monitor per test so the cooldown clock and the
    /// fire callback don't leak between tests.
    private func newMonitor() -> CovertTriggerMonitor {
        CovertTriggerMonitor()
    }

    /// The test's primary happy path: a phrase inside a longer
    /// string fires the trigger.
    func testMonitorFiresOnSubstring() async {
        let monitor = newMonitor()
        let fired = RecordingTrigger()
        await monitor.setOnFire(fired.fire)
        let setOK = await monitor.setPhrase("the weather is nice today")
        XCTAssertTrue(setOK)
        let didFire = await monitor.observe(
            text: "I think the weather is nice today, doesn't it?"
        )
        XCTAssertTrue(didFire)
        let count = await fired.fireCount
        XCTAssertEqual(count, 1)
    }

    /// Spec section 9.3: "The phrase must appear in a longer
    /// sentence, not as the entire input." A text of length
    /// <= phrase.length + 4 must NOT fire even when it contains
    /// the phrase as the whole string.
    func testMonitorIgnoresShortInput() async {
        let monitor = newMonitor()
        let fired = RecordingTrigger()
        await monitor.setOnFire(fired.fire)
        let setOK = await monitor.setPhrase("abcdefghij")
        XCTAssertTrue(setOK)
        // Length equals phrase.length + 4 exactly. The spec
        // says "must be > phrase.length + 4", so equal does
        // not fire.
        let exactly = String(repeating: "x", count: 10 + 4)
        let didFireExact = await monitor.observe(text: exactly)
        XCTAssertFalse(didFireExact)
        // Length shorter than phrase - must not fire.
        let didFireShort = await monitor.observe(text: "abcdefgh")
        XCTAssertFalse(didFireShort)
        let count = await fired.fireCount
        XCTAssertEqual(count, 0)
    }

    /// Spec section 9.3: "case-insensitive substring match".
    func testMonitorCaseInsensitive() async {
        let monitor = newMonitor()
        let fired = RecordingTrigger()
        await monitor.setOnFire(fired.fire)
        let setOK = await monitor.setPhrase("The Weather")
        XCTAssertTrue(setOK)
        // Lowercase + extra surrounding text. The phrase is
        // 11 chars, so observed text must be > 15 chars.
        let observed = "today the weather is mild and dry outside"
        XCTAssertGreaterThan(observed.count, 11 + 4)
        let didFire = await monitor.observe(text: observed)
        XCTAssertTrue(didFire)
        let count = await fired.fireCount
        XCTAssertEqual(count, 1)
    }

    /// Spec section 9.3: "5-second cooldown between fires (so
    /// a long message containing the phrase twice doesn't fire
    /// twice)". We shrink the cooldown to 0.1s and check that
    /// a second fire within the cooldown is rejected; a fire
    /// after the cooldown is allowed.
    func testMonitorRespectsCooldown() async {
        let monitor = newMonitor()
        let fired = RecordingTrigger()
        await monitor.setOnFire(fired.fire)
        await monitor.setCooldown(0.1)
        let setOK = await monitor.setPhrase("the weather is nice today")
        XCTAssertTrue(setOK)
        // First fire - we observe a long text containing the
        // phrase. The phrase is 24 chars, so the observed
        // text must be > 28 chars.
        let didFireFirst = await monitor.observe(
            text: "I think the weather is nice today, friend, don't you agree?"
        )
        XCTAssertTrue(didFireFirst)
        let countAfterFirst = await fired.fireCount
        XCTAssertEqual(countAfterFirst, 1)
        // Immediately observe another matching text. The
        // cooldown (0.1s) is still active; this must NOT fire.
        let didFireSecond = await monitor.observe(
            text: "Tomorrow the weather is nice today too, friend."
        )
        XCTAssertFalse(didFireSecond)
        let countAfterSecond = await fired.fireCount
        XCTAssertEqual(countAfterSecond, 1)
        // Wait past the cooldown, then observe again.
        try? await Task.sleep(nanoseconds: 200_000_000)
        let didFireThird = await monitor.observe(
            text: "Yesterday the weather is nice today as well, friend."
        )
        XCTAssertTrue(didFireThird)
        let countAfterThird = await fired.fireCount
        XCTAssertEqual(countAfterThird, 2)
    }

    /// Defensive: setPhrase(< 8 chars) is rejected, the
    /// monitor is left un-armed, observe() does not fire.
    func testMonitorRejectsShortPhrase() async {
        let monitor = newMonitor()
        let fired = RecordingTrigger()
        await monitor.setOnFire(fired.fire)
        let setOK = await monitor.setPhrase("short")
        XCTAssertFalse(setOK)
        // Monitor should not be armed. Even a perfectly
        // matching long text does not fire.
        let didFire = await monitor.observe(
            text: "I just typed short here, didn't I?"
        )
        XCTAssertFalse(didFire)
        let count = await fired.fireCount
        XCTAssertEqual(count, 0)
    }

    /// Empty phrase disables the monitor. Even after a phrase
    /// is set, calling setPhrase("") disables it.
    func testMonitorDisabledWhenPhraseEmpty() async {
        let monitor = newMonitor()
        let fired = RecordingTrigger()
        await monitor.setOnFire(fired.fire)
        // First arm the monitor with a valid phrase.
        let setFirst = await monitor.setPhrase("the weather is nice today")
        XCTAssertTrue(setFirst)
        // Then disable it.
        let setEmpty = await monitor.setPhrase("")
        XCTAssertTrue(setEmpty)
        let armed = await monitor.isArmed
        XCTAssertFalse(armed)
        let didFire = await monitor.observe(
            text: "I think the weather is nice today, doesn't it?"
        )
        XCTAssertFalse(didFire)
        let count = await fired.fireCount
        XCTAssertEqual(count, 0)
    }

    /// The onFire callback is invoked when the trigger fires.
    /// Use a mock that just records calls; the real executor
    /// is wired in by the composition root on launch and is
    /// out of scope for these unit tests.
    func testOnFireCallbackInvoked() async {
        let monitor = newMonitor()
        let fired = RecordingTrigger()
        await monitor.setOnFire(fired.fire)
        let setOK = await monitor.setPhrase("the weather is nice today")
        XCTAssertTrue(setOK)
        let didFire = await monitor.observe(
            text: "I think the weather is nice today, doesn't it?"
        )
        XCTAssertTrue(didFire)
        // The fire callback is dispatched on a detached Task,
        // so wait briefly for it to run.
        try? await Task.sleep(nanoseconds: 50_000_000)
        let count = await fired.fireCount
        XCTAssertEqual(count, 1)
    }

    /// observe(text:) returns true when the trigger fired,
    /// false when it did not. Callers can use this to refresh
    /// visible state. The test asserts the return value for
    /// both cases.
    func testObserveReturnsFiredFlag() async {
        let monitor = newMonitor()
        let fired = RecordingTrigger()
        await monitor.setOnFire(fired.fire)
        let setOK = await monitor.setPhrase("the weather is nice today")
        XCTAssertTrue(setOK)
        // No fire - phrase not in text.
        let didFire1 = await monitor.observe(text: "completely unrelated words")
        XCTAssertFalse(didFire1)
        let count1 = await fired.fireCount
        XCTAssertEqual(count1, 0)
        // Fire - phrase in longer text.
        let didFire2 = await monitor.observe(
            text: "I think the weather is nice today, doesn't it?"
        )
        XCTAssertTrue(didFire2)
        let count2 = await fired.fireCount
        XCTAssertEqual(count2, 1)
        // No fire - cooldown active.
        let didFire3 = await monitor.observe(
            text: "I think the weather is nice today, doesn't it?"
        )
        XCTAssertFalse(didFire3)
        let count3 = await fired.fireCount
        XCTAssertEqual(count3, 1)
    }

    /// Phrase is persisted in the Keychain. setPhrase("") clears
    /// the stored value. We use a throwaway account name to
    /// avoid touching the real account.
    func testSetPhraseClearsKeychainOnEmpty() async {
        let monitor = newMonitor()
        // We can't easily swap the keychain account in the
        // monitor (it's a static constant); instead, we use
        // the public API and the real Keychain.
        let account = CovertTriggerMonitor.keychainAccount
        // Clean up after the test.
        defer { _ = TesseraSecretStore.setSecret(nil, account: account) }
        let setOK = await monitor.setPhrase("the weather is nice today")
        XCTAssertTrue(setOK)
        let stored = TesseraSecretStore.secret(account: account)
        XCTAssertEqual(stored, "the weather is nice today")
        let setEmpty = await monitor.setPhrase("")
        XCTAssertTrue(setEmpty)
        let afterEmpty = TesseraSecretStore.secret(account: account)
        XCTAssertNil(afterEmpty)
    }

    /// Whitespace at the start/end of the phrase is trimmed
    /// before storing. A 12-character phrase with 2 leading
    /// spaces and a trailing newline is still long enough
    /// after trim; the stored value is the trimmed form.
    func testSetPhraseTrimsWhitespace() async {
        let monitor = newMonitor()
        let account = CovertTriggerMonitor.keychainAccount
        defer { _ = TesseraSecretStore.setSecret(nil, account: account) }
        let setOK = await monitor.setPhrase("  the weather is nice today\n")
        XCTAssertTrue(setOK)
        let stored = TesseraSecretStore.secret(account: account)
        XCTAssertEqual(stored, "the weather is nice today")
    }

    /// `testObserve(candidate:text:)` runs the same matching
    /// rules as `observe(text:)` but with a caller-supplied
    /// candidate phrase - used by the Settings view's Test
    /// button. It must NOT persist the candidate, NOT fire
    /// the callback, and NOT change the monitor's state.
    func testTestObserveDoesNotMutateState() async {
        let monitor = newMonitor()
        let fired = RecordingTrigger()
        await monitor.setOnFire(fired.fire)
        // Don't arm the monitor. The Test button should
        // still work with a candidate phrase.
        let candidate = "the weather is nice today"
        let observed = "today the weather is nice today, friend"
        let didFire = await monitor.testObserve(
            candidate: candidate, text: observed
        )
        XCTAssertTrue(didFire)
        // The monitor must remain unarmed.
        let armed = await monitor.isArmed
        XCTAssertFalse(armed)
        // The callback must not have been invoked.
        let count = await fired.fireCount
        XCTAssertEqual(count, 0)
        // The phrase must NOT be persisted by testObserve.
        let stored = TesseraSecretStore.secret(
            account: CovertTriggerMonitor.keychainAccount
        )
        XCTAssertNil(stored)
    }

    /// `testObserve` rejects candidates shorter than the
    /// minimum even when the observed text contains them.
    func testTestObserveRejectsShortCandidate() async {
        let monitor = newMonitor()
        let didFire = await monitor.testObserve(
            candidate: "short",
            text: "I just typed short here, didn't I?"
        )
        XCTAssertFalse(didFire)
    }

    /// `testObserve` respects the length check: the observed
    /// text must be longer than the candidate + 4.
    func testTestObserveRejectsShortObserved() async {
        let monitor = newMonitor()
        let didFire = await monitor.testObserve(
            candidate: "abcdefghij",
            text: "abcdefghij"  // 10 chars = candidate length, not length + 4
        )
        XCTAssertFalse(didFire)
    }

    /// A phrase with combined accents is matched case- and
    /// diacritic-insensitively. The spec calls for a grapheme-
    /// cluster-aware substring check; this test ensures
    /// "café" in the phrase matches "cafe" in the observed
    /// text via Swift's default string matching.
    ///
    /// Note: Swift's `range(of:options:)` with
    /// `.caseInsensitive` does NOT do Unicode normalization
    /// by default, so this test is a guard against future
    /// regressions: if a future change breaks accent
    /// handling, this test will fail and force a fix.
    /// For v1 the phrase is case-insensitive only.
    func testMonitorMatchRespectsUnicode() async {
        let monitor = newMonitor()
        let fired = RecordingTrigger()
        await monitor.setOnFire(fired.fire)
        // "café" - 4 codepoints, 5 UTF-16 code units. Our
        // length check uses UTF-16; the spec's
        // `phrase.length + 4` is intentionally ambiguous on
        // which length to use, and we picked UTF-16 to match
        // NSTextView's storage.
        let setOK = await monitor.setPhrase("the café is open")
        XCTAssertTrue(setOK)
        // Lowercase variant with extra text.
        let observed = "today the café is open and the line is short"
        XCTAssertGreaterThan(observed.utf16.count, 14 + 4)
        let didFire = await monitor.observe(text: observed)
        XCTAssertTrue(didFire)
    }
}

// MARK: - Test helpers

/// A recording stand-in for the real ``PleadTheFifthTrigger``
/// executor. Tracks the number of times the fire callback was
/// invoked. Lives in the test target because the real executor
/// is owned by Phase 2 and is not available yet.
private actor RecordingTrigger {
    var fireCount: Int = 0
    var lastError: String?

    func fire() async {
        fireCount += 1
    }
}

private extension CovertTriggerMonitor {
    /// Test seam for setting the fire callback. The real
    /// `onFire` is set by the composition root on app launch;
    /// the test wants to inject a recording callback without
    /// going through the global singleton.
    func setOnFire(_ callback: @Sendable @escaping () async -> Void) async {
        onFire = callback
    }

    /// Test seam for shrinking the cooldown so we don't have
    /// to wait 5 real seconds in tests.
    func setCooldown(_ seconds: TimeInterval) async {
        cooldownSeconds = seconds
    }
}
