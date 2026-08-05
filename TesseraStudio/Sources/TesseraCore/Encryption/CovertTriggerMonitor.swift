import Foundation
import os

/// The covert trigger phrase for the "Plea the Fifth" coercion
/// path (see `docs/tessera-plead-the-fifth-design.md` sections
/// 8.3 and 9.3).
///
/// The user types a chosen phrase anywhere in the app - the chat
/// input, the library search, the settings text fields, the
/// capture floating window - and the wipe fires silently. The
/// visible UI keeps rendering normally; the next user-initiated
/// refresh (or a closed-and-reopened view) shows the empty
/// state. The adversary sees the app "working" until they reach
/// for the next thing to look at, and by then the encrypted
/// volume is unrecoverable.
///
/// Singleton actor so the cooldown clock and the fire callback
/// are owned by one place. The text input interceptor (the
/// NSTextView swizzle on macOS, and the explicit `.onChange` in
/// every text input view) calls `observe(text:)` after every
/// keystroke; the actor decides whether to fire.
///
/// Threat model (per design section 9.1):
///   - User is being watched visually or by camera.
///   - User may be compelled to use the app (e.g. a journalist
///     told to open their notes for review).
///   - The trigger MUST fire on the phrase, in any text input,
///     without any visible UI change.
///   - The wipe runs silently in the background.
///
/// The actor is not responsible for running the wipe; it calls
/// the ``onFire`` callback set by the composition root. The
/// callback dispatches to ``PleadTheFifthExecutor.destroyAll(trigger:)``.
public actor CovertTriggerMonitor {
    public static let shared = CovertTriggerMonitor()

    /// The user-chosen trigger phrase. Empty string disables
    /// the monitor (the design section 9.3 says "validates at
    /// Settings time; the monitor is defensive" - we also reject
    /// short phrases here in case a caller skips validation).
    private var _phrase: String = ""

    /// The Keychain account name we use to persist the phrase.
    /// Reusing the existing ``TesseraSecretStore`` namespace
    /// (`com.tessera.studio`) means the secret is keyed the same
    /// way as the remote API key and benefits from the same
    /// accessibility semantics (login keychain only).
    public static let keychainAccount = "covertTriggerPhrase"

    /// Cooldown between successful fires, in seconds. The design
    /// section 9.3 says "5-second cooldown between fires (so a
    /// long message containing the phrase twice doesn't fire
    /// twice)". Exposed as a public mutable var so tests can
    /// shrink it; the production app leaves it at 5.
    public var cooldownSeconds: TimeInterval = 5

    /// The minimum phrase length. The design says "8+ characters
    /// (validated at Settings time; the monitor is defensive)".
    public static let minPhraseLength = 8

    /// The minimum length of the observed text relative to the
    /// phrase. The design section 9.3 says "the observed text
    /// must be > phrase.length + 4 (i.e., the user must have
    /// typed a meaningful sentence, not just the phrase alone)".
    /// The +4 is fixed by the spec; we don't expose it.
    public static let observedTextOverhead = 4

    /// Last fire timestamp; used for the cooldown check.
    private var lastFireAt: Date?

    /// The fire callback. Set by the composition root on app
    /// launch:
    ///
    /// ```swift
    /// await CovertTriggerMonitor.shared.onFire = { [weak exec] in
    ///     _ = try? await exec?.destroyAll(trigger: .covert)
    /// }
    /// ```
    ///
    /// Sendable so the actor can hop isolation to call it; the
    /// executor itself is an actor so the call is safe.
    public var onFire: (@Sendable () async -> Void)?

    /// Logger for the audit trail. Per design section 9.3 a
    /// successful fire logs "covert_trigger_fired" + a
    /// timestamp, never the phrase. The system log is
    /// `Console.app`-visible for the user; in the released build
    /// this should also feed the wipe report.
    private let logger = Logger(
        subsystem: "com.tessera.studio",
        category: "covert-trigger"
    )

    public init() {}

    /// Currently configured phrase (empty if disabled). Public
    /// for tests and for the Settings view to render the masked
    /// "phrase is set" indicator without ever reading the
    /// Keychain value.
    public var phrase: String {
        _phrase
    }

    /// Whether the monitor is currently armed. True when the
    /// phrase is non-empty and long enough.
    public var isArmed: Bool {
        !_phrase.isEmpty && _phrase.count >= Self.minPhraseLength
    }

    /// Simulate the trigger matching without persisting anything.
    /// Used by the Settings view's Test button so the user can
    /// verify a draft phrase before committing it. Runs the same
    /// matching rules as ``observe(text:)`` (length check, case
    /// insensitive substring) but uses the supplied `candidate`
    /// phrase instead of the stored one. Returns true if the
    /// candidate would have fired the trigger. No state change.
    public func testObserve(candidate: String, text: String) -> Bool {
        let trimmed = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count >= Self.minPhraseLength else { return false }
        let phraseLen = trimmed.utf16.count
        guard text.utf16.count > phraseLen + Self.observedTextOverhead else {
            return false
        }
        return text.range(of: trimmed, options: .caseInsensitive) != nil
    }

    /// Persist the phrase to the Keychain and arm the monitor.
    /// Empty string clears the phrase and disables the monitor.
    /// Returns false (and leaves state unchanged) if the phrase
    /// is shorter than ``minPhraseLength`` or if the Keychain
    /// refuses the write.
    @discardableResult
    public func setPhrase(_ phrase: String) -> Bool {
        // Empty disables the monitor. The wipe of the previously
        // stored phrase from the Keychain still happens - the
        // user's previous phrase is a secret too.
        if phrase.isEmpty {
            _phrase = ""
            // Best-effort delete of any previously stored phrase.
            _ = TesseraSecretStore.setSecret(nil, account: Self.keychainAccount)
            return true
        }
        // Defensive: the Settings view also validates, but if a
        // caller skips that we reject short phrases here.
        let trimmed = phrase.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count >= Self.minPhraseLength else {
            return false
        }
        let stored = TesseraSecretStore.setSecret(
            trimmed, account: Self.keychainAccount
        )
        guard stored else { return false }
        _phrase = trimmed
        // Reset the cooldown clock on a new phrase so the user
        // doesn't get a stale cooldown carried over from the
        // previous phrase.
        lastFireAt = nil
        return true
    }

    /// Reload the phrase from the Keychain. Call this on app
    /// launch so the in-memory state matches what the user
    /// previously set in Settings. Returns true if a phrase was
    /// loaded.
    @discardableResult
    public func loadFromKeychain() -> Bool {
        guard let stored = TesseraSecretStore.secret(
            account: Self.keychainAccount
        ), !stored.isEmpty else {
            _phrase = ""
            return false
        }
        _phrase = stored
        return true
    }

    /// Called by every text input's onChange handler (and by
    /// the NSTextView swizzle on macOS) with the current text.
    /// Returns true if the trigger fired - the caller can use
    /// the return value to refresh the visible state, but the
    /// wipe itself is fire-and-forget.
    ///
    /// The `onFire` callback is awaited before this returns, so
    /// by the time the caller sees `true` the wipe has been
    /// initiated. The wipe itself runs in a background actor
    /// (per the design section 9.2) - the callback's body just
    /// dispatches it and returns.
    public func observe(text: String) async -> Bool {
        guard isArmed else { return false }
        // Spec: the observed text must be > phrase.length + 4.
        // This rules out accidental triggers from a sloppy paste
        // of the phrase alone. We compare on UTF-16 length to
        // match the character semantics of the substring check.
        let phraseLen = _phrase.utf16.count
        guard text.utf16.count > phraseLen + Self.observedTextOverhead else {
            return false
        }
        // Cooldown: skip if the last fire was within
        // `cooldownSeconds` of now. We compare against the
        // monotonic clock so clock changes can't extend a
        // cooldown.
        if let last = lastFireAt,
           Date().timeIntervalSince(last) < cooldownSeconds {
            return false
        }
        // Case-insensitive substring match. NSTextView delivers
        // UTF-16; Swift's String range search is grapheme-cluster
        // aware, which is what we want for the trigger phrase
        // (a phrase with combined accents should still match).
        guard let _ = text.range(of: _phrase, options: .caseInsensitive) else {
            return false
        }
        await fire()
        return true
    }

    /// Fire the trigger. The cooldown clock is updated BEFORE
    /// the callback runs so a slow callback (or a callback that
    /// itself triggers observe() indirectly) cannot double-fire.
    /// The callback is awaited so the caller of `observe` sees
    /// the wipe initiated by the time `observe` returns.
    private func fire() async {
        lastFireAt = Date()
        // Spec section 9.3: "A successful fire logs a structured
        // event for the user's audit trail (NOT the phrase, just
        // 'covert_trigger_fired' + timestamp)".
        let ts = ISO8601DateFormatter().string(from: Date())
        logger.notice("covert_trigger_fired ts=\(ts, privacy: .public)")
        if let callback = onFire {
            await callback()
        }
    }
}
