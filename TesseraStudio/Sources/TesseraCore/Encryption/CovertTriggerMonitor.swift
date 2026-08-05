import Foundation

/// Substring-based covert trigger monitor. Every text input the user
/// types is passed to ``check(_:)`` (typically via a SwiftUI
/// `.onChange(of: text)` handler). When the configured phrase appears
/// as a substring of the input, the monitor returns `true` and the
/// caller is expected to fire the wipe.
///
/// Design properties (see docs/tessera-plead-the-fifth-design.md
/// section 9.3):
/// - Substring match (the phrase can appear inside a longer sentence)
/// - Case-insensitive
/// - The phrase must be at least
///   ``PleadTheFifthSettings.minCovertPhraseLength`` characters
/// - The input must be longer than the phrase + 4 characters (so a
///   pure paste of the phrase alone does not fire)
/// - 5-second cooldown between fires prevents the same phrase typed
///   in two windows from double-firing
public struct CovertTriggerMonitor: Sendable {

    public init() {}

    /// Returns the configured trigger phrase, lowercased, or nil if
    /// none is configured.
    public var configuredPhrase: String? {
        let phrase = PleadTheFifthSettings.covertTriggerPhrase
        guard !phrase.isEmpty, phrase.count >= PleadTheFifthSettings.minCovertPhraseLength
        else { return nil }
        return phrase.lowercased()
    }

    /// Evaluate the latest text in a text input. Returns `true` when
    /// the trigger should fire RIGHT NOW (i.e. the phrase is present
    /// and the cooldown is satisfied). The caller is responsible for
    /// recording the fire via ``recordFire()`` if it actually runs
    /// the wipe, so the cooldown reflects reality.
    public func shouldTrigger(in text: String) -> Bool {
        guard let phrase = configuredPhrase else { return false }
        let haystack = text.lowercased()
        guard haystack.contains(phrase) else { return false }
        // The phrase must appear inside a longer string. The design's
        // spec is "phrase + 4 characters" to defeat a paste of just
        // the phrase.
        guard text.count >= phrase.count + 4 else { return false }
        if let last = PleadTheFifthSettings.lastCovertTriggerAt,
           Date().timeIntervalSince(last) < PleadTheFifthSettings.covertTriggerCooldownSeconds {
            return false
        }
        return true
    }

    /// Record that a trigger has fired. The next call to
    /// ``shouldTrigger(in:)`` within the cooldown window returns
    /// false even if the phrase is still in the input.
    public func recordFire(at date: Date = Date()) {
        PleadTheFifthSettings.recordCovertTriggerFire(at: date)
    }

    /// A failed-attempt counter helper. The caller is expected to
    /// call this on every text change where the trigger WOULD have
    /// fired but the cooldown is active, so the cap is enforced and
    /// the audit trail is honest.
    public func recordFailedAttempt() {
        PleadTheFifthSettings.incrementFailedCovertTriggerAttempts()
    }
}
