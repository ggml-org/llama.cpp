import Foundation

/// UserDefaults keys and helpers for the "Plea the Fifth" feature.
/// Separate from ``TesseraSettings`` so the privacy-critical settings
/// can be audited independently and so the executor's view of state
/// stays focused.
public enum PleadTheFifthSettingsKey {
    /// User-chosen covert trigger phrase. Empty means the feature is
    /// off. The phrase must be at least
    /// ``PleadTheFifthSettings/minCovertPhraseLength`` characters;
    /// shorter values are rejected at write time.
    public static let covertTriggerPhrase = "tessera.pleadTheFifth.covertTriggerPhrase"
    /// Coercion mode toggle. When on, the menu bar item is hidden
    /// and only the hot-key + covert trigger remain.
    public static let coercionMode = "tessera.pleadTheFifth.coercionMode"
    /// Last successful covert trigger fire time (Date.timeIntervalSince1970
    /// as a Double). Used to enforce the 5-second cooldown so a
    /// phrase typed in two windows does not double-fire.
    public static let lastCovertTriggerAt = "tessera.pleadTheFifth.lastCovertTriggerAt"
    /// Number of consecutive failed covert trigger attempts. Capped
    /// at 3 per session per the design; the 4th attempt is logged to
    /// the user's audit trail.
    public static let failedCovertTriggerAttempts = "tessera.pleadTheFifth.failedCovertTriggerAttempts"
}

/// Defaults and validation for the "Plea the Fifth" settings.
public enum PleadTheFifthSettings {
    /// Minimum length for a covert trigger phrase. 8 characters is the
    /// design's recommendation; shorter phrases have too high a false-
    /// positive rate on natural text.
    public static let minCovertPhraseLength = 8

    /// Cooldown between covert trigger fires. Prevents double-fires
    /// when the same phrase is typed in two windows.
    public static let covertTriggerCooldownSeconds: TimeInterval = 5

    /// Cap on consecutive failed covert trigger attempts before the
    /// 4th attempt is logged to the audit trail. The user's privacy
    /// page documents the cap.
    public static let failedAttemptCap = 3

    /// The current covert trigger phrase, or empty if the feature is
    /// off. Always read fresh from UserDefaults so a Settings change
    /// is honored immediately.
    public static var covertTriggerPhrase: String {
        UserDefaults.standard.string(forKey: PleadTheFifthSettingsKey.covertTriggerPhrase) ?? ""
    }

    public static func setCovertTriggerPhrase(_ phrase: String) throws {
        let trimmed = phrase.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty && trimmed.count < minCovertPhraseLength {
            throw PleadTheFifthSettingsError.phraseTooShort(
                minLength: minCovertPhraseLength, got: trimmed.count
            )
        }
        UserDefaults.standard.set(trimmed, forKey: PleadTheFifthSettingsKey.covertTriggerPhrase)
    }

    public static var coercionMode: Bool {
        UserDefaults.standard.bool(forKey: PleadTheFifthSettingsKey.coercionMode)
    }

    public static func setCoercionMode(_ on: Bool) {
        UserDefaults.standard.set(on, forKey: PleadTheFifthSettingsKey.coercionMode)
    }

    /// Whether the covert trigger is configured (a non-empty phrase of
    /// at least the minimum length is set).
    public static var covertTriggerConfigured: Bool {
        let phrase = covertTriggerPhrase
        return !phrase.isEmpty && phrase.count >= minCovertPhraseLength
    }

    /// Last covert trigger fire time, or nil if it has never fired.
    public static var lastCovertTriggerAt: Date? {
        let raw = UserDefaults.standard.double(forKey: PleadTheFifthSettingsKey.lastCovertTriggerAt)
        guard raw > 0 else { return nil }
        return Date(timeIntervalSince1970: raw)
    }

    public static func recordCovertTriggerFire(at date: Date = Date()) {
        UserDefaults.standard.set(
            date.timeIntervalSince1970,
            forKey: PleadTheFifthSettingsKey.lastCovertTriggerAt
        )
    }

    public static var failedCovertTriggerAttempts: Int {
        UserDefaults.standard.integer(forKey: PleadTheFifthSettingsKey.failedCovertTriggerAttempts)
    }

    public static func incrementFailedCovertTriggerAttempts() {
        let next = failedCovertTriggerAttempts + 1
        UserDefaults.standard.set(
            next,
            forKey: PleadTheFifthSettingsKey.failedCovertTriggerAttempts
        )
    }

    public static func resetFailedCovertTriggerAttempts() {
        UserDefaults.standard.set(
            0,
            forKey: PleadTheFifthSettingsKey.failedCovertTriggerAttempts
        )
    }
}

public enum PleadTheFifthSettingsError: Error, LocalizedError {
    case phraseTooShort(minLength: Int, got: Int)

    public var errorDescription: String? {
        switch self {
        case .phraseTooShort(let min, let got):
            return "Covert trigger phrase must be at least \(min) characters (got \(got))."
        }
    }
}
