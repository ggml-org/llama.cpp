import Foundation

// MARK: - ReminderCommandParser

/// Best-effort parser for natural-language reminder commands.
/// The chat panel's "remind me 15 min before the Q3 review
/// meeting" intent gets translated into a
/// ``ParsedReminderCommand`` with the offset (in minutes) and
/// a fuzzy event-title match. The agent then resolves the
/// title to a calendar event id via ``ReminderStore`` +
/// the data layer.
///
/// The parser is intentionally simple — it does NOT try to
/// be clever. The intent types are:
///   * "remind me <n> min/hour[s] [before|after] <event title>"
///   * "remind me at <time>"
///   * "list my reminders"
///   * "what are my reminders"
///   * "dismiss the <event title> reminder"
///   * "snooze the <event title> reminder for <n> min"
///
/// Anything that doesn't match returns ``nil`` from
/// ``parse(_:)``. The agent falls back to asking the user
/// for clarification.
///
/// **Case preservation.** The parser does its matching on
/// a lowercased copy of the input but returns the
/// case-preserved fragment. The agent's fuzzy-match step
/// against `graph_entities.label` works best when the
/// fragment's case matches the user's input.
public struct ReminderCommandParser: Sendable {

    public init() {}

    /// One parsed intent. The chat panel's agent loop uses
    /// the `kind` to pick the right tool, then fills the
    /// tool's parameters from the remaining fields.
    public enum Kind: String, Sendable, Hashable {
        case create
        case list
        case dismiss
        case snooze
    }

    public struct ParsedReminderCommand: Sendable, Hashable {
        public let kind: Kind
        /// Minutes relative to the linked event's start.
        /// Negative = before, positive = after, 0 = at start.
        public let offsetMinutes: Int?
        /// The event title fragment the user mentioned. The
        /// agent uses this to fuzzy-match a calendar event.
        public let eventTitleFragment: String?
        /// How long to snooze, in minutes. Set when `kind` is
        /// `.snooze`.
        public let snoozeMinutes: Int?
        /// Raw user input — preserved for the receipt.
        public let rawInput: String

        public init(
            kind: Kind,
            offsetMinutes: Int? = nil,
            eventTitleFragment: String? = nil,
            snoozeMinutes: Int? = nil,
            rawInput: String
        ) {
            self.kind = kind
            self.offsetMinutes = offsetMinutes
            self.eventTitleFragment = eventTitleFragment
            self.snoozeMinutes = snoozeMinutes
            self.rawInput = rawInput
        }
    }

    /// Parse a chat-panel user input. Returns nil when the
    /// input doesn't look like a reminder command. The
    /// matching is case-insensitive and tolerates extra
    /// whitespace.
    public func parse(_ input: String) -> ParsedReminderCommand? {
        let trimmed = input.trimmingCharacters(
            in: .whitespacesAndNewlines
        )
        guard !trimmed.isEmpty else { return nil }
        let lower = trimmed.lowercased()

        // List intents.
        if lower.contains("list my reminders")
            || lower.contains("show my reminders")
            || lower.contains("what are my reminders")
            || lower.contains("what reminders")
            || lower == "reminders" {
            return ParsedReminderCommand(kind: .list, rawInput: trimmed)
        }

        // Dismiss intents.
        if let fragment = extractFragment(
            after: ["dismiss", "acknowledge", "mark"],
            inLower: lower,
            casePreserved: trimmed
        ) {
            let cleaned = stripReminderSuffix(from: fragment)
            if !cleaned.isEmpty {
                return ParsedReminderCommand(
                    kind: .dismiss,
                    eventTitleFragment: cleaned,
                    rawInput: trimmed
                )
            }
        }

        // Snooze intents.
        if let fragment = extractFragment(
            after: ["snooze"],
            inLower: lower,
            casePreserved: trimmed
        ) {
            // "snooze the X reminder for 10 min"
            let minutes = extractMinutes(
                inLower: fragment.lowercased(),
                casePreserved: fragment
            )
            let cleaned = stripReminderSuffix(
                from: minutes.remainderCasePreserved
            )
            if !cleaned.isEmpty {
                return ParsedReminderCommand(
                    kind: .snooze,
                    eventTitleFragment: cleaned,
                    snoozeMinutes: minutes.value ?? 10,
                    rawInput: trimmed
                )
            }
        }

        // Create intents. "remind me …" or "set a reminder …".
        if let rest = stripPrefix(
            ["remind me to", "remind me", "set a reminder to", "set a reminder for", "set a reminder"],
            inLower: lower,
            casePreserved: trimmed
        ) {
            if let parsed = parseCreate(
                restLower: rest,
                restCasePreserved: rest,
                raw: trimmed
            ) {
                return parsed
            }
        }

        return nil
    }

    // MARK: - Helpers

    /// A pair of strings that share an index range: the
    /// lowercased version for matching, the case-preserved
    /// version for returning to the user.
    private struct FragmentMatch: Sendable, Hashable {
        let lowercased: String
        let casePreserved: String
    }

    private func stripPrefix(
        _ prefixes: [String],
        inLower lower: String,
        casePreserved original: String
    ) -> FragmentMatch? {
        for p in prefixes {
            if lower.hasPrefix(p) {
                let rLower = lower.dropFirst(p.count)
                    .trimmingCharacters(in: .whitespaces)
                let rCase = original.dropFirst(p.count)
                    .trimmingCharacters(in: .whitespaces)
                if !rLower.isEmpty {
                    return FragmentMatch(
                        lowercased: String(rLower),
                        casePreserved: String(rCase)
                    )
                }
            }
        }
        return nil
    }

    /// Find the fragment that follows one of the given
    /// verbs. Returns the case-preserved fragment (not the
    /// lowercased one). The lowercased string is used for
    /// the verb match; the same character offset in the
    /// original string is returned to the caller.
    private func extractFragment(
        after verbs: [String],
        inLower lower: String,
        casePreserved original: String
    ) -> String? {
        for v in verbs {
            guard let r = lower.range(
                of: "\\b\(NSRegularExpression.escapedPattern(for: v))\\b",
                options: .regularExpression
            ) else { continue }
            // Compute the same offset in the original
            // (case-preserved) string. For chat input this
            // is all ASCII, so Character.distance and
            // utf16.distance give the same answer.
            let lowerOffset = lower.distance(
                from: lower.startIndex, to: r.upperBound
            )
            guard lowerOffset <= original.count else { continue }
            let originalIdx = original.index(
                original.startIndex,
                offsetBy: lowerOffset
            )
            let afterCase = String(original[originalIdx...])
                .trimmingCharacters(in: .whitespaces)
            // Strip a leading "the" so "dismiss the X" yields
            // "X" rather than "the X".
            if afterCase.lowercased().hasPrefix("the ") {
                return String(afterCase.dropFirst(4))
            }
            return afterCase
        }
        return nil
    }

    private func stripReminderSuffix(from s: String) -> String {
        var out = s
        let suffixes = ["reminder", "reminders"]
        for suf in suffixes {
            // Case-insensitive suffix check.
            if out.lowercased().hasSuffix(suf) {
                out = String(out.dropLast(suf.count))
                    .trimmingCharacters(in: .whitespaces)
            }
        }
        return out
    }

    private struct ExtractedMinutes: Sendable, Hashable {
        let value: Int?
        let remainderLower: String
        let remainderCasePreserved: String
    }

    /// Look for a "<n> min|minute[s]|hour[s]" anywhere in
    /// the lowercased string. Returns the parsed value and
    /// the input with the matched phrase removed from both
    /// the lowercased and case-preserved versions. An
    /// optional leading "for" is consumed too — snooze
    /// inputs read "snooze the X reminder for 10 min" and
    /// we want the "for" to be gone from the title.
    private func extractMinutes(
        inLower s: String,
        casePreserved sCase: String
    ) -> ExtractedMinutes {
        let pattern = #"(?:for\s+)?(\d+)\s*(minutes?|mins?|hours?|hrs?)\b"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) else {
            return ExtractedMinutes(value: nil, remainderLower: s, remainderCasePreserved: sCase)
        }
        let range = NSRange(s.startIndex..., in: s)
        guard let match = regex.firstMatch(in: s, options: [], range: range),
              let numberRange = Range(match.range(at: 1), in: s),
              let unitRange = Range(match.range(at: 2), in: s) else {
            return ExtractedMinutes(value: nil, remainderLower: s, remainderCasePreserved: sCase)
        }
        let n = Int(s[numberRange]) ?? 0
        let unit = String(s[unitRange]).lowercased()
        let minutes: Int
        if unit.hasPrefix("h") {
            minutes = n * 60
        } else {
            minutes = n
        }
        // Remove the matched phrase from the lowercased
        // string (where the match was found).
        let matchStart = s.index(s.startIndex, offsetBy: match.range.location)
        let matchEnd = s.index(s.startIndex, offsetBy: match.range.location + match.range.length)
        let remainderLower = s.replacingCharacters(in: matchStart..<matchEnd, with: "")
            .trimmingCharacters(in: .whitespaces)
        // Remove the same Character range from the
        // case-preserved string. For chat input this is
        // all ASCII so Character distances match.
        let caseStartIdx = sCase.index(
            sCase.startIndex,
            offsetBy: match.range.location
        )
        let caseEndIdx = sCase.index(
            sCase.startIndex,
            offsetBy: match.range.location + match.range.length
        )
        let remainderCase = sCase.replacingCharacters(
            in: caseStartIdx..<caseEndIdx, with: ""
        ).trimmingCharacters(in: .whitespaces)
        return ExtractedMinutes(
            value: minutes,
            remainderLower: remainderLower,
            remainderCasePreserved: remainderCase
        )
    }

    private func parseCreate(
        restLower: FragmentMatch,
        restCasePreserved: FragmentMatch,
        raw: String
    ) -> ParsedReminderCommand? {
        // Try to extract "<n> min/hour[s] [before|after] <event title>".
        let pattern = #"(\d+)\s*(minutes?|mins?|hours?|hrs?)\s+(before|after)\s+(.+)$"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) else {
            return nil
        }
        let s = restLower.lowercased
        let sCase = restCasePreserved.casePreserved
        let range = NSRange(s.startIndex..., in: s)
        if let match = regex.firstMatch(in: s, options: [], range: range),
           let numberRange = Range(match.range(at: 1), in: s),
           let unitRange = Range(match.range(at: 2), in: s),
           let beforeRange = Range(match.range(at: 3), in: s),
           let titleRange = Range(match.range(at: 4), in: s) {
            let n = Int(s[numberRange]) ?? 0
            let unit = String(s[unitRange]).lowercased()
            let direction = String(s[beforeRange]).lowercased()
            // The case-preserved title sits at the same
            // Character offset in the case-preserved
            // string. For chat input this is all ASCII.
            let lowerStartOffset = s.distance(
                from: s.startIndex, to: titleRange.lowerBound
            )
            let lowerEndOffset = s.distance(
                from: s.startIndex, to: titleRange.upperBound
            )
            let titleStart = sCase.index(
                sCase.startIndex, offsetBy: lowerStartOffset
            )
            let titleEnd = sCase.index(
                sCase.startIndex, offsetBy: lowerEndOffset
            )
            let title = String(sCase[titleStart..<titleEnd])
                .trimmingCharacters(in: .whitespaces)
            // Strip a leading "the" so "remind me 15 min
            // before the Q3 review" yields "Q3 review" rather
            // than "the Q3 review".
            let titleNoThe: String
            if title.lowercased().hasPrefix("the ") {
                titleNoThe = String(title.dropFirst(4))
            } else {
                titleNoThe = title
            }
            let minutes = unit.hasPrefix("h") ? n * 60 : n
            let signed = direction == "before" ? -minutes : minutes
            return ParsedReminderCommand(
                kind: .create,
                offsetMinutes: signed,
                eventTitleFragment: stripReminderSuffix(from: titleNoThe),
                rawInput: raw
            )
        }

        // Fallback: "remind me about <event title>" with no offset.
        if let title = stripPrefix(
            ["about", "for"],
            inLower: restLower.lowercased,
            casePreserved: restCasePreserved.casePreserved
        ) {
            let cleaned = stripReminderSuffix(from: title.casePreserved)
            if !cleaned.isEmpty {
                return ParsedReminderCommand(
                    kind: .create,
                    offsetMinutes: -15, // 15 min before is the spec default
                    eventTitleFragment: cleaned,
                    rawInput: raw
                )
            }
        }
        return nil
    }
}
