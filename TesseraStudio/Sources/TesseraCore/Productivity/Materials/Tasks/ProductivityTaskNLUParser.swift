import Foundation

// MARK: - ParsedProductivityTask

/// The output of ``ProductivityTaskNLUParser/parse(_:)``. Holds the
/// extracted title, due date, priority, and linked entity ids.
/// The caller (the ProductivityTasks surface or the chat panel) takes a
/// ``ParsedProductivityTask`` and calls ``ProductivityTaskStore/upsert(_:actor:)`` to
/// persist it.
public struct ParsedProductivityTask: Codable, Sendable, Hashable {
    public var title: String
    public var dueAt: Date?
    public var priority: ProductivityTask.Priority
    public var linkedEntityIDs: [UUID]
    public var list: ProductivityTask.List

    public init(
        title: String,
        dueAt: Date? = nil,
        priority: ProductivityTask.Priority = .none,
        linkedEntityIDs: [UUID] = [],
        list: ProductivityTask.List = .inbox
    ) {
        self.title = title
        self.dueAt = dueAt
        self.priority = priority
        self.linkedEntityIDs = linkedEntityIDs
        self.list = list
    }

    /// The fully-built ``Task`` value the parser would
    /// persist. The caller can override any field after
    /// parsing (e.g. the user assigns a different list in
    /// the triage UI).
    public func toTask(now: Date = Date()) -> ProductivityTask {
        let chosenList: ProductivityTask.List = {
            if list != .inbox { return list }
            guard let dueAt else { return .anytime }
            let now24 = now.addingTimeInterval(24 * 60 * 60)
            let now7 = now.addingTimeInterval(7 * 24 * 60 * 60)
            if dueAt <= now24 { return .today }
            if dueAt <= now7 { return .upcoming }
            return .anytime
        }()
        return ProductivityTask(
            title: title,
            dueAt: dueAt,
            priority: priority,
            list: chosenList,
            linkedEntityIDs: linkedEntityIDs
        )
    }
}

// MARK: - ProductivityTaskNLUParser

/// A rule-based natural-language parser for the "Things 3-style"
/// input box. Given a free-form string like
/// `"tomorrow at 3pm, call John about the contract"`, the
/// parser extracts:
///
///   * the title (the verb-led phrase),
///   * a due date (the "tomorrow at 3pm" prefix or suffix),
///   * a priority (the "high priority:" prefix),
///   * linked entity ids (the names that match a contact or
///     document the user has on file).
///
/// v1 is rule-based; the patterns are well-known and an LLM
/// call would be overkill. The architecture supports an
/// LLM-based enhancement later (the parser is a struct, the
/// injection point is `init(contacts:documents:)`), but the
/// v1 path is deterministic and fast.
///
/// **Patterns recognised:**
///
/// | Input fragment | Effect |
/// |---|---|
/// | `high priority: ...` | priority = .high |
/// | `medium priority: ...` | priority = .medium |
/// | `low priority: ...` | priority = .low |
/// | `!` suffix | priority = .high (Things 3 convention) |
/// | `today`, `tonight`, `tomorrow`, `tomorrow at 3pm` | dueAt = next instance |
/// | `at 3pm`, `at 14:30`, `at noon` | time on the resolved date |
/// | `next monday`, `next tuesday`, ... | dueAt = next weekday instance |
/// | `in 3 days`, `in 2 weeks` | dueAt = now + offset |
/// | `on Jan 15`, `on January 15` | dueAt = parsed absolute date |
/// | bare noun (matches a contact's display name) | linkedEntityIDs += contact.id |
/// | bare noun (matches a document's title) | linkedEntityIDs += document.id |
///
/// The parser is **lenient**: ambiguous input falls back to
/// Anytime / no due date / normal priority. The user can
/// always edit the parsed values in the triage UI.
public struct ProductivityTaskNLUParser: Sendable {

    /// Contact lookup. The parser passes the candidate
    /// noun through here to find a matching `Contact.id`.
    public let contacts: ContactsAdapter?

    /// Document lookup. The parser passes the candidate
    /// noun through here to find a matching document's id.
    public let documents: DocumentStoreNLU?

    /// A "now" provider. Injectable for tests; defaults to
    /// `Date()`.
    public var now: () -> Date

    public init(
        contacts: ContactsAdapter? = nil,
        documents: DocumentStoreNLU? = nil,
        now: @escaping () -> Date = { Date() }
    ) {
        self.contacts = contacts
        self.documents = documents
        self.now = now
    }

    /// Parse a free-form input string. Always returns a
    /// ``ParsedProductivityTask``; the parser never throws (the user's
    /// input is unbounded; a partial parse is the success
    /// path).
    public func parse(_ input: String) -> ParsedProductivityTask {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        var working = trimmed
        var priority: ProductivityTask.Priority = .none
        var dueAt: Date?
        var linkedEntityIDs: [UUID] = []
        var list: ProductivityTask.List = .inbox

        // 1. Strip a leading priority prefix.
        if let (stripped, newPriority) = stripPriorityPrefix(working) {
            working = stripped
            priority = newPriority
        }

        // 2. Strip a trailing `!` (Things 3 convention: a
        // single `!` = high priority).
        if working.hasSuffix("!") {
            working = String(working.dropLast()).trimmingCharacters(in: .whitespaces)
            priority = .high
        }

        // 3. Parse a leading date phrase ("tomorrow at 3pm,
        // call John about the contract" -> strip the
        // "tomorrow at 3pm" prefix, parse a date).
        if let (stripped, parsedDate) = stripLeadingDate(working) {
            working = stripped
            dueAt = parsedDate
        }

        // 4. Parse a trailing date phrase ("call John
        // tomorrow" -> strip the trailing "tomorrow" suffix,
        // parse a date).
        if dueAt == nil, let (stripped, parsedDate) = stripTrailingDate(working) {
            working = stripped
            dueAt = parsedDate
        }

        // 5. If the working string now starts with a comma
        // (from a leading date parse), strip it.
        if working.hasPrefix(",") {
            working = String(working.dropFirst()).trimmingCharacters(in: .whitespaces)
        }

        // 6. Find contact / document links. The parser
        // looks at each "name-like" token in the working
        // string and queries the contacts / documents
        // adapters. The adapters are synchronous wrappers
        // around the data layer; in practice they're
        // in-memory caches the chat panel maintains.
        linkedEntityIDs = findLinkedEntities(in: working)

        // 7. Build the title. The title is the remaining
        // working string with leading/trailing punctuation
        // stripped. Empty titles fall back to the original
        // input.
        let title = working
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: ",;."))

        // 8. Pick a list. A leading "today" / "tonight" /
        // "tomorrow" date goes to .today. A future date in
        // 7 days goes to .upcoming. A bare "someday" goes
        // to .someday. Otherwise .anytime.
        list = inferredList(from: dueAt, rawInput: trimmed)

        return ParsedProductivityTask(
            title: title,
            dueAt: dueAt,
            priority: priority,
            linkedEntityIDs: linkedEntityIDs,
            list: list
        )
    }

    // MARK: - Priority prefix

    /// Recognise `high priority:`, `medium priority:`,
    /// `low priority:`, and `urgent:` prefixes. Returns the
    /// stripped string + the new priority.
    private func stripPriorityPrefix(_ input: String) -> (String, ProductivityTask.Priority)? {
        let lower = input.lowercased()
        let prefixes: [(String, ProductivityTask.Priority)] = [
            ("high priority:", .high),
            ("urgent:", .high),
            ("medium priority:", .medium),
            ("med priority:", .medium),
            ("low priority:", .low),
        ]
        for prefix in prefixes {
            if lower.hasPrefix(prefix.0) {
                let stripped = String(input.dropFirst(prefix.0.count))
                    .trimmingCharacters(in: .whitespaces)
                return (stripped, prefix.1)
            }
        }
        return nil
    }

    // MARK: - Leading date

    /// Strip a leading date phrase. The phrase is one of:
    ///   * `today` / `tonight`
    ///   * `tomorrow` (with optional `at 3pm`)
    ///   * `next monday` / `next tuesday` / ...
    ///   * `in N days` / `in N weeks`
    ///   * `on Jan 15` / `on January 15` / `on 1/15`
    ///
    /// Returns the stripped string + the parsed date.
    private func stripLeadingDate(_ input: String) -> (String, Date)? {
        // "tomorrow at 3pm, ..." or "tomorrow, ..."
        for keyword in ["tomorrow", "today", "tonight"] {
            if let date = matchRelativeKeyword(keyword, in: input, atStart: true) {
                return date
            }
        }

        if let match = matchNextWeekday(input) { return match }
        if let match = matchInDays(input) { return match }
        if let match = matchOnDate(input) { return match }
        return nil
    }

    /// Strip a trailing date phrase. The phrase is one of:
    ///   * `... tomorrow` / `... today` / `... tonight`
    private func stripTrailingDate(_ input: String) -> (String, Date)? {
        for keyword in ["tomorrow", "today", "tonight"] {
            if input.lowercased().hasSuffix(keyword) {
                let stripped = String(input.dropLast(keyword.count))
                    .trimmingCharacters(in: .whitespaces)
                let date = relativeDate(keyword: keyword, timeOffset: nil, now: now())
                return (stripped, date)
            }
        }
        return nil
    }

    /// Match a relative keyword at the start (or, with
    /// `atStart = false`, anywhere) of the input. Returns
    /// the trailing string + the parsed date.
    private func matchRelativeKeyword(
        _ keyword: String,
        in input: String,
        atStart: Bool
    ) -> (String, Date)? {
        let lower = input.lowercased()
        guard let range = lower.range(of: keyword) else { return nil }
        if atStart && range.lowerBound != lower.startIndex {
            // Allow the keyword only when nothing (or only
            // whitespace) precedes it.
            let prefix = String(lower[lower.startIndex..<range.lowerBound])
                .trimmingCharacters(in: .whitespaces)
            if !prefix.isEmpty { return nil }
        }
        let afterKeyword = String(input[range.upperBound...])
        let (timeSuffix, timeConsumed) = parseTimePrefix(afterKeyword)
        let date = relativeDate(keyword: keyword, timeOffset: timeSuffix, now: now())
        // Strip the keyword + the time suffix from the input.
        var consumedEnd = range.upperBound
        if timeConsumed > 0 {
            // The `at 3pm` form: the consumed text is
            // `timeConsumed` characters past the keyword.
            // `parseTimePrefix` returns the consumed count
            // in `afterKeyword`'s character index space.
            let afterK = input[range.upperBound...]
            if timeConsumed <= afterK.count {
                let kIdx = afterK.index(afterK.startIndex, offsetBy: timeConsumed)
                consumedEnd = kIdx
            }
        }
        // Trim any trailing whitespace + an optional
        // comma that bridges the date phrase to the title.
        var rest = String(input[consumedEnd...])
        while rest.hasPrefix(" ") || rest.hasPrefix(",") {
            rest = String(rest.dropFirst())
        }
        rest = rest.trimmingCharacters(in: .whitespaces)
        return (rest, date)
    }

    /// Parse an optional time prefix of the form
    /// `at 3pm` / `at 3:30pm` / `at 15:00` / `at noon` /
    /// `at midnight`. Returns `(hour, minute)` and the
    /// number of characters consumed in the ORIGINAL
    /// string (so the caller can advance `consumedEnd` by
    /// that many characters).
    private func parseTimePrefix(_ s: String) -> ((Int, Int)?, Int) {
        // Count leading whitespace so the caller can skip
        // past it.
        var leadingWS = 0
        for c in s {
            if c == " " { leadingWS += 1 } else { break }
        }
        let trimmed = String(s.dropFirst(leadingWS))
        let lower = trimmed.lowercased()
        guard lower.hasPrefix("at ") else { return (nil, 0) }
        let after = String(trimmed.dropFirst(3))
        // "noon"
        if after.lowercased().hasPrefix("noon") {
            return ((12, 0), leadingWS + 3 + 4)
        }
        if after.lowercased().hasPrefix("midnight") {
            return ((0, 0), leadingWS + 3 + 8)
        }
        if let parsed = parseClockTime(after) {
            return (parsed.time, leadingWS + 3 + parsed.consumed)
        }
        return (nil, 0)
    }

    /// Parse a clock-time string. Returns `(hour, minute)`
    /// and the number of characters consumed.
    private func parseClockTime(_ s: String) -> (time: (Int, Int), consumed: Int)? {
        let lower = s.lowercased()
        // "3pm" / "3 pm" / "12pm" / "12 am"
        if let (hourText, afterHour) = takeInt(lower) {
            var pos = 0
            // optional whitespace between hour and am/pm
            while pos < afterHour.count, afterHour[afterHour.index(afterHour.startIndex, offsetBy: pos)] == " " {
                pos += 1
            }
            let h = Int(hourText) ?? 0
            // The hour digit is part of the consumed text.
            // "3" is 1 char; "12" is 2 chars.
            let hourConsumed = hourText.count
            // Check for "am" / "pm" suffix.
            if pos + 1 < afterHour.count {
                let suffix = String(afterHour[afterHour.index(afterHour.startIndex, offsetBy: pos)...])
                if suffix.hasPrefix("am") {
                    if h == 12 { return ((0, 0), hourConsumed + pos + 2) }
                    return ((h, 0), hourConsumed + pos + 2)
                } else if suffix.hasPrefix("pm") {
                    if h == 12 { return ((12, 0), hourConsumed + pos + 2) }
                    return ((h + 12, 0), hourConsumed + pos + 2)
                }
            }
            // "3:30" / "14:30"
            if pos < afterHour.count, afterHour[afterHour.index(afterHour.startIndex, offsetBy: pos)] == ":" {
                let afterColon = String(afterHour[afterHour.index(afterHour.startIndex, offsetBy: pos + 1)...])
                if let (minuteText, afterMin) = takeInt(afterColon) {
                    var consumed = hourConsumed + pos + 1 + minuteText.count
                    let m = Int(minuteText) ?? 0
                    // optional am/pm
                    var scanIdx = 0
                    while scanIdx < afterMin.count, afterMin[afterMin.index(afterMin.startIndex, offsetBy: scanIdx)] == " " {
                        scanIdx += 1
                    }
                    let tail = String(afterMin[afterMin.index(afterMin.startIndex, offsetBy: scanIdx)...])
                    if tail.hasPrefix("pm") {
                        if h == 12 { return ((12, m), consumed + scanIdx + 2) }
                        return ((h + 12, m), consumed + scanIdx + 2)
                    } else if tail.hasPrefix("am") {
                        if h == 12 { return ((0, m), consumed + scanIdx + 2) }
                        return ((h, m), consumed + scanIdx + 2)
                    }
                    return ((h, m), consumed)
                }
            }
            // Just an hour, no suffix. e.g. "14"
            if h >= 0, h < 24 {
                return ((h, 0), hourConsumed)
            }
        }
        return nil
    }

    /// Read an integer at the start of `s`. Returns the
    /// digit text + the rest of the string after the digits.
    private func takeInt(_ s: String) -> (text: String, rest: String)? {
        var chars: [Character] = []
        for c in s {
            if c.isNumber { chars.append(c) } else { break }
        }
        guard !chars.isEmpty else { return nil }
        let text = String(chars)
        let rest = String(s.dropFirst(chars.count))
        return (text, rest)
    }

    /// Resolve a relative-date keyword to an absolute date
    /// for the given `now`.
    private func relativeDate(
        keyword: String,
        timeOffset: (Int, Int)?,
        now: Date
    ) -> Date {
        let calendar = Calendar.current
        let lowered = keyword.lowercased()
        let baseDay: Date
        switch lowered {
        case "today", "tonight":
            baseDay = calendar.startOfDay(for: now)
        case "tomorrow":
            baseDay = calendar.date(byAdding: .day, value: 1, to: calendar.startOfDay(for: now)) ?? now
        default:
            baseDay = now
        }
        guard let (h, m) = timeOffset else {
            if lowered == "tonight" {
                return calendar.date(bySettingHour: 20, minute: 0, second: 0, of: baseDay) ?? baseDay
            }
            return calendar.date(bySettingHour: 9, minute: 0, second: 0, of: baseDay) ?? baseDay
        }
        return calendar.date(bySettingHour: h, minute: m, second: 0, of: baseDay) ?? baseDay
    }

    /// Match "next monday at 3pm, ..." (case-insensitive).
    private func matchNextWeekday(_ input: String) -> (rest: String, date: Date)? {
        let lower = input.lowercased()
        guard lower.hasPrefix("next ") else { return nil }
        let after = String(lower.dropFirst(5))
        let weekdays: [(String, Int)] = [
            ("sunday", 1), ("monday", 2), ("tuesday", 3),
            ("wednesday", 4), ("thursday", 5), ("friday", 6), ("saturday", 7),
        ]
        for (name, weekday) in weekdays {
            if after.hasPrefix(name) {
                let now = now()
                let calendar = Calendar.current
                let today = calendar.component(.weekday, from: now)
                var daysToAdd = weekday - today
                if daysToAdd <= 0 { daysToAdd += 7 }
                let baseDay = calendar.date(
                    byAdding: .day, value: daysToAdd, to: calendar.startOfDay(for: now)
                ) ?? now
                let afterName = String(input.dropFirst(5 + name.count))
                let (timeSuffix, _) = parseTimePrefix(afterName)
                let date: Date
                if let (h, m) = timeSuffix {
                    date = calendar.date(bySettingHour: h, minute: m, second: 0, of: baseDay) ?? baseDay
                } else {
                    date = calendar.date(bySettingHour: 9, minute: 0, second: 0, of: baseDay) ?? baseDay
                }
                // Strip "next <weekday> [at <time>]" from input.
                let consumed = 5 + name.count
                var rest = String(input.dropFirst(consumed)).trimmingCharacters(in: .whitespaces)
                // Re-scan to drop the optional time suffix.
                let (timeSuffixAgain, consumedTime) = parseTimePrefix(rest)
                if timeSuffixAgain != nil, consumedTime > 0 {
                    rest = String(rest.dropFirst(consumedTime))
                        .trimmingCharacters(in: .whitespaces)
                }
                if rest.hasPrefix(",") {
                    rest = String(rest.dropFirst()).trimmingCharacters(in: .whitespaces)
                }
                return (rest, date)
            }
        }
        return nil
    }

    /// Match "in N days" / "in N weeks" / "in a day" /
    /// "in a week".
    private func matchInDays(_ input: String) -> (rest: String, date: Date)? {
        let lower = input.lowercased()
        guard lower.hasPrefix("in ") else { return nil }
        let after = String(lower.dropFirst(3))
        if after.hasPrefix("a day") {
            let date = Calendar.current.date(byAdding: .day, value: 1, to: now()) ?? now()
            return (String(input.dropFirst(3 + 5)).trimmingCharacters(in: .whitespaces), date)
        }
        if after.hasPrefix("a week") {
            let date = Calendar.current.date(byAdding: .day, value: 7, to: now()) ?? now()
            return (String(input.dropFirst(3 + 6)).trimmingCharacters(in: .whitespaces), date)
        }
        if let (numText, afterNum) = takeInt(after), let num = Int(numText) {
            let pos = numText.count
            let unit = afterNum
            var daysToAdd = 0
            var consumedUnit = 0
            // `unit` starts with a space (" days, ..."), so
            // the keyword is at offset 1.
            let unitTrimmed = unit.drop(while: { $0 == " " })
            if unitTrimmed.hasPrefix("day") {
                daysToAdd = num
                consumedUnit = 1 + 4
            } else if unitTrimmed.hasPrefix("week") {
                daysToAdd = num * 7
                consumedUnit = 1 + 5
            } else if unitTrimmed.hasPrefix("month") {
                daysToAdd = num * 30
                consumedUnit = 1 + 5
            }
            if daysToAdd > 0 {
                let date = Calendar.current.date(byAdding: .day, value: daysToAdd, to: now()) ?? now()
                let consumed = 3 + pos + consumedUnit
                var rest = String(input.dropFirst(consumed)).trimmingCharacters(in: .whitespaces)
                // Strip a leading comma that bridges the
                // date phrase to the title.
                if rest.hasPrefix(",") {
                    rest = String(rest.dropFirst()).trimmingCharacters(in: .whitespaces)
                }
                return (rest, date)
            }
        }
        return nil
    }

    /// Match "on Jan 15" / "on January 15" / "on 1/15".
    private func matchOnDate(_ input: String) -> (rest: String, date: Date)? {
        let lower = input.lowercased()
        guard lower.hasPrefix("on ") else { return nil }
        let after = String(lower.dropFirst(3))
        let monthNames: [(String, Int)] = [
            ("january", 1), ("february", 2), ("march", 3),
            ("april", 4), ("may", 5), ("june", 6),
            ("july", 7), ("august", 8), ("september", 9),
            ("october", 10), ("november", 11), ("december", 12),
            ("jan", 1), ("feb", 2), ("mar", 3), ("apr", 4),
            ("jun", 6), ("jul", 7), ("aug", 8), ("sep", 9),
            ("sept", 9), ("oct", 10), ("nov", 11), ("dec", 12),
        ]
        for (name, month) in monthNames {
            if after.hasPrefix(name) {
                let afterName = String(after.dropFirst(name.count)).trimmingCharacters(in: .whitespaces)
                if let (dayText, _) = takeInt(afterName), let day = Int(dayText) {
                    let year = Calendar.current.component(.year, from: now())
                    let date = Calendar.current.date(
                        from: DateComponents(year: year, month: month, day: day, hour: 9)
                    ) ?? now()
                    let consumed = 3 + name.count + dayText.count
                    var rest = String(input.dropFirst(consumed)).trimmingCharacters(in: .whitespaces)
                    if rest.hasPrefix(",") {
                        rest = String(rest.dropFirst()).trimmingCharacters(in: .whitespaces)
                    }
                    return (rest, date)
                }
            }
        }
        if let (firstText, afterFirst) = takeInt(after),
           afterFirst.hasPrefix("/") {
            let afterSlash = String(afterFirst.dropFirst(1))
            if let (secondText, afterSecond) = takeInt(afterSlash) {
                let month = Int(firstText) ?? 1
                let day = Int(secondText) ?? 1
                var year = Calendar.current.component(.year, from: now())
                var consumed = 3 + firstText.count + 1 + secondText.count
                if afterSecond.hasPrefix("/") {
                    let afterThird = String(afterSecond.dropFirst(1))
                    if let (yearText, _) = takeInt(afterThird),
                       let y = Int(yearText), y > 1900, y < 3000 {
                        year = y
                        consumed += 1 + yearText.count
                    }
                }
                let date = Calendar.current.date(
                    from: DateComponents(year: year, month: month, day: day, hour: 9)
                ) ?? now()
                var rest = String(input.dropFirst(consumed)).trimmingCharacters(in: .whitespaces)
                if rest.hasPrefix(",") {
                    rest = String(rest.dropFirst()).trimmingCharacters(in: .whitespaces)
                }
                return (rest, date)
            }
        }
        return nil
    }

    // MARK: - Linked entity discovery

    /// Find contact / document links in the working string.
    /// The candidates are extracted first, then the
    /// contacts / documents adapters do the actual matching
    /// (which is where the multi-word problem is solved).
    private func findLinkedEntities(in text: String) -> [UUID] {
        var ids: [UUID] = []
        let candidates = extractNameCandidates(from: text)
        // First try contacts (more common for tasks).
        if let contact = contacts?.find(matchingAny: candidates) {
            if !ids.contains(contact.id) { ids.append(contact.id) }
        }
        // Then documents.
        if let document = documents?.findStub(matchingAny: candidates) {
            if !ids.contains(document.id) { ids.append(document.id) }
        }
        return ids
    }

    /// Extract "name-like" candidates. Quoted strings first
    /// (explicit references), then all words (the
    /// contact/document adapter is case-insensitive and does
    /// the multi-word matching).
    private func extractNameCandidates(from text: String) -> [String] {
        var out: [String] = []
        // Quoted strings
        var inString = false
        var stringDelimiter: Character = "\""
        var buffer = ""
        for c in text {
            if !inString, c == "\"" || c == "'" {
                inString = true
                stringDelimiter = c
                buffer = ""
            } else if inString, c == stringDelimiter {
                if !buffer.isEmpty { out.append(buffer) }
                inString = false
                buffer = ""
            } else if inString {
                buffer.append(c)
            }
        }
        // All words
        let separators = CharacterSet(charactersIn: " \t\n,.;:!?()[]{}\"")
        let words = text.components(separatedBy: separators)
        for word in words {
            guard let first = word.first else { continue }
            if first.isLetter, word.count >= 2 {
                out.append(word)
            }
        }
        return out
    }

    // MARK: - List inference

    /// Pick the list a task should land in. A leading
    /// "today" / "tonight" / "tomorrow" date goes to
    /// .today. A future date in 7 days goes to .upcoming. A
    /// bare "someday" in the input goes to .someday.
    /// Otherwise .anytime.
    private func inferredList(from dueAt: Date?, rawInput: String) -> ProductivityTask.List {
        let lower = rawInput.lowercased()
        if lower.contains("someday") { return .someday }
        if let dueAt {
            let now = now()
            if dueAt <= now.addingTimeInterval(24 * 60 * 60) { return .today }
            if dueAt <= now.addingTimeInterval(7 * 24 * 60 * 60) { return .upcoming }
        }
        return .anytime
    }
}

// MARK: - Synchronous adapter protocols

/// The synchronous contact lookup the NLU parser uses. The
/// chat panel integration wraps ``ContactStore`` in an actor
/// that maintains an in-memory cache; the parser reads from
/// the cache (which is fast enough for the per-keystroke NLU
/// path).
///
/// The adapter does the matching: it returns a contact
/// when ANY of the candidate words appears in the contact's
/// display name (case-insensitive). The adapter is
/// responsible for resolving the multi-word problem (e.g.,
/// the contact "John Doe" should match the candidate "John").
public protocol ContactsAdapter: Sendable {
    func find(matchingAny candidates: [String]) -> Contact?
}

/// The synchronous document lookup the NLU parser uses.
/// The chat panel integration wraps ``DocumentStore`` in an
/// actor that maintains an in-memory cache; the parser reads
/// from the cache.
///
/// The parser only needs the document's id and title (to
/// match against user input), so we expose a minimal stub
/// rather than the full document body.
public struct DocumentStub: Sendable, Hashable {
    public let id: UUID
    public let title: String

    public init(id: UUID, title: String) {
        self.id = id
        self.title = title
    }
}

public protocol DocumentStoreNLU: Sendable {
    /// Returns a document stub when any of the candidate
    /// words appears in the document's title
    /// (case-insensitive). Same matching model as
    /// ``ContactsAdapter``.
    func findStub(matchingAny candidates: [String]) -> DocumentStub?
}
