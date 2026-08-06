import Foundation

// MARK: - RecurrenceRule

/// A small RFC 5545 RRULE parser + occurrence expander.
///
/// **Why build instead of adopt:** there is no maintained
/// Swift RRULE library that isn't a heavyweight EventKit
/// wrapper, and the subset the calendar surface needs is
/// small (FREQ, INTERVAL, COUNT, UNTIL, BYDAY, BYMONTHDAY,
/// BYMONTH). The parser rejects anything outside the
/// supported subset rather than half-interpreting it — an
/// unsupported rule throws, and ``CalendarEvent`` degrades
/// such an event to its single base occurrence instead of
/// silently dropping it.
///
/// The type is immutable and `Sendable`; occurrence
/// expansion is a pure function of (rule, anchor, range,
/// calendar).
public struct RecurrenceRule: Sendable, Equatable, Hashable {

    public enum Frequency: String, Sendable, Equatable, Hashable, CaseIterable {
        case daily = "DAILY"
        case weekly = "WEEKLY"
        case monthly = "MONTHLY"
        case yearly = "YEARLY"
    }

    /// RFC 5545 weekday abbreviations.
    public enum Weekday: String, Sendable, Equatable, Hashable, CaseIterable {
        case monday = "MO"
        case tuesday = "TU"
        case wednesday = "WE"
        case thursday = "TH"
        case friday = "FR"
        case saturday = "SA"
        case sunday = "SU"

        /// Map to `Calendar`'s weekday numbering
        /// (Sunday == 1 ... Saturday == 7).
        public var calendarWeekday: Int {
            switch self {
            case .sunday: return 1
            case .monday: return 2
            case .tuesday: return 3
            case .wednesday: return 4
            case .thursday: return 5
            case .friday: return 6
            case .saturday: return 7
            }
        }

        /// Build from a `Calendar` weekday number.
        public static func from(calendarWeekday: Int) -> Weekday {
            switch calendarWeekday {
            case 1: return .sunday
            case 2: return .monday
            case 3: return .tuesday
            case 4: return .wednesday
            case 5: return .thursday
            case 6: return .friday
            default: return .saturday
            }
        }
    }

    public var frequency: Frequency
    /// Repeat step. `INTERVAL=2` with `FREQ=WEEKLY` is
    /// every other week. Defaults to 1 per RFC 5545.
    public var interval: Int
    /// Stop after this many occurrences (COUNT), or nil.
    public var count: Int?
    /// Stop at or before this instant (UNTIL), or nil.
    public var until: Date?
    /// Day-of-week filter (BYDAY). For WEEKLY rules this is
    /// the set of days of the week the event lands on; when
    /// empty a WEEKLY rule defaults to the anchor's weekday.
    public var byDay: [Weekday]
    /// Day-of-month filter (BYMONTHDAY) for MONTHLY rules.
    /// When empty a MONTHLY rule defaults to the anchor's
    /// day of month.
    public var byMonthDay: [Int]
    /// Month filter (BYMONTH) for YEARLY rules. When empty
    /// a YEARLY rule defaults to the anchor's month.
    public var byMonth: [Int]

    public init(
        frequency: Frequency,
        interval: Int = 1,
        count: Int? = nil,
        until: Date? = nil,
        byDay: [Weekday] = [],
        byMonthDay: [Int] = [],
        byMonth: [Int] = []
    ) {
        self.frequency = frequency
        self.interval = max(1, interval)
        self.count = count
        self.until = until
        self.byDay = byDay
        self.byMonthDay = byMonthDay
        self.byMonth = byMonth
    }

    // MARK: - Parsing

    public enum ParseError: Error, Sendable, Equatable {
        case empty
        case missingFrequency
        case unknownFrequency(String)
        case unknownPart(String)
        case invalidValue(part: String, value: String)
        case unsupportedOrdinal(part: String, value: String)
    }

    /// Parse an RRULE string ("FREQ=WEEKLY;BYDAY=MO,WE").
    /// Throws ``ParseError`` for anything outside the
    /// supported subset.
    public init(rrule: String) throws {
        let trimmed = rrule.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw ParseError.empty }

        var frequency: Frequency?
        var interval = 1
        var count: Int?
        var until: Date?
        var byDay: [Weekday] = []
        var byMonthDay: [Int] = []
        var byMonth: [Int] = []

        let body = trimmed.hasPrefix("RRULE:") ? String(trimmed.dropFirst("RRULE:".count)) : trimmed
        for rawPart in body.split(separator: ";", omittingEmptySubsequences: true) {
            let part = String(rawPart).trimmingCharacters(in: .whitespaces)
            guard let eq = part.firstIndex(of: "=") else {
                throw ParseError.unknownPart(part)
            }
            let key = String(part[..<eq]).uppercased()
            let value = String(part[part.index(after: eq)...]).trimmingCharacters(in: .whitespaces)

            switch key {
            case "FREQ":
                guard let freq = Frequency(rawValue: value.uppercased()) else {
                    throw ParseError.unknownFrequency(value)
                }
                frequency = freq
            case "INTERVAL":
                guard let v = Int(value), v >= 1 else {
                    throw ParseError.invalidValue(part: key, value: value)
                }
                interval = v
            case "COUNT":
                guard let v = Int(value), v >= 1 else {
                    throw ParseError.invalidValue(part: key, value: value)
                }
                count = v
            case "UNTIL":
                guard let d = Self.parseUntil(value) else {
                    throw ParseError.invalidValue(part: key, value: value)
                }
                until = d
            case "BYDAY":
                for token in value.split(separator: ",") {
                    let t = token.trimmingCharacters(in: .whitespaces).uppercased()
                    // Ordinal weekdays ("1MO", "-1FR") are out
                    // of scope for v1; reject loudly.
                    guard let wd = Weekday(rawValue: t) else {
                        if Weekday(rawValue: String(t.suffix(2))) != nil {
                            throw ParseError.unsupportedOrdinal(part: key, value: t)
                        }
                        throw ParseError.invalidValue(part: key, value: t)
                    }
                    byDay.append(wd)
                }
                guard !byDay.isEmpty else {
                    throw ParseError.invalidValue(part: key, value: value)
                }
            case "BYMONTHDAY":
                for token in value.split(separator: ",") {
                    let t = token.trimmingCharacters(in: .whitespaces)
                    guard let d = Int(t), (1...31).contains(abs(d)), d > 0 else {
                        // Negative monthdays ("-1") are out of
                        // scope for v1; reject loudly.
                        throw ParseError.invalidValue(part: key, value: t)
                    }
                    byMonthDay.append(d)
                }
            case "BYMONTH":
                for token in value.split(separator: ",") {
                    let t = token.trimmingCharacters(in: .whitespaces)
                    guard let m = Int(t), (1...12).contains(m) else {
                        throw ParseError.invalidValue(part: key, value: t)
                    }
                    byMonth.append(m)
                }
            case "WKST":
                // Accepted and ignored: the expander walks the
                // calendar day-by-day, so the week start only
                // matters for week-number math we don't do.
                continue
            default:
                throw ParseError.unknownPart(part)
            }
        }

        guard let frequency else { throw ParseError.missingFrequency }
        self.init(
            frequency: frequency,
            interval: interval,
            count: count,
            until: until,
            byDay: byDay,
            byMonthDay: byMonthDay,
            byMonth: byMonth
        )
    }

    /// UNTIL accepts the RFC 5545 forms `20260101` and
    /// `20260101T090000Z` (the trailing Z is treated as UTC;
    /// a missing Z is treated as local time — RFC 5545 calls
    /// that form "floating", local is the closest honest
    /// interpretation).
    private static func parseUntil(_ value: String) -> Date? {
        let v = value.uppercased().trimmingCharacters(in: .whitespaces)
        let utcFormatter = DateFormatter()
        utcFormatter.locale = Locale(identifier: "en_US_POSIX")
        utcFormatter.timeZone = TimeZone(identifier: "UTC")
        let localFormatter = DateFormatter()
        localFormatter.locale = Locale(identifier: "en_US_POSIX")

        if v.hasSuffix("Z") {
            utcFormatter.dateFormat = "yyyyMMdd'T'HHmmss'Z'"
            if let d = utcFormatter.date(from: v) { return d }
        }
        if v.contains("T") {
            localFormatter.dateFormat = "yyyyMMdd'T'HHmmss"
            if let d = localFormatter.date(from: v) { return d }
        }
        localFormatter.dateFormat = "yyyyMMdd"
        if let d = localFormatter.date(from: v) { return d }
        // Be lenient about ISO-8601 too (the NLU path
        // serializes dates that way).
        let iso = ISO8601DateFormatter()
        return iso.date(from: value)
    }

    // MARK: - Serialization

    /// Serialize back to the RRULE string form. Order is
    /// fixed (FREQ first) so round-trips are stable.
    public var rruleString: String {
        var parts = ["FREQ=\(frequency.rawValue)"]
        if interval > 1 { parts.append("INTERVAL=\(interval)") }
        if let count { parts.append("COUNT=\(count)") }
        if let until {
            let f = DateFormatter()
            f.locale = Locale(identifier: "en_US_POSIX")
            f.timeZone = TimeZone(identifier: "UTC")
            f.dateFormat = "yyyyMMdd'T'HHmmss'Z'"
            parts.append("UNTIL=\(f.string(from: until))")
        }
        if !byDay.isEmpty { parts.append("BYDAY=\(byDay.map(\.rawValue).joined(separator: ","))") }
        if !byMonthDay.isEmpty { parts.append("BYMONTHDAY=\(byMonthDay.map(String.init).joined(separator: ","))") }
        if !byMonth.isEmpty { parts.append("BYMONTH=\(byMonth.map(String.init).joined(separator: ","))") }
        return parts.joined(separator: ";")
    }

    // MARK: - Occurrence expansion

    /// Expansion safety bound. A malicious / mistaken
    /// `FREQ=DAILY` rule with no COUNT / UNTIL would
    /// otherwise expand forever across a wide range; the
    /// cap bounds the work while covering any realistic
    /// view window.
    public static let maxOccurrences = 1000

    /// The occurrence start instants in `[range.lowerBound,
    /// range.upperBound]`, derived from `anchor` (the event's
    /// first occurrence). Pure and calendar-aware: weekday /
    /// month math goes through `calendar` so the user's
    /// first-weekday setting is respected.
    public func occurrences(anchor: Date, in range: ClosedRange<Date>, calendar: Calendar) -> [Date] {
        var out: [Date] = []
        var emitted = 0
        var visited = 0
        // Walk the candidate space from the anchor forward.
        // The anchor's time-of-day is preserved on every
        // occurrence.
        let comps = calendar.dateComponents([.hour, .minute, .second], from: anchor)
        let anchorDay = calendar.startOfDay(for: anchor)
        var cursor = anchorDay
        let maxVisits = 5000

        while emitted < Self.maxOccurrences && visited < maxVisits {
            visited += 1
            // Stop when the cursor passes the range (the
            // occurrence's time-of-day can only push it
            // later in the same day).
            if let occurrence = occurrenceInstant(day: cursor, time: comps, calendar: calendar),
               occurrence > range.upperBound {
                break
            }

            if matches(day: cursor, anchorDay: anchorDay, calendar: calendar) {
                if let occurrence = occurrenceInstant(day: cursor, time: comps, calendar: calendar) {
                    let withinCount = count.map { emitted < $0 } ?? true
                    let withinUntil = until.map { occurrence <= $0 } ?? true
                    if !withinCount || !withinUntil { break }
                    emitted += 1
                    if occurrence >= range.lowerBound && occurrence <= range.upperBound {
                        out.append(occurrence)
                    }
                }
            }

            guard let next = nextCandidate(after: cursor, anchorDay: anchorDay, calendar: calendar) else {
                break
            }
            cursor = next
        }
        return out
    }

    private func occurrenceInstant(
        day: Date,
        time: DateComponents,
        calendar: Calendar
    ) -> Date? {
        var c = calendar.dateComponents([.year, .month, .day], from: day)
        c.hour = time.hour ?? 0
        c.minute = time.minute ?? 0
        c.second = time.second ?? 0
        return calendar.date(from: c)
    }

    /// True when `day` satisfies the rule's filters,
    /// relative to `anchorDay` (the interval math is
    /// anchored to the first occurrence).
    private func matches(day: Date, anchorDay: Date, calendar: Calendar) -> Bool {
        let weekday = Weekday.from(calendarWeekday: calendar.component(.weekday, from: day))
        let dom = calendar.component(.day, from: day)
        let month = calendar.component(.month, from: day)

        switch frequency {
        case .daily:
            let days = calendar.dateComponents([.day], from: anchorDay, to: day).day ?? 0
            return days % interval == 0
        case .weekly:
            // Weeks are counted between calendar week starts
            // (respecting the user's first-weekday setting),
            // so INTERVAL aligns with real weeks even when
            // BYDAY lists a weekday before the anchor's.
            let weekIndex = weeksBetween(startOfWeek(anchorDay, calendar: calendar), startOfWeek(day, calendar: calendar), calendar: calendar)
            if weekIndex < 0 || weekIndex % interval != 0 { return false }
            let daysInWeek = byDay.isEmpty
                ? [Weekday.from(calendarWeekday: calendar.component(.weekday, from: anchorDay))]
                : byDay
            return daysInWeek.contains(weekday)
        case .monthly:
            let m1 = calendar.component(.year, from: anchorDay) * 12 + calendar.component(.month, from: anchorDay)
            let m2 = calendar.component(.year, from: day) * 12 + calendar.component(.month, from: day)
            guard m2 >= m1, (m2 - m1) % interval == 0 else { return false }
            let daysInMonth = byMonthDay.isEmpty ? [calendar.component(.day, from: anchorDay)] : byMonthDay
            return daysInMonth.contains(dom)
        case .yearly:
            let y1 = calendar.component(.year, from: anchorDay)
            let y2 = calendar.component(.year, from: day)
            guard y2 >= y1, (y2 - y1) % interval == 0 else { return false }
            let months = byMonth.isEmpty ? [calendar.component(.month, from: anchorDay)] : byMonth
            guard months.contains(month) else { return false }
            let daysInMonth = byMonthDay.isEmpty ? [calendar.component(.day, from: anchorDay)] : byMonthDay
            return daysInMonth.contains(dom)
        }
    }

    /// Advance the cursor to the next day worth testing.
    /// DAILY / WEEKLY step one day at a time (the filter is
    /// cheap); MONTHLY jumps to the first of the next
    /// matching month; YEARLY jumps to Jan 1 of the next
    /// candidate year. The day-step keeps the BYDAY logic
    /// simple and correct across DST transitions (the
    /// calendar does the date arithmetic).
    private func nextCandidate(after day: Date, anchorDay: Date, calendar: Calendar) -> Date? {
        switch frequency {
        case .daily, .weekly:
            return calendar.date(byAdding: .day, value: 1, to: day)
        case .monthly:
            let dom = calendar.component(.day, from: day)
            let maxDom = byMonthDay.isEmpty ? 31 : byMonthDay.max() ?? 31
            if dom < maxDom {
                return calendar.date(byAdding: .day, value: 1, to: day)
            }
            return calendar.date(byAdding: .month, value: interval, to: startOfMonth(day, calendar: calendar))
        case .yearly:
            let month = calendar.component(.month, from: day)
            let dom = calendar.component(.day, from: day)
            let maxMonth = byMonth.isEmpty ? 12 : byMonth.max() ?? 12
            if month < maxMonth || (month == maxMonth && dom < 31) {
                return calendar.date(byAdding: .day, value: 1, to: day)
            }
            var c = calendar.dateComponents([.year], from: day)
            c.month = 1
            c.day = 1
            guard let jan1 = calendar.date(from: c) else { return nil }
            return calendar.date(byAdding: .year, value: interval, to: jan1)
        }
    }

    private func startOfMonth(_ day: Date, calendar: Calendar) -> Date {
        var c = calendar.dateComponents([.year, .month], from: day)
        c.day = 1
        return calendar.date(from: c) ?? day
    }

    private func startOfWeek(_ day: Date, calendar: Calendar) -> Date {
        calendar.dateInterval(of: .weekOfYear, for: day)?.start ?? calendar.startOfDay(for: day)
    }

    private func weeksBetween(_ a: Date, _ b: Date, calendar: Calendar) -> Int {
        let days = calendar.dateComponents([.day], from: a, to: b).day ?? 0
        return days / 7
    }
}
