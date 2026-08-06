import Foundation

// MARK: - Resolver protocols

/// Synchronous contact lookup for the NLU parser. The real
/// implementation is a snapshot of the ``ContactStore``
/// loaded ahead of time (``ContactSnapshotAdapter``); tests
/// use ``StaticContactsAdapter``. Sync on purpose: the
/// parser is a pure rule engine and must stay testable
/// without async plumbing.
public protocol ContactsAdapter: Sendable {
    /// Contacts whose display name or first name matches
    /// `name` (case-insensitive prefix or substring).
    func contacts(matching name: String) -> [Contact]
}

/// Synchronous document lookup for the NLU parser ("...see
/// the \"Q3 roadmap\" doc before"). The real implementation
/// snapshots the document entity labels via the data layer.
public protocol DocumentResolver: Sendable {
    /// Documents whose title matches `title`
    /// (case-insensitive substring).
    func documents(matching title: String) -> [ResolvedDocument]
}

/// One document the parser can link to an event.
public struct ResolvedDocument: Sendable, Equatable, Hashable {
    public let id: UUID
    public let title: String

    public init(id: UUID, title: String) {
        self.id = id
        self.title = title
    }
}

/// Synchronous location geocoding for the NLU parser.
/// Geocoding itself is async + network-touching, so the
/// parser only consults a cache-backed resolver; the
/// ``GeocodingLocationResolver`` fills the cache from
/// `CLGeocoder` off the parse path.
public protocol LocationResolver: Sendable {
    /// The cached coordinate for `location`, or nil when the
    /// location was never geocoded. Never blocks, never
    /// throws.
    func coordinate(for location: String) -> CalendarEvent.Coordinate?
}

// MARK: - Static adapters (tests + offline use)

/// In-memory contacts adapter for tests and for the
/// snapshot the view model preloads.
public struct StaticContactsAdapter: ContactsAdapter {
    public var contacts: [Contact]

    public init(contacts: [Contact] = []) {
        self.contacts = contacts
    }

    public func contacts(matching name: String) -> [Contact] {
        let q = name.lowercased().trimmingCharacters(in: .whitespaces)
        guard !q.isEmpty else { return [] }
        return contacts.filter { c in
            let display = c.displayName.lowercased()
            if display == q || display.hasPrefix(q) || display.contains(q) { return true }
            let first = (c.name.first ?? "").lowercased()
            return first == q || first.hasPrefix(q)
        }
    }
}

/// In-memory document resolver for tests.
public struct StaticDocumentResolver: DocumentResolver {
    public var documents: [ResolvedDocument]

    public init(documents: [ResolvedDocument] = []) {
        self.documents = documents
    }

    public func documents(matching title: String) -> [ResolvedDocument] {
        let q = title.lowercased().trimmingCharacters(in: .whitespaces)
        guard !q.isEmpty else { return [] }
        return documents.filter { $0.title.lowercased().contains(q) }
    }
}

/// In-memory location resolver for tests.
public struct StaticLocationResolver: LocationResolver {
    public var coordinates: [String: CalendarEvent.Coordinate]

    public init(coordinates: [String: CalendarEvent.Coordinate] = [:]) {
        self.coordinates = coordinates
    }

    public func coordinate(for location: String) -> CalendarEvent.Coordinate? {
        let key = Self.normalize(location)
        return coordinates.first(where: { Self.normalize($0.key) == key })?.value
    }

    private static func normalize(_ s: String) -> String {
        s.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// MARK: - ParsedEvent

/// The parser's output: everything needed to build a
/// ``CalendarEvent``, minus the persistence fields (id,
/// timestamps). `Codable` so the chat panel can persist a
/// pending quick-add as JSON before the store commits it.
public struct ParsedEvent: Codable, Sendable, Equatable {
    public var title: String
    public var startAt: Date
    public var endAt: Date
    public var allDay: Bool
    public var location: String?
    public var locationCoordinate: CalendarEvent.Coordinate?
    public var attendees: [CalendarEvent.Attendee]
    public var recurrence: CalendarEvent.Recurrence?
    public var linkedDocumentIDs: [UUID]

    public init(
        title: String,
        startAt: Date,
        endAt: Date,
        allDay: Bool = false,
        location: String? = nil,
        locationCoordinate: CalendarEvent.Coordinate? = nil,
        attendees: [CalendarEvent.Attendee] = [],
        recurrence: CalendarEvent.Recurrence? = nil,
        linkedDocumentIDs: [UUID] = []
    ) {
        self.title = title
        self.startAt = startAt
        self.endAt = endAt
        self.allDay = allDay
        self.location = location
        self.locationCoordinate = locationCoordinate
        self.attendees = attendees
        self.recurrence = recurrence
        self.linkedDocumentIDs = linkedDocumentIDs
    }

    /// Convert to a persistable event.
    public func makeEvent(id: UUID = UUID(), now: Date = Date()) -> CalendarEvent {
        CalendarEvent(
            id: id,
            title: title,
            startAt: startAt,
            endAt: endAt,
            allDay: allDay,
            location: location,
            locationCoordinate: locationCoordinate,
            attendees: attendees,
            recurrence: recurrence,
            linkedDocumentIDs: linkedDocumentIDs,
            createdAt: now,
            updatedAt: now
        )
    }
}

// MARK: - CalendarNLUParser

/// Fantastical-style natural language event parsing.
///
/// The engine is deliberately small and rule-based:
///
///  1. **Recurrence** is extracted first with a regex pass
///     ("every monday", "daily", "every 2 weeks",
///     "starting jan 1"). The recurrence span is removed
///     from the input before date detection so the RRULE's
///     "starting" date doesn't fight the event's start.
///  2. **Dates and times** come from `NSDataDetector`
///     (Foundation). The detector handles "tomorrow",
///     "next monday", "2pm-4pm", "noon", "jan 1 3pm".
///  3. **Attendees** come from the "with <names>" span,
///     resolved against the ``ContactsAdapter``.
///  4. **Location** comes from the trailing "in / at / @
///     <place>" span, geocoded against the
///     ``LocationResolver`` cache.
///  5. **Title** is what's left after the spans above are
///     removed, with queue verbs ("schedule", "add")
///     stripped. Empty remainder falls back to
///     "New event".
///
/// Ambiguous input never fails: every stage has a default
/// (today 09:00-10:00, no attendees, no location). The
/// parser is pure — same input, same resolvers, same
/// reference date => same output — which is what makes the
/// test suite deterministic.
public struct CalendarNLUParser: Sendable {

    /// Defaults applied when the input leaves a field
    /// unspecified. Injectable for tests.
    public struct Defaults: Sendable, Equatable {
        /// Time of day for events that name a day but no
        /// time ("Coffee with John tomorrow").
        public var defaultHour: Int
        /// Duration for events that name a start but no end.
        public var defaultDuration: TimeInterval
        /// Fallback title when nothing survives span
        /// extraction.
        public var fallbackTitle: String

        public init(
            defaultHour: Int = 9,
            defaultDuration: TimeInterval = CalendarEvent.defaultDuration,
            fallbackTitle: String = "New event"
        ) {
            self.defaultHour = defaultHour
            self.defaultDuration = defaultDuration
            self.fallbackTitle = fallbackTitle
        }
    }

    private let contacts: ContactsAdapter
    private let documents: DocumentResolver
    private let locations: LocationResolver
    private let defaults: Defaults
    private let calendar: Calendar
    /// "Now" for relative-date resolution. Injectable so
    /// tests are deterministic.
    private let referenceDate: Date

    public init(
        contacts: ContactsAdapter,
        documents: DocumentResolver,
        locations: LocationResolver,
        defaults: Defaults = Defaults(),
        calendar: Calendar = .current,
        referenceDate: Date = Date()
    ) {
        self.contacts = contacts
        self.documents = documents
        self.locations = locations
        self.defaults = defaults
        self.calendar = calendar
        self.referenceDate = referenceDate
    }

    // MARK: - Parse

    public func parse(_ input: String) -> ParsedEvent {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)

        // 1. Recurrence (and the optional "starting <date>"
        //    anchor) is extracted first; its spans are
        //    consumed so step 2 doesn't double-book the
        //    anchor date as the event's start.
        let recurrenceResult = extractRecurrence(from: trimmed)
        return parseBody(
            trimmed,
            ruleSpan: recurrenceResult.range,
            startDateRanges: recurrenceResult.startDateRanges,
            recurrenceAnchor: recurrenceResult.startDate,
            recurrence: recurrenceResult.recurrence
        )
    }

    private func parseBody(
        _ input: String,
        ruleSpan: Range<String.Index>?,
        startDateRanges: [Range<String.Index>],
        recurrenceAnchor: Date?,
        recurrence: CalendarEvent.Recurrence?
    ) -> ParsedEvent {
        var excluded: [Range<String.Index>] = []
        if let ruleSpan { excluded.append(ruleSpan) }
        excluded.append(contentsOf: startDateRanges)

        // 2. Dates + times via NSDataDetector. Hits fully
        //    inside a consumed span ("monday" within "every
        //    monday") and hits overlapping the recurrence
        //    anchor ("starting jan 1") are dropped. A hit
        //    that merely overlaps the rule span ("monday at
        //    9am") survives so its time-of-day feeds the
        //    window - that is how "every monday at 9am
        //    starting jan 1" keeps its 9am.
        let dateHits = detectDates(in: input)
            .filter { hit in
                if startDateRanges.contains(where: { $0.overlaps(hit.range) }) { return false }
                if let ruleSpan, Self.contains(ruleSpan, hit.range) { return false }
                return true
            }
        excluded.append(contentsOf: dateHits.map(\.range))

        // 3. Attendees ("with John and Jane").
        let attendeeResult = extractAttendees(from: input, excluding: dateHits.map(\.range))
        excluded.append(contentsOf: attendeeResult.ranges)

        // 4. Location ("in the blue room").
        let locationResult = extractLocation(from: input, excluding: excluded)
        if let r = locationResult.range { excluded.append(r) }

        // 5. Linked documents ("\"Q3 roadmap\"").
        let documentResult = extractDocuments(from: input, excluding: excluded)
        excluded.append(contentsOf: documentResult.ranges)

        // Assemble the time window.
        let window = resolveWindow(
            dateHits: dateHits,
            anchor: recurrenceAnchor,
            input: input
        )

        // Assemble the title.
        let title = assembleTitle(from: input, excluding: excluded)

        var parsed = ParsedEvent(
            title: title,
            startAt: window.start,
            endAt: window.end,
            allDay: window.allDay,
            location: locationResult.text,
            locationCoordinate: locationResult.text.flatMap { locations.coordinate(for: $0) },
            attendees: attendeeResult.attendees,
            recurrence: recurrence,
            linkedDocumentIDs: documentResult.ids
        )
        if let anchor = recurrenceAnchor, recurrence != nil {
            // Move the window onto the recurrence anchor,
            // preserving both the duration and the parsed
            // time-of-day ("every monday at 9am starting
            // jan 1" keeps its 9am).
            let duration = parsed.endAt.timeIntervalSince(parsed.startAt)
            let time = calendar.dateComponents([.hour, .minute, .second], from: parsed.startAt)
            var day = calendar.dateComponents([.year, .month, .day], from: anchor)
            if !parsed.allDay {
                day.hour = time.hour ?? defaults.defaultHour
                day.minute = time.minute ?? 0
                day.second = time.second ?? 0
            } else {
                day.hour = 0
                day.minute = 0
                day.second = 0
            }
            let start = calendar.date(from: day) ?? anchor
            parsed.startAt = start
            parsed.endAt = start.addingTimeInterval(duration)
        }
        return parsed
    }

    // MARK: - Date detection

    /// The first date detected in `text`, or nil. Exposed
    /// for the chat handler's "move X to <date>" path,
    /// which needs a date without a full event parse.
    public func firstDate(in text: String) -> Date? {
        detectDates(in: text).first?.date
    }

    private struct DateHit: Sendable {
        let range: Range<String.Index>
        let date: Date
        let duration: TimeInterval?
        let matchedText: String
    }

    /// Range containment for same-string index ranges
    /// (`Range.contains` only takes elements, not ranges).
    private static func contains(_ outer: Range<String.Index>, _ inner: Range<String.Index>) -> Bool {
        inner.lowerBound >= outer.lowerBound && inner.upperBound <= outer.upperBound
    }

    private func detectDates(in input: String) -> [DateHit] {
        guard let detector = try? NSDataDetector(types: NSTextCheckingResult.CheckingType.date.rawValue) else {
            return []
        }
        let ns = input as NSString
        let matches = detector.matches(in: input, options: [], range: NSRange(location: 0, length: ns.length))
        return matches.compactMap { m in
            guard let date = m.date, let r = Range(m.range, in: input) else { return nil }
            return DateHit(
                range: r,
                date: date,
                duration: m.duration > 0 ? m.duration : nil,
                matchedText: String(input[r])
            )
        }
    }

    /// True when a matched date expression carries no
    /// time-of-day. Heuristic: the match text has no
    /// am/pm marker, no `h:mm`, and none of the clock
    /// words. The detector's own midnight normalization is
    /// not enough ("tomorrow at 00:00" is a real time).
    private func looksTimed(_ text: String) -> Bool {
        let t = text.lowercased()
        let clockWords = [
            "am", "pm", "noon", "midnight", "morning",
            "afternoon", "evening", "night", "o'clock",
            "oclock",
        ]
        for w in clockWords where t.contains(w) { return true }
        if t.range(of: #"\d{1,2}:\d{2}"#, options: .regularExpression) != nil { return true }
        // Bare numbers that the detector interpreted as a
        // time ("meet at 3").
        if t.range(of: #"\bat\s+\d{1,2}\b"#, options: .regularExpression) != nil { return true }
        return false
    }

    private struct TimeWindow: Sendable {
        var start: Date
        var end: Date
        var allDay: Bool
    }

    private func resolveWindow(
        dateHits: [DateHit],
        anchor: Date?,
        input: String
    ) -> TimeWindow {
        if let first = dateHits.first {
            let timed = looksTimed(first.matchedText)
            var start = first.date
            var end: Date
            if let d = first.duration, d > 0 {
                end = start.addingTimeInterval(d)
            } else if dateHits.count >= 2,
                      dateHits[1].date > first.date,
                      calendar.isDate(dateHits[1].date, inSameDayAs: first.date) {
                // "2pm 4pm" / "tomorrow 2pm to 4pm": second
                // same-day hit is the end.
                end = dateHits[1].date
            } else {
                end = start.addingTimeInterval(defaults.defaultDuration)
            }
            if !timed {
                // Day-only expression ("tomorrow",
                // "next monday"): all-day unless an anchor
                // time was given elsewhere in the input.
                start = rebasedToDefaultTime(calendar.startOfDay(for: start))
                end = start.addingTimeInterval(defaults.defaultDuration)
                return TimeWindow(start: start, end: end, allDay: true)
            }
            return TimeWindow(start: start, end: end, allDay: false)
        }

        // No date detected at all: today at the default
        // time ("Coffee with John").
        if let anchor {
            let timed = !looksAllDay(anchor)
            return TimeWindow(
                start: anchor,
                end: anchor.addingTimeInterval(defaults.defaultDuration),
                allDay: !timed
            )
        }
        let day = calendar.startOfDay(for: referenceDate)
        let start = rebasedToDefaultTime(day)
        return TimeWindow(start: start, end: start.addingTimeInterval(defaults.defaultDuration), allDay: false)
    }

    private func looksAllDay(_ date: Date) -> Bool {
        let c = calendar.dateComponents([.hour, .minute, .second], from: date)
        return (c.hour ?? 0) == 0 && (c.minute ?? 0) == 0 && (c.second ?? 0) == 0
    }

    private func rebasedToDefaultTime(_ day: Date) -> Date {
        var c = calendar.dateComponents([.year, .month, .day], from: day)
        c.hour = defaults.defaultHour
        c.minute = 0
        c.second = 0
        return calendar.date(from: c) ?? day.addingTimeInterval(TimeInterval(defaults.defaultHour * 3600))
    }

    // MARK: - Attendees

    private struct AttendeeResult: Sendable {
        var attendees: [CalendarEvent.Attendee] = []
        var ranges: [Range<String.Index>] = []
    }

    /// "with John and Jane" / "with John, Jane and Bob".
    /// The span runs from the "with" keyword to the end of
    /// the name list; name list parsing stops at the first
    /// non-name token (a preposition like "in" / "at", a
    /// date, or the end of the string).
    private func extractAttendees(from input: String, excluding: [Range<String.Index>]) -> AttendeeResult {
        var result = AttendeeResult()
        let pattern = #"\bwith\s+((?:[A-Z][\p{L}\p{M}.'-]*)(?:\s*(?:,|and|\&|\+)\s*[A-Z][\p{L}\p{M}.'-]*)*)"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []),
              let match = regex.firstMatch(
                  in: input,
                  options: [],
                  range: NSRange(location: 0, length: (input as NSString).length)
              ),
              let fullRange = Range(match.range, in: input),
              let namesRange = Range(match.range(at: 1), in: input)
        else {
            return result
        }
        // Don't consume a span the date detector already
        // owns (guards against "with" inside a date string).
        guard !excluding.contains(where: { $0.overlaps(fullRange) }) else { return result }

        let namesText = String(input[namesRange])
        let names = splitNames(namesText)
        for name in names {
            result.attendees.append(resolveAttendee(named: name))
        }
        if !result.attendees.isEmpty {
            result.ranges.append(fullRange)
        }
        return result
    }

    private func splitNames(_ text: String) -> [String] {
        let parts = text
            .components(separatedBy: CharacterSet(charactersIn: ",&+"))
            .flatMap { $0.components(separatedBy: " and ") }
        return parts
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }

    private func resolveAttendee(named name: String) -> CalendarEvent.Attendee {
        let matches = contacts.contacts(matching: name)
        if let best = matches.first {
            return CalendarEvent.Attendee(
                contactID: best.id,
                email: best.emails.first(where: \.isPrimary)?.value ?? best.emails.first?.value,
                name: best.displayName,
                responseStatus: .needsAction
            )
        }
        return CalendarEvent.Attendee(name: name, responseStatus: .needsAction)
    }

    // MARK: - Location

    private struct LocationResult: Sendable {
        var text: String?
        var range: Range<String.Index>?
    }

    /// Trailing "in <place>" / "at <place>" / "@ <place>".
    /// The span must not overlap a date or attendee span
    /// (that's how "at noon" and "at 2pm" stay times). The
    /// LAST qualifying preposition wins, matching how
    /// people write the place at the end of the sentence.
    private func extractLocation(from input: String, excluding: [Range<String.Index>]) -> LocationResult {
        let pattern = #"\b(?:in|at|@)\s+([^,.;]+)$"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]),
              let match = regex.firstMatch(
                  in: input,
                  options: [],
                  range: NSRange(location: 0, length: (input as NSString).length)
              ),
              let fullRange = Range(match.range, in: input),
              let placeRange = Range(match.range(at: 1), in: input)
        else {
            return LocationResult()
        }
        guard !excluding.contains(where: { $0.overlaps(fullRange) }) else {
            return LocationResult()
        }
        let place = String(input[placeRange]).trimmingCharacters(in: .whitespaces)
        guard !place.isEmpty, place.count <= 80 else { return LocationResult() }
        // A bare time-of-day that slipped past the detector
        // ("at 5") is not a place.
        if place.range(of: #"^\d{1,2}(:\d{2})?\s*(am|pm)?$"#, options: [.regularExpression, .caseInsensitive]) != nil {
            return LocationResult()
        }
        return LocationResult(text: place, range: fullRange)
    }

    // MARK: - Documents

    private struct DocumentResult: Sendable {
        var ids: [UUID] = []
        var ranges: [Range<String.Index>] = []
    }

    /// Double-quoted substrings are document references
    /// ("...review the \"Q3 roadmap\" first"). Each is
    /// matched against the document resolver; unmatched
    /// quotes are left in the title.
    private func extractDocuments(from input: String, excluding: [Range<String.Index>]) -> DocumentResult {
        var result = DocumentResult()
        let pattern = #"\"([^\"]{2,80})\""#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else { return result }
        let matches = regex.matches(in: input, options: [], range: NSRange(location: 0, length: (input as NSString).length))
        for m in matches {
            guard let fullRange = Range(m.range, in: input),
                  let titleRange = Range(m.range(at: 1), in: input) else { continue }
            guard !excluding.contains(where: { $0.overlaps(fullRange) }) else { continue }
            let title = String(input[titleRange])
            if let doc = documents.documents(matching: title).first {
                result.ids.append(doc.id)
                result.ranges.append(fullRange)
            }
        }
        return result
    }

    // MARK: - Recurrence

    private struct RecurrenceResult: Sendable {
        var recurrence: CalendarEvent.Recurrence?
        var range: Range<String.Index>?
        var startDate: Date?
        var startDateRanges: [Range<String.Index>] = []
    }

    /// Extract "every monday" / "daily" / "every 2 weeks" /
    /// "weekly" and the optional "starting <date>" anchor.
    /// All regex work runs against `input` itself (case-
    /// insensitive) so the NSRange -> Range conversions are
    /// always index-consistent.
    private func extractRecurrence(from input: String) -> RecurrenceResult {
        var result = RecurrenceResult()
        let fullRange = NSRange(location: 0, length: (input as NSString).length)

        // "starting <date>" / "from <date>" anchor.
        let startPattern = #"\b(?:starting|starts|beginning|from)\s+([\p{L}0-9 ,'/-]{3,30})"#
        var anchorText: String?
        var anchorRange: Range<String.Index>?
        if let regex = try? NSRegularExpression(pattern: startPattern, options: [.caseInsensitive]),
           let m = regex.firstMatch(in: input, options: [], range: fullRange),
           let r = Range(m.range(at: 1), in: input) {
            anchorText = String(input[r]).trimmingCharacters(in: .whitespaces)
            anchorRange = Range(m.range, in: input)
        }

        // Frequency phrases, longest-first so "every other
        // week" beats "every week".
        let weekdayNames: [(pattern: String, day: RecurrenceRule.Weekday)] = [
            ("mondays?", .monday), ("tuesdays?", .tuesday),
            ("wednesdays?", .wednesday), ("thursdays?", .thursday),
            ("fridays?", .friday), ("saturdays?", .saturday),
            ("sundays?", .sunday),
        ]

        var frequency: RecurrenceRule.Frequency?
        var interval = 1
        var byDay: [RecurrenceRule.Weekday] = []
        var ruleRange: Range<String.Index>?

        for (namePattern, day) in weekdayNames {
            let p = #"\bevery\s+(?:other\s+)?\#(namePattern)\b|\bweekly\s+on\s+\#(namePattern)\b"#
            guard let regex = try? NSRegularExpression(pattern: p, options: [.caseInsensitive]),
                  let m = regex.firstMatch(in: input, options: [], range: fullRange),
                  let r = Range(m.range, in: input) else { continue }
            frequency = .weekly
            byDay = [day]
            interval = input[r].lowercased().contains("other") ? 2 : 1
            ruleRange = r
            break
        }

        if frequency == nil {
            let simple: [(pattern: String, freq: RecurrenceRule.Frequency, interval: Int)] = [
                (#"\bevery\s+other\s+week\b"#, .weekly, 2),
                (#"\bevery\s+(\d+)\s*(?:weeks?|wks?)\b"#, .weekly, -1),
                (#"\bevery\s+weekday\b"#, .weekly, 1),
                (#"\bevery\s+week\b|\bweekly\b"#, .weekly, 1),
                (#"\bevery\s+(\d+)\s*days?\b"#, .daily, -1),
                (#"\bevery\s+day\b|\bdaily\b|\beveryday\b"#, .daily, 1),
                (#"\bevery\s+month\b|\bmonthly\b"#, .monthly, 1),
                (#"\bevery\s+year\b|\byearly\b|\bannually\b"#, .yearly, 1),
            ]
            for (pattern, freq, intv) in simple {
                guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]),
                      let m = regex.firstMatch(in: input, options: [], range: fullRange),
                      let r = Range(m.range, in: input) else { continue }
                frequency = freq
                if intv == -1 {
                    // Capture the number ("every 2 weeks").
                    if let numRange = Range(m.range(at: 1), in: input),
                       let n = Int(input[numRange]) {
                        interval = max(1, n)
                    }
                } else {
                    interval = intv
                }
                if pattern.contains("weekday") {
                    byDay = [.monday, .tuesday, .wednesday, .thursday, .friday]
                }
                ruleRange = r
                break
            }
        }

        guard let frequency else { return result }

        let rule = RecurrenceRule(frequency: frequency, interval: interval, byDay: byDay)
        result.recurrence = CalendarEvent.Recurrence(rrule: rule.rruleString)
        result.range = ruleRange

        // Resolve the anchor date ("starting jan 1").
        if let anchorText, let anchorRange {
            let hits = detectDates(in: anchorText)
            if let first = hits.first {
                result.startDate = first.date
                // Exclude the whole "starting ..." span so
                // the anchor date doesn't leak into the
                // title or double-book as the event start.
                result.startDateRanges.append(anchorRange)
            }
        }
        return result
    }

    // MARK: - Title assembly

    /// Strip the excluded spans from the input, drop queue
    /// verbs, and tidy whitespace / punctuation. Overlapping
    /// spans merge ("every monday" + "monday at 9am" share
    /// the word "monday"); the walk takes their union.
    private func assembleTitle(from input: String, excluding: [Range<String.Index>]) -> String {
        var kept = ""
        var cursor = input.startIndex
        let sorted = excluding.sorted { $0.lowerBound < $1.lowerBound }
        for range in sorted {
            guard range.upperBound > cursor else { continue }
            if range.lowerBound >= cursor {
                kept += input[cursor..<range.lowerBound]
            }
            cursor = range.upperBound
        }
        kept += input[cursor...]

        var title = kept
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
        // Drop leading queue verbs.
        let verbs = [
            "schedule", "add", "create", "make", "new",
            "book", "put", "plan", "set up", "setup",
        ]
        for verb in verbs {
            let p = #"^\#(verb)\s+(?:a\s+|an\s+|the\s+)?"#
            title = title.replacingOccurrences(
                of: p, with: "", options: [.regularExpression, .caseInsensitive]
            )
        }
        // Drop dangling prepositions left behind by span
        // extraction ("Lunch with John in" never happens —
        // the location span includes its preposition — but
        // a "with" at the tail can survive a partial match).
        title = title.replacingOccurrences(
            of: #"\s+(?:with|in|at|on|from|to)\s*$"#,
            with: "",
            options: [.regularExpression, .caseInsensitive]
        )
        title = title.trimmingCharacters(in: .whitespacesAndNewlines)
        title = title.trimmingCharacters(in: CharacterSet(charactersIn: ",.;- "))
        return title.isEmpty ? defaults.fallbackTitle : title
    }
}
