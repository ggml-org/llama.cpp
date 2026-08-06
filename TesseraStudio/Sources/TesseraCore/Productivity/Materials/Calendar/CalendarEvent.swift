import Foundation

// MARK: - CalendarEvent

/// A first-class material in the Tessera productivity surface.
///
/// Calendar events are stored as `graph_entity` rows with
/// `entity_type = 'calendar_event'`, matching the universal
/// "one row per thing" pattern documented in
/// `docs/tessera-data-layer-design.md` §3. The `body` column
/// carries the event fields as JSON; `label` is the event
/// title so the graph view and `searchByLabelPrefix` work
/// without a calendar-specific query.
///
/// The struct is intentionally self-contained: no references
/// to the data layer, no framework imports beyond
/// `Foundation`. The ``CalendarNLUParser`` builds
/// `CalendarEvent` values from natural language; the
/// ``CalendarStore`` writes them to Postgres and produces
/// the constitutional receipt.
public struct CalendarEvent: Codable, Sendable, Identifiable, Hashable {

    /// Stable identifier. New events get a fresh UUID at
    /// creation; the id doubles as the `graph_entities.id`
    /// and as the target of `entity_links` rows.
    public let id: UUID

    public var title: String
    public var notes: String

    /// Start instant. Stored as an absolute `Date` (encoded
    /// ISO-8601 in the JSONB body), so all-day and timed
    /// events share one representation. All-day events are
    /// anchored to midnight of the event's local day; the
    /// `allDay` flag tells the view layer to ignore the
    /// time-of-day.
    public var startAt: Date
    public var endAt: Date

    /// True for date-only events (birthdays, deadlines,
    /// "out of office"). The day / week / month views render
    /// these in the all-day lane instead of the hour grid.
    public var allDay: Bool

    /// Free-form location string ("the blue room",
    /// "Blue Bottle on Market St"). Displayed verbatim.
    public var location: String?

    /// Geocoded coordinate for the location, when the
    /// ``CalendarLocationResolving`` resolver found one.
    /// Optional because most locations never get geocoded
    /// (geocoding is best-effort, offline-safe).
    public var locationCoordinate: Coordinate?

    /// Attendees. Each entry is either a linked contact
    /// (`contactID` set) or an external attendee (`email`
    /// and / or display name only).
    public var attendees: [Attendee]

    /// RFC 5545 RRULE recurrence, when the event repeats.
    public var recurrence: Recurrence?

    /// Linked reminder entity ids (the Reminders surface
    /// owns the rows; the calendar only carries the
    /// references + the `entity_links` edges).
    public var reminders: [UUID]

    /// Documents that are prep / agenda material for this
    /// event. Mirrored as `entity_links` rows by the store.
    public var linkedDocumentIDs: [UUID]

    /// Tasks that are prep for this event. Mirrored as
    /// `entity_links` rows by the store.
    public var linkedTaskIDs: [UUID]

    public var createdAt: Date
    public var updatedAt: Date

    public init(
        id: UUID = UUID(),
        title: String,
        notes: String = "",
        startAt: Date,
        endAt: Date,
        allDay: Bool = false,
        location: String? = nil,
        locationCoordinate: Coordinate? = nil,
        attendees: [Attendee] = [],
        recurrence: Recurrence? = nil,
        reminders: [UUID] = [],
        linkedDocumentIDs: [UUID] = [],
        linkedTaskIDs: [UUID] = [],
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.title = title
        self.notes = notes
        self.startAt = startAt
        self.endAt = endAt
        self.allDay = allDay
        self.location = location
        self.locationCoordinate = locationCoordinate
        self.attendees = attendees
        self.recurrence = recurrence
        self.reminders = reminders
        self.linkedDocumentIDs = linkedDocumentIDs
        self.linkedTaskIDs = linkedTaskIDs
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    /// The `entity_type` value persisted in
    /// `graph_entities.entity_type`. Pinned so the calendar
    /// store, the graph view's node color / icon table, and
    /// the migration's partial index all agree. Matches the
    /// `calendar_event` case in `GraphNode.iconName(for:)`.
    public static let entityType = "calendar_event"

    /// The default event duration the NLU parser and the
    /// quick-add path apply when the user gave a start time
    /// but no end time (Fantastical uses one hour too).
    public static let defaultDuration: TimeInterval = 3600

    // MARK: - Validation

    /// True when the event's invariants hold. The store
    /// refuses to persist events that fail this check.
    public var isValid: Bool {
        if title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return false }
        if !allDay && endAt < startAt { return false }
        if allDay && endAt < startAt { return false }
        return true
    }

    // MARK: - Occurrence queries

    /// The instants this event occupies in `[rangeStart,
    /// rangeEnd)`. A non-recurring event contributes its
    /// own start when it overlaps the range; a recurring
    /// event expands its RRULE (capped) and skips
    /// `recurrence.exDates`.
    public func occurrences(in range: ClosedRange<Date>, calendar: Calendar = .current) -> [Date] {
        // An unparseable RRULE degrades to the single
        // occurrence (never silently drop the event).
        guard let recurrence, let rule = try? RecurrenceRule(rrule: recurrence.rrule) else {
            let duration = max(endAt.timeIntervalSince(startAt), 0)
            let overlaps = startAt <= range.upperBound && startAt.addingTimeInterval(duration) >= range.lowerBound
            return overlaps ? [startAt] : []
        }
        let exDateDays = Set(recurrence.exDates.map { calendar.startOfDay(for: $0) })
        return rule.occurrences(anchor: startAt, in: range, calendar: calendar)
            .filter { !exDateDays.contains(calendar.startOfDay(for: $0)) }
    }

    /// True when the event occupies any instant in the
    /// given day (used by the month grid's markers).
    public func occurs(on day: Date, calendar: Calendar = .current) -> Bool {
        let dayRange = calendar.startOfDay(for: day)...calendar.startOfDay(for: day).addingTimeInterval(86_400)
        return !occurrences(in: dayRange, calendar: calendar).isEmpty
    }
}

// MARK: - Attendee

extension CalendarEvent {

    /// One attendee. Either a linked contact (`contactID`
    /// set — the graph view draws the `attendee_of` edge)
    /// or an external attendee (email / name only). Both
    /// `contactID` and `email` nil is allowed for a
    /// name-only attendee the resolver couldn't match.
    public struct Attendee: Codable, Sendable, Hashable {
        public var contactID: UUID?
        public var email: String?
        public var name: String
        public var responseStatus: ResponseStatus

        public init(
            contactID: UUID? = nil,
            email: String? = nil,
            name: String,
            responseStatus: ResponseStatus = .needsAction
        ) {
            self.contactID = contactID
            self.email = email
            self.name = name
            self.responseStatus = responseStatus
        }

        /// True when this attendee resolved to a contact in
        /// the graph (the store only creates an
        /// `attendee_of` edge for resolved attendees).
        public var isResolved: Bool { contactID != nil }
    }

    /// RFC 5545 PARTSTAT vocabulary (subset).
    public enum ResponseStatus: String, Codable, Sendable, Hashable, CaseIterable {
        case accepted
        case declined
        case tentative
        case needsAction
    }

    /// RFC 5545 RRULE string + exception dates. The rrule
    /// string is the serialized form ("FREQ=WEEKLY;BYDAY=MO");
    /// ``RecurrenceRule`` parses it for occurrence expansion.
    public struct Recurrence: Codable, Sendable, Hashable {
        public var rrule: String
        /// Exception dates (RFC 5545 EXDATE). An occurrence
        /// whose local day matches an exDate day is skipped.
        public var exDates: [Date]

        public init(rrule: String, exDates: [Date] = []) {
            self.rrule = rrule
            self.exDates = exDates
        }
    }

    /// A geocoded location. Kept as a plain struct (not
    /// `CLLocationCoordinate2D`) so the model has no
    /// CoreLocation dependency — the resolver is the one
    /// place that touches CoreLocation.
    public struct Coordinate: Codable, Sendable, Hashable {
        public var latitude: Double
        public var longitude: Double

        public init(latitude: Double, longitude: Double) {
            self.latitude = latitude
            self.longitude = longitude
        }
    }
}

// MARK: - JSON helpers

extension CalendarEvent {

    /// Encode to JSON data (ISO-8601 dates, matching the
    /// `Contact` convention).
    public func jsonData() throws -> Data {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(self)
    }

    /// Decode from JSON data. Inverse of ``jsonData()``.
    public static func from(jsonData data: Data) throws -> CalendarEvent {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(CalendarEvent.self, from: data)
    }

    /// Encode to a UTF-8 string. Convenience for the store,
    /// which deals in `String` bodies for the data layer.
    public func jsonDataString() throws -> String {
        let data = try jsonData()
        guard let s = String(data: data, encoding: .utf8) else {
            throw CalendarStoreError.invalidEventBody(reason: "encoded JSON is not UTF-8")
        }
        return s
    }

    /// Decode from a UTF-8 string. Inverse of
    /// ``jsonDataString()``.
    public static func from(jsonDataString: String) throws -> CalendarEvent {
        guard let data = jsonDataString.data(using: .utf8) else {
            throw CalendarStoreError.invalidEventBody(reason: "body is not valid UTF-8")
        }
        return try from(jsonData: data)
    }
}
