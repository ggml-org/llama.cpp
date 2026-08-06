import Foundation

// MARK: - Reminder

/// A calendar-event-relative reminder (e.g. "15 min before the
/// Q3 review meeting"). Reminders are the user-facing layer on
/// top of `UNUserNotificationCenter` — each reminder maps to one
/// `UNCalendarNotificationTrigger` at `triggerAt`.
///
/// Reminders are stored as `graph_entity` rows with
/// `entity_type = 'reminder'`, following the universal
/// "one row per thing" pattern documented in
/// `docs/tessera-data-layer-design.md` §3. The `body` column
/// carries the reminder fields as JSON; `label` is the title.
/// Migrations 0001 (data layer) + 0005 (reminder index) cover
/// the schema.
///
/// **Constitutional invariant:** every mutation
/// (create / update / acknowledge / snooze / delete) is a
/// receipt. The receipt chain is the audit trail the user
/// sees in the receipt drawer.
///
/// The struct is intentionally self-contained: no
/// `UNUserNotificationCenter` references, no data-layer
/// imports beyond `Foundation`. ``ReminderStore`` wires the
/// struct to the data layer; ``ReminderNotificationScheduler``
/// wires it to the notification center.
public struct Reminder: Codable, Sendable, Identifiable, Hashable {

    /// Stable identifier. New reminders get a fresh UUID at
    /// creation; importer round-trips preserve the existing id
    /// so re-imports upsert in place rather than duplicating.
    public let id: UUID

    /// Short user-visible headline ("15 min before Q3 review").
    /// The graph view uses this as the node label; the list
    /// view shows it as the row title.
    public var title: String

    /// Free-form description shown in the detail view. Empty
    /// string when the user didn't fill it in (we use empty
    /// rather than nil so the JSON shape is stable).
    public var notes: String

    /// The calendar event this reminder is tied to. Every
    /// reminder is for an event — the offset is relative to the
    /// event's start. The calendar event's `start + offsetMinutes`
    /// is what `triggerAt` captures.
    public var calendarEventID: UUID

    /// Minutes relative to the linked event's start. Negative
    /// for "before" (-15 = "15 min before"), positive for
    /// "after" (60 = "1 hour after"). Zero is a degenerate case
    /// ("at the start of the event"); we don't reject it.
    public var offsetMinutes: Int

    /// The wall-clock time the reminder fires. Stored
    /// redundantly with `calendarEventID + offsetMinutes` so
    /// the list view can sort and filter by time without
    /// joining back to the event row. The store recomputes
    /// `triggerAt` when the user edits the offset (so the
    /// persisted value is always the source of truth — the
    /// event start is the independent variable, the offset is
    /// a user-facing knob, and `triggerAt` is the
    /// denominator).
    public var triggerAt: Date

    /// When the user dismissed/acted on the reminder. nil
    /// while the reminder is still pending. The list view's
    /// "Acknowledged" filter shows reminders with a non-nil
    /// `acknowledgedAt`.
    public var acknowledgedAt: Date?

    /// When the user snoozed the reminder until. nil while
    /// the reminder is not snoozed. A snoozed reminder
    /// (a) has a non-nil `snoozedUntil` in the future, and
    /// (b) has a scheduled notification in the notification
    /// center at `snoozedUntil` (the store cancels the
    /// original `triggerAt` notification and re-schedules).
    public var snoozedUntil: Date?

    /// User-set priority. Shared with the future Tasks
    /// material — the enum lives in its own file so both
    /// surfaces can adopt it without cross-dependency.
    public var priority: TesseraTaskPriority

    public var createdAt: Date
    public var updatedAt: Date

    public init(
        id: UUID = UUID(),
        title: String,
        notes: String = "",
        calendarEventID: UUID,
        offsetMinutes: Int,
        triggerAt: Date,
        acknowledgedAt: Date? = nil,
        snoozedUntil: Date? = nil,
        priority: TesseraTaskPriority = .none,
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.title = title
        self.notes = notes
        self.calendarEventID = calendarEventID
        self.offsetMinutes = offsetMinutes
        self.triggerAt = triggerAt
        self.acknowledgedAt = acknowledgedAt
        self.snoozedUntil = snoozedUntil
        self.priority = priority
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    /// The `entity_type` value persisted in
    /// `graph_entities.entity_type`. Pinned so the
    /// Reminders surface's `hybrid_search` filter and the
    /// graph view's "by type" grouping all agree.
    public static let entityType = "reminder"

    // MARK: - Convenience

    /// True while the reminder is still in the future AND
    /// the user hasn't acted on it AND it isn't snoozed.
    /// The list view's "Upcoming" filter uses this — the
    /// simpler expression `(triggerAt > now AND
    /// acknowledgedAt == nil AND snoozedUntil == nil)` reads
    /// worse at the call site.
    public func isUpcoming(now: Date = Date()) -> Bool {
        guard acknowledgedAt == nil else { return false }
        if let snooze = snoozedUntil, snooze > now { return false }
        return triggerAt > now
    }

    /// True when the user snoozed the reminder AND the snooze
    /// is still in the future. Used by the "Snoozed" filter
    /// and by the row icon.
    public func isSnoozed(now: Date = Date()) -> Bool {
        guard let snooze = snoozedUntil else { return false }
        return snooze > now
    }

    /// True when the user has dismissed / acknowledged the
    /// reminder. Used by the "Acknowledged" filter.
    public func isAcknowledged() -> Bool {
        acknowledgedAt != nil
    }

    /// Human-readable offset label, e.g. "15 min before",
    /// "1 hour after", "at start". Used by the list row and
    /// the detail view header.
    public var offsetLabel: String {
        Self.formatOffset(offsetMinutes)
    }

    /// Static formatter so the chat panel's
    /// "remind me 15 min before the Q3 review meeting" can
    /// produce a "15 min before" chip without an instance.
    public static func formatOffset(_ minutes: Int) -> String {
        if minutes == 0 { return "at start" }
        let abs = Swift.abs(minutes)
        let unit: String
        let value: Int
        if abs % 60 == 0 {
            value = abs / 60
            unit = value == 1 ? "hour" : "hours"
        } else {
            value = abs
            unit = "min"
        }
        let phrase = value == 1 && unit == "min" ? "1 min" : "\(value) \(unit)"
        if minutes < 0 { return "\(phrase) before" }
        return "\(phrase) after"
    }

    /// Display string for the row / detail header. Includes
    /// the title and the offset, e.g.
    /// "15 min before Q3 review meeting".
    public func displayLine(calendarEventTitle: String? = nil) -> String {
        let tail: String
        if let e = calendarEventTitle, !e.isEmpty {
            tail = e
        } else {
            tail = "event"
        }
        return "\(offsetLabel) \(tail)"
    }

    // Equatable + Hashable are auto-synthesized across all
    // stored properties (Reminder is `Identifiable,
    // Hashable`). Two reminders with the same fields hash
    // and compare equal; Set<Reminder> lookups work the
    // way the user expects.
}

// MARK: - JSON helpers

extension Reminder {
    /// Encode the reminder to JSON. The encoded form is what
    /// the `body` column of `graph_entities` stores. Uses
    /// ISO-8601 dates + sorted keys so the bytes are
    /// deterministic (the receipt drawer and the agent's
    /// `hybrid_search` both read the same bytes).
    public func jsonData() throws -> Data {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(self)
    }

    /// Decode a reminder from JSON. The inverse of
    /// ``jsonData()``.
    public static func from(jsonData: Data) throws -> Reminder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(Reminder.self, from: jsonData)
    }

    /// UTF-8 string round-trip. The data layer's
    /// `body` column is a `text`, so the store layer
    /// (and the chat panel's receipt chip) read/write
    /// strings rather than `Data`.
    func jsonDataString() throws -> String {
        let data = try jsonData()
        guard let s = String(data: data, encoding: .utf8) else {
            throw ReminderStoreError.invalidReminderBody(reason: "encoded JSON is not UTF-8")
        }
        return s
    }

    static func from(jsonDataString: String) throws -> Reminder {
        guard let data = jsonDataString.data(using: .utf8) else {
            throw ReminderStoreError.invalidReminderBody(reason: "body is not valid UTF-8")
        }
        return try from(jsonData: data)
    }
}

// MARK: - Shared priority enum

/// User-set priority. The enum lives in the Reminders
/// directory because Reminders is the first material to
/// adopt it; the future Tasks material will adopt the same
/// type (per spec §12.2's "shared enum with Tasks"). We use
/// the `TesseraTaskPriority` name (not `ReminderPriority`)
/// so a future Tasks material can use it without churn.
///
/// **Important:** this is NOT `Foundation.Task.Priority`
/// (which is the Swift concurrency priority). The two types
/// are unrelated — `TesseraTaskPriority` is the user's UI
/// priority (none / low / medium / high), and
/// `Foundation.Task.Priority` is the scheduler priority
/// (background / userInitiated / ...). The store / view
/// never reaches for the concurrency version.
public enum TesseraTaskPriority: String, Codable, Sendable, Hashable, CaseIterable, Comparable {
    case none
    case low
    case medium
    case high

    /// Sort ordering for the list view: high at the top, none
    /// at the bottom. The list view does the time sort
    /// separately; this ordering is used when two reminders
    /// have the same `triggerAt` (the user expects the more
    /// important one first).
    public static func < (lhs: TesseraTaskPriority, rhs: TesseraTaskPriority) -> Bool {
        lhs.rank > rhs.rank
    }

    private var rank: Int {
        switch self {
        case .none: return 0
        case .low: return 1
        case .medium: return 2
        case .high: return 3
        }
    }

    /// Short label for the row badge (e.g. "!" for high).
    public var shortLabel: String {
        switch self {
        case .none: return ""
        case .low: return "·"
        case .medium: return "•"
        case .high: return "!"
        }
    }
}
