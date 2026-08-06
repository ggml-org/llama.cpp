import Foundation

// MARK: - ReminderCreateTool

/// Agent tool: create a reminder linked to a calendar event.
///
/// The agent calls this when the user says something like
/// "remind me 15 min before the Q3 review meeting". The
/// tool's input is the calendar event id (resolved by the
/// agent via the data layer's hybrid search) plus the
/// offset. The store creates the row, writes the receipt,
/// and the caller is expected to also schedule a
/// notification via ``ReminderNotificationScheduler``.
///
/// **Approval.** The default is `notify` — the tool mutates
/// user-visible state (adds a notification to the
/// notification center) but the user's intent is explicit in
/// the chat input. We surface the resulting reminder in the
/// receipt chip so the user can see what was created.
public struct ReminderCreateTool: TesseraTool {
    public let name = "reminder_create"
    public let description = """
Create a calendar-event-relative reminder. Returns the new reminder's id and trigger time. \
Use this when the user says "remind me X min before the Y event" — first resolve the calendar \
event id (via hybrid_search with entity_type='calendar_event' or by listing events), then call \
this tool with the offset in minutes (negative = before, positive = after, 0 = at start).
"""
    public let defaultApprovalLevel = ApprovalLevel.notify

    public let parameters = JSONSchema(
        type: "object",
        properties: [
            "title": SchemaProperty(
                type: "string",
                description: "Short headline for the reminder. e.g. '15 min before Q3 review meeting'."
            ),
            "calendar_event_id": SchemaProperty(
                type: "string",
                description: "UUID of the linked calendar event."
            ),
            "offset_minutes": SchemaProperty(
                type: "integer",
                description: "Minutes relative to the event's start. Negative = before, positive = after."
            ),
            "trigger_at": SchemaProperty(
                type: "string",
                description: "ISO-8601 timestamp when the reminder fires. The agent can compute this from the event start + offset."
            ),
            "notes": SchemaProperty(
                type: "string",
                description: "Free-form notes shown in the detail view. Optional."
            ),
            "priority": SchemaProperty(
                type: "string",
                description: "One of: none, low, medium, high. Default none.",
                enumValues: ["none", "low", "medium", "high"],
                defaultValue: "none"
            ),
        ],
        required: ["title", "calendar_event_id", "offset_minutes", "trigger_at"]
    )

    private let store: any ReminderStoring

    public init(store: any ReminderStoring) {
        self.store = store
    }

    public func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        guard let title = arguments["title"]?.stringValue, !title.isEmpty else {
            return .fail("title is required")
        }
        guard let eventIDString = arguments["calendar_event_id"]?.stringValue,
              let eventID = UUID(uuidString: eventIDString) else {
            return .fail("calendar_event_id must be a UUID")
        }
        guard let offsetRaw = arguments["offset_minutes"]?.numberValue else {
            return .fail("offset_minutes is required (integer)")
        }
        let offsetMinutes = Int(offsetRaw)
        guard let triggerAtString = arguments["trigger_at"]?.stringValue,
              let triggerAt = Self.parseISO8601(triggerAtString) else {
            return .fail("trigger_at must be ISO-8601")
        }
        let notes = arguments["notes"]?.stringValue ?? ""
        let priorityRaw = arguments["priority"]?.stringValue ?? "none"
        let priority = TesseraTaskPriority(rawValue: priorityRaw) ?? .none
        let now = Date()

        let reminder = Reminder(
            title: title,
            notes: notes,
            calendarEventID: eventID,
            offsetMinutes: offsetMinutes,
            triggerAt: triggerAt,
            priority: priority,
            createdAt: now,
            updatedAt: now
        )
        let saved = try await store.upsert(reminder)
        return .ok(
            "Reminder created: \(saved.displayLine())",
            data: [
                "reminder_id": .string(saved.id.uuidString),
                "calendar_event_id": .string(saved.calendarEventID.uuidString),
                "title": .string(saved.title),
                "offset_minutes": .number(Double(saved.offsetMinutes)),
                "trigger_at": .string(Self.iso8601(saved.triggerAt)),
                "priority": .string(saved.priority.rawValue),
            ]
        )
    }

    private static func parseISO8601(_ s: String) -> Date? {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let d = f.date(from: s) { return d }
        f.formatOptions = [.withInternetDateTime]
        return f.date(from: s)
    }

    private static func iso8601(_ d: Date) -> String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f.string(from: d)
    }
}

// MARK: - ReminderListTool

/// Agent tool: list reminders, optionally filtered.
///
/// Returns the most recent N reminders (default 20) sorted
/// by `triggerAt` ascending. The agent uses this to answer
/// "what are my reminders today?" or "what reminders do I
/// have for the Q3 review meeting?".
public struct ReminderListTool: TesseraTool {
    public let name = "reminder_list"
    public let description = """
List reminders. Returns the most recent N reminders sorted by trigger time ascending. \
Use this when the user asks "what are my reminders?" or "what reminders do I have for event X?". \
The result is a compact text table the chat panel renders inline.
"""
    public let defaultApprovalLevel = ApprovalLevel.notify

    public let parameters = JSONSchema(
        type: "object",
        properties: [
            "limit": SchemaProperty(
                type: "integer",
                description: "Max reminders to return. Default 20.",
                defaultValue: "20"
            ),
            "calendar_event_id": SchemaProperty(
                type: "string",
                description: "Optional: scope to a single calendar event."
            ),
            "filter": SchemaProperty(
                type: "string",
                description: "Optional filter: upcoming, acknowledged, snoozed, all. Default upcoming.",
                enumValues: ["upcoming", "acknowledged", "snoozed", "all"],
                defaultValue: "upcoming"
            ),
        ],
        required: []
    )

    private let store: any ReminderStoring

    public init(store: any ReminderStoring) {
        self.store = store
    }

    public func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        let limit = arguments["limit"]?.numberValue.map { max(1, Int($0)) } ?? 20
        let filterRaw = arguments["filter"]?.stringValue ?? "upcoming"
        let filter = ReminderFilter(rawValue: filterRaw) ?? .upcoming
        let scopeEvent = arguments["calendar_event_id"]?.stringValue
            .flatMap(UUID.init(uuidString:))

        let raw = try await store.list(limit: max(limit, 200))
        let scoped: [Reminder]
        if let e = scopeEvent {
            scoped = raw.filter { $0.calendarEventID == e }
        } else {
            scoped = raw
        }
        let bucketed = filter.apply(to: scoped)
        let sliced = Array(bucketed.prefix(limit))

        let lines = sliced.map { r -> String in
            let when = r.isSnoozed()
                ? "snoozed until \(Self.relative(r.snoozedUntil!))"
                : (r.isAcknowledged()
                    ? "acknowledged"
                    : Self.relative(r.triggerAt))
            return "- [\(r.priority.rawValue)] \(r.title) — \(when)"
        }
        let text = lines.isEmpty
            ? "No reminders match."
            : lines.joined(separator: "\n")
        return .ok(text, data: [
            "count": .number(Double(sliced.count)),
            "filter": .string(filter.rawValue),
            "reminder_ids": .array(sliced.map { .string($0.id.uuidString) }),
        ])
    }

    private static func relative(_ d: Date) -> String {
        let f = RelativeDateTimeFormatter()
        f.unitsStyle = .full
        return f.localizedString(for: d, relativeTo: Date())
    }
}

// MARK: - ReminderDismissTool

/// Agent tool: acknowledge / dismiss a reminder.
///
/// The agent uses this when the user says "dismiss the Q3
/// review reminder" — the agent resolves the title to a
/// reminder id (via the data layer) and calls this tool.
/// The store writes the `acknowledgedAt` timestamp and the
/// reminder_acknowledged receipt; the caller is expected to
/// also cancel the pending notification.
public struct ReminderDismissTool: TesseraTool {
    public let name = "reminder_dismiss"
    public let description = """
Dismiss (acknowledge) a reminder. Sets acknowledgedAt to the current time and cancels any \
pending notification. The reminder is no longer in the Upcoming filter; it shows in the \
Acknowledged filter.
"""
    public let defaultApprovalLevel = ApprovalLevel.notify

    public let parameters = JSONSchema(
        type: "object",
        properties: [
            "reminder_id": SchemaProperty(
                type: "string",
                description: "UUID of the reminder to dismiss."
            ),
        ],
        required: ["reminder_id"]
    )

    private let store: any ReminderStoring

    public init(store: any ReminderStoring) {
        self.store = store
    }

    public func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        guard let idString = arguments["reminder_id"]?.stringValue,
              let id = UUID(uuidString: idString) else {
            return .fail("reminder_id must be a UUID")
        }
        let updated = try await store.acknowledge(id: id)
        guard let r = updated else {
            return .fail("reminder not found: \(id.uuidString)")
        }
        return .ok(
            "Reminder dismissed: \(r.title)",
            data: [
                "reminder_id": .string(r.id.uuidString),
                "acknowledged_at": .string(Self.iso8601(r.acknowledgedAt ?? Date())),
            ]
        )
    }

    private static func iso8601(_ d: Date) -> String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f.string(from: d)
    }
}

// MARK: - ReminderSnoozeTool

/// Agent tool: snooze a reminder.
///
/// The agent uses this when the user says "snooze the Q3
/// review reminder for 10 min" — the agent resolves the
/// title to a reminder id and calls this tool with the
/// snooze duration. The store writes `snoozedUntil`; the
/// caller is expected to also cancel + re-schedule the
/// notification.
public struct ReminderSnoozeTool: TesseraTool {
    public let name = "reminder_snooze"
    public let description = """
Snooze a reminder. The reminder fires again at the snooze time. Pass the snooze duration in \
minutes; max 1440 (one day) per spec §12.3.
"""
    public let defaultApprovalLevel = ApprovalLevel.notify

    public let parameters = JSONSchema(
        type: "object",
        properties: [
            "reminder_id": SchemaProperty(
                type: "string",
                description: "UUID of the reminder to snooze."
            ),
            "snooze_minutes": SchemaProperty(
                type: "integer",
                description: "Snooze duration in minutes. Min 1, max 1440 (one day per spec).",
                minimum: 1,
                maximum: 1440
            ),
        ],
        required: ["reminder_id", "snooze_minutes"]
    )

    private let store: any ReminderStoring

    public init(store: any ReminderStoring) {
        self.store = store
    }

    public func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        guard let idString = arguments["reminder_id"]?.stringValue,
              let id = UUID(uuidString: idString) else {
            return .fail("reminder_id must be a UUID")
        }
        guard let minutesRaw = arguments["snooze_minutes"]?.numberValue else {
            return .fail("snooze_minutes is required (integer)")
        }
        let minutes = Int(minutesRaw)
        guard minutes >= 1, minutes <= 1440 else {
            return .fail("snooze_minutes must be between 1 and 1440")
        }
        let until = Date().addingTimeInterval(Double(minutes) * 60)
        let updated = try await store.snooze(id: id, until: until)
        guard let r = updated else {
            return .fail("reminder not found: \(id.uuidString)")
        }
        return .ok(
            "Reminder snoozed for \(minutes) min: \(r.title)",
            data: [
                "reminder_id": .string(r.id.uuidString),
                "snoozed_until": .string(Self.iso8601(r.snoozedUntil ?? until)),
            ]
        )
    }

    private static func iso8601(_ d: Date) -> String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f.string(from: d)
    }
}
