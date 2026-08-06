import Foundation

// MARK: - CalendarStore

/// The calendar-specific seam between the NLU parser / the
/// SwiftUI views and the data layer facade
/// (``TesseraDataLayer``). Mirrors ``ContactStore``'s role
/// for the contact material: own the calendar event <->
/// data layer integration, enforce the constitutional
/// receipt invariant for every mutation, expose a
/// domain-shaped API to the rest of the productivity
/// surface.
///
/// **Receipt contract:** every mutation that changes event
/// state (create, update, delete, respond, link) appends a
/// receipt to `graph_receipts` with a `receipt_type` from
/// ``CalendarEventReceiptType`` and a payload naming the
/// affected `entity_id` plus a structural summary. The
/// receipt chain IS the event's history: the drawer renders
/// `receipts(forEvent:)` directly.
///
/// **Range queries.** The data layer facade exposes
/// `listByEntityType` + `searchByLabelPrefix`, not a JSONB
/// range query; the store loads the calendar rows and
/// filters / expands occurrences in memory. The
/// `idx_entities_event_start` index (migration 0006) keeps
/// the server-side scan narrow for a future range query on
/// the facade.
public struct CalendarStore: Sendable {

    private let dataLayer: TesseraDataLayer

    public init(dataLayer: TesseraDataLayer) {
        self.dataLayer = dataLayer
    }

    // MARK: - CRUD

    /// Persist an event. New ids insert; existing ids update.
    /// Emits `event_created` for a new event, `event_updated`
    /// for a known id. Attendee / document / task links are
    /// synced to `entity_links` (idempotent inserts).
    @discardableResult
    public func upsert(_ event: CalendarEvent) async throws -> CalendarEvent {
        guard event.isValid else {
            throw CalendarStoreError.invalidEvent(reason: "title empty or endAt before startAt")
        }
        let existing = try await dataLayer.getEntity(id: event.id)
        let isNew = existing == nil || existing?.entityType != CalendarEvent.entityType

        let body = try event.jsonDataString()
        _ = try await dataLayer.upsertEntity(
            GraphEntityUpsert(
                id: event.id,
                entityType: CalendarEvent.entityType,
                subtype: event.allDay ? "all_day" : "timed",
                label: event.title,
                body: body,
                sourceURL: nil,
                embedding: nil
            )
        )

        var payload: [String: JSONValue] = [
            "title": .string(event.title),
            "startAt": .string(Self.iso8601.string(from: event.startAt)),
            "endAt": .string(Self.iso8601.string(from: event.endAt)),
            "allDay": .bool(event.allDay),
        ]
        if let recurrence = event.recurrence {
            payload["rrule"] = .string(recurrence.rrule)
        }
        if let location = event.location {
            payload["location"] = .string(location)
        }
        payload["attendeeCount"] = .number(Double(event.attendees.count))

        try await appendReceipt(
            entityID: event.id,
            receiptType: isNew
                ? CalendarEventReceiptType.eventCreated.rawValue
                : CalendarEventReceiptType.eventUpdated.rawValue,
            payload: payload
        )

        try await syncLinks(for: event)
        return event
    }

    /// Fetch one event by id. Returns nil when no
    /// `graph_entity` row with `entity_type =
    /// 'calendar_event'` matches.
    public func get(id: UUID) async throws -> CalendarEvent? {
        guard let entity = try await dataLayer.getEntity(id: id) else {
            return nil
        }
        return try Self.eventFromEntity(entity)
    }

    /// Delete an event by id. Emits an `event_deleted`
    /// receipt after the row is gone (the receipt row
    /// survives: `graph_receipts.entity_id` has no cascading
    /// FK, same as the contact surface — the receipt chain
    /// is the audit trail of the deletion).
    @discardableResult
    public func delete(id: UUID) async throws -> Bool {
        // Capture the title before the delete so the receipt
        // payload can say what was deleted.
        let doomed = try await get(id: id)
        let didDelete = try await dataLayer.deleteEntity(id: id)
        if didDelete {
            var payload: [String: JSONValue] = [:]
            if let doomed {
                payload["title"] = .string(doomed.title)
                payload["startAt"] = .string(Self.iso8601.string(from: doomed.startAt))
            }
            try await appendReceipt(
                entityID: id,
                receiptType: CalendarEventReceiptType.eventDeleted.rawValue,
                payload: payload
            )
        }
        return didDelete
    }

    // MARK: - Responding (RSVP)

    /// Record the user's response to an event. Updates the
    /// attendee entry identified by `attendeeIndex` (or the
    /// first entry matching `attendeeName` when the index
    /// is nil) and emits an `event_responded` receipt.
    @discardableResult
    public func respond(
        to eventID: UUID,
        attendeeIndex: Int? = nil,
        attendeeName: String? = nil,
        status: CalendarEvent.ResponseStatus
    ) async throws -> CalendarEvent {
        guard var event = try await get(id: eventID) else {
            throw CalendarStoreError.eventNotFound(id: eventID)
        }
        guard !event.attendees.isEmpty else {
            throw CalendarStoreError.noAttendees(eventID: eventID)
        }

        let index: Int
        if let attendeeIndex, event.attendees.indices.contains(attendeeIndex) {
            index = attendeeIndex
        } else if let attendeeName,
                  let found = event.attendees.firstIndex(where: {
                      $0.name.caseInsensitiveCompare(attendeeName) == .orderedSame
                  }) {
            index = found
        } else {
            index = 0
        }

        event.attendees[index].responseStatus = status
        event.updatedAt = Date()
        _ = try await upsertEntityRow(event)
        try await appendReceipt(
            entityID: eventID,
            receiptType: CalendarEventReceiptType.eventResponded.rawValue,
            payload: [
                "attendee": .string(event.attendees[index].name),
                "status": .string(status.rawValue),
            ]
        )
        return event
    }

    // MARK: - Queries

    /// Every calendar event, newest first.
    public func list(limit: Int = 1000) async throws -> [CalendarEvent] {
        let rows = try await dataLayer.listByEntityType(
            entityType: CalendarEvent.entityType,
            limit: limit
        )
        return rows.compactMap { try? Self.eventFromEntity($0) }
    }

    /// Events with at least one occurrence in the range
    /// (recurring events are expanded; exDates are
    /// skipped). Sorted by first occurrence.
    public func events(in range: ClosedRange<Date>, calendar: Calendar = .current) async throws -> [CalendarEvent] {
        let all = try await list()
        return all
            .filter { !$0.occurrences(in: range, calendar: calendar).isEmpty }
            .sorted { lhs, rhs in
                let l = lhs.occurrences(in: range, calendar: calendar).first ?? lhs.startAt
                let r = rhs.occurrences(in: range, calendar: calendar).first ?? rhs.startAt
                return l < r
            }
    }

    /// Find events whose title matches `query`
    /// (case-insensitive substring). Used by the chat
    /// panel's "move the Q3 review" resolution.
    public func search(matching query: String, limit: Int = 20) async throws -> [CalendarEvent] {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !q.isEmpty else { return [] }
        let all = try await list()
        return Array(
            all
                .filter { $0.title.lowercased().contains(q) }
                .sorted { $0.startAt < $1.startAt }
                .prefix(limit)
        )
    }

    // MARK: - Linking

    /// Sync the event's cross-surface edges into
    /// `entity_links`. Idempotent: the data layer's unique
    /// constraint no-ops duplicates. Edges are append-only
    /// in v1 (there is no facade deleteLink); removing an
    /// attendee in the editor drops it from the event body
    /// but the historical edge stays — the receipt chain
    /// explains why.
    public func syncLinks(for event: CalendarEvent) async throws {
        for attendee in event.attendees {
            guard let contactID = attendee.contactID else { continue }
            try await linkEvent(event.id, to: contactID, linkType: CalendarLinkType.attendeeOf.rawValue)
        }
        for documentID in event.linkedDocumentIDs {
            try await linkEvent(event.id, to: documentID, linkType: CalendarLinkType.prepDocument.rawValue)
        }
        for taskID in event.linkedTaskIDs {
            try await linkEvent(event.id, to: taskID, linkType: CalendarLinkType.prepTask.rawValue)
        }
        for reminderID in event.reminders {
            try await linkEvent(event.id, to: reminderID, linkType: CalendarLinkType.reminderFor.rawValue)
        }
    }

    /// Link an event to another graph entity + emit a
    /// receipt. The link is the load-bearing artifact for
    /// the graph view; the receipt is the audit artifact.
    @discardableResult
    public func linkEvent(
        _ eventID: UUID,
        to otherEntityID: UUID,
        linkType: String,
        weight: Float = 1.0
    ) async throws -> EntityLink {
        let link = try await dataLayer.linkEntities(
            sourceID: eventID,
            targetID: otherEntityID,
            linkType: linkType,
            weight: weight
        )
        try await appendReceipt(
            entityID: eventID,
            receiptType: CalendarEventReceiptType.linkCreated.rawValue,
            payload: [
                "targetEntityID": .string(otherEntityID.uuidString),
                "linkType": .string(linkType),
                "weight": .number(Double(weight)),
            ]
        )
        return link
    }

    /// The event's outgoing edges (attendees, prep docs,
    /// prep tasks, reminders). The detail view uses these
    /// to render the "linked" section.
    public func links(for eventID: UUID) async throws -> [EntityLink] {
        try await dataLayer.outLinks(sourceID: eventID)
    }

    // MARK: - Receipts

    /// All constitutional receipts for an event, oldest
    /// first. This is the event's full history: created ->
    /// updated* -> responded* -> deleted.
    public func receipts(forEvent eventID: UUID) async throws -> [GraphReceipt] {
        try await dataLayer.receipts(forEntity: eventID)
    }

    // MARK: - Helpers

    /// Upsert the entity row WITHOUT emitting a receipt.
    /// Used by ``respond(to:attendeeIndex:attendeeName:status:)``
    /// which emits its own `event_responded` receipt (a
    /// response is not an edit; the drawer should show one
    /// receipt for it, not two).
    private func upsertEntityRow(_ event: CalendarEvent) async throws -> CalendarEvent {
        let body = try event.jsonDataString()
        _ = try await dataLayer.upsertEntity(
            GraphEntityUpsert(
                id: event.id,
                entityType: CalendarEvent.entityType,
                subtype: event.allDay ? "all_day" : "timed",
                label: event.title,
                body: body,
                sourceURL: nil,
                embedding: nil
            )
        )
        return event
    }

    private func appendReceipt(
        entityID: UUID,
        receiptType: String,
        payload: [String: JSONValue]
    ) async throws {
        _ = try await dataLayer.appendReceipt(
            entityID: entityID,
            receiptType: receiptType,
            payload: payload
        )
    }

    private static func eventFromEntity(_ entity: GraphEntity) throws -> CalendarEvent? {
        guard entity.entityType == CalendarEvent.entityType else { return nil }
        guard let body = entity.body, !body.isEmpty else { return nil }
        return try CalendarEvent.from(jsonDataString: body)
    }

    private static let iso8601: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        return f
    }()
}

// MARK: - Receipt type taxonomy

/// The receipt types ``CalendarStore`` emits. The string
/// values are what show up in
/// `graph_receipts.receipt_type`; the enum keeps the call
/// sites self-documenting and pins the vocabulary for the
/// tests (changing a raw value is a schema event, the same
/// way the contact surface pins its strings).
public enum CalendarEventReceiptType: String, Codable, Sendable, CaseIterable {
    /// An event was added.
    case eventCreated = "event_created"
    /// An event was edited (time, title, attendees, ...).
    case eventUpdated = "event_updated"
    /// An event was deleted. The receipt outlives the row
    /// (voids it: the chain records what was removed and
    /// when).
    case eventDeleted = "event_deleted"
    /// The user responded to an event (accept / decline /
    /// tentative).
    case eventResponded = "event_responded"
    /// A cross-surface link was created (attendee, prep
    /// doc, prep task, reminder).
    case linkCreated = "event_link_created"
}

// MARK: - Link type taxonomy

/// The `entity_links.link_type` values the calendar
/// surface creates. Pinned for the graph view's edge
/// coloring and the tests.
public enum CalendarLinkType: String, Codable, Sendable, CaseIterable {
    /// Event -> contact (an attendee). Matches the link
    /// type named in the productivity spec §12.8.
    case attendeeOf = "attendee_of"
    /// Event -> document (prep doc / agenda).
    case prepDocument = "prep_document"
    /// Event -> task (prep task).
    case prepTask = "prep_task"
    /// Event -> reminder.
    case reminderFor = "reminder_for"
}

// MARK: - Errors

public enum CalendarStoreError: Error, Sendable, Equatable {
    case eventNotFound(id: UUID)
    case invalidEventBody(reason: String)
    case invalidEvent(reason: String)
    case noAttendees(eventID: UUID)
}
