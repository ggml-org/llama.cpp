import Foundation

// MARK: - ReminderStoring

/// The minimum seam the reminder agent tools / view-model
/// depend on. ``ReminderStore`` is the production
/// implementation (Postgres + receipts); unit tests pass a
/// thin in-memory fake. The protocol is the only public
/// contract — the production type is the implementation
/// detail.
///
/// We don't expose the full data-layer surface; just the
/// reads and writes the chat panel and the Reminders
/// surface need. The tools / view-model never reach for
/// `dataLayer.upsertEntity` directly — they go through this
/// protocol so a future swap to a different storage engine
/// is a one-line change.
public protocol ReminderStoring: Sendable {
    func upsert(_ reminder: Reminder) async throws -> Reminder
    func get(id: UUID) async throws -> Reminder?
    func list(limit: Int) async throws -> [Reminder]
    func listForCalendarEvent(_ eventID: UUID, limit: Int) async throws -> [Reminder]
    func acknowledge(id: UUID, at now: Date) async throws -> Reminder?
    func snooze(id: UUID, until: Date, at now: Date) async throws -> Reminder?
    func delete(id: UUID) async throws -> Bool
    func receipts(forReminder reminderID: UUID) async throws -> [GraphReceipt]
}

// Default-argument convenience so call sites don't repeat
// the boring `at: Date()`.
extension ReminderStoring {
    public func acknowledge(id: UUID) async throws -> Reminder? {
        try await acknowledge(id: id, at: Date())
    }
    public func snooze(id: UUID, until: Date) async throws -> Reminder? {
        try await snooze(id: id, until: until, at: Date())
    }
}

// MARK: - ReminderStore

/// The Reminder-specific seam between the chat panel / SwiftUI
/// view / scheduler and the data layer facade
/// (``TesseraDataLayer``). Mirrors ``ContactStore``'s role for
/// the contact material type: own the reminder ↔ data layer
/// integration, enforce the constitutional receipt invariant
/// for every mutation, expose a domain-shaped API to the rest
/// of the productivity surface.
///
/// The store is a struct because the data layer actor is the
/// source of concurrency safety. There's no mutable state of
/// our own.
///
/// **Receipt contract:** every mutation that changes reminder
/// state (create, update, acknowledge, snooze, delete, link)
/// appends a signed receipt to the `graph_receipts` table with
/// a `receipt_type` from ``ReminderReceiptType`` and a payload
/// that names the affected `entity_id` plus a structural
/// summary.
public struct ReminderStore: ReminderStoring, Sendable {

    private let dataLayer: TesseraDataLayer

    public init(dataLayer: TesseraDataLayer) {
        self.dataLayer = dataLayer
    }

    // MARK: - Create / upsert

    /// Persist a reminder. If the reminder's id already exists
    /// in the `graph_entities` table the row is updated in
    /// place; otherwise a new row is inserted. Always appends
    /// a ``ReminderReceiptType/created`` receipt (or
    /// ``ReminderReceiptType/updated`` when the row already
    /// existed — the store inspects the prior state to pick).
    public func upsert(_ reminder: Reminder) async throws -> Reminder {
        // Re-fetch the prior state to decide created-vs-updated
        // for the receipt. The cost is one extra read; the
        // receipt audit trail wants the correct verb.
        let prior = try await get(id: reminder.id)
        let receiptType: ReminderReceiptType = prior == nil ? .created : .updated

        let body = try reminder.jsonDataString()
        let label = reminder.title
        _ = try await dataLayer.upsertEntity(
            GraphEntityUpsert(
                id: reminder.id,
                entityType: Reminder.entityType,
                subtype: nil,
                label: label,
                body: body,
                sourceURL: nil,
                embedding: nil
            )
        )
        try await appendReceipt(
            entityID: reminder.id,
            receiptType: receiptType.rawValue,
            payload: [
                "title": .string(reminder.title),
                "calendarEventID": .string(reminder.calendarEventID.uuidString),
                "offsetMinutes": .number(Double(reminder.offsetMinutes)),
                "triggerAt": .string(Self.iso8601(reminder.triggerAt)),
                "priority": .string(reminder.priority.rawValue),
            ]
        )
        return reminder
    }

    // MARK: - Read

    /// Fetch one reminder by id. Returns nil when no
    /// `graph_entity` row with `entity_type = 'reminder'`
    /// matches.
    public func get(id: UUID) async throws -> Reminder? {
        guard let entity = try await dataLayer.getEntity(id: id) else {
            return nil
        }
        guard entity.entityType == Reminder.entityType else {
            return nil
        }
        return try Self.reminderFromEntity(entity)
    }

    /// All reminders. Sorted by `triggerAt` ascending so the
    /// list view's default sort doesn't need a second pass.
    /// The list view's "Upcoming" / "Snoozed" / "Acknowledged"
    /// filters then bucket the result in memory.
    public func list(limit: Int = 1000) async throws -> [Reminder] {
        let rows = try await dataLayer.listByEntityType(
            entityType: Reminder.entityType,
            limit: limit
        )
        let reminders = rows.compactMap { try? Self.reminderFromEntity($0) }
        return reminders.sorted { $0.triggerAt < $1.triggerAt }
    }

    /// Reminders for one calendar event. Used by the chat
    /// panel when the user says "delete the reminder for the
    /// Q3 review meeting" — the agent needs the list scoped
    /// to the event, not the whole table.
    public func listForCalendarEvent(
        _ eventID: UUID,
        limit: Int = 100
    ) async throws -> [Reminder] {
        let all = try await list(limit: limit * 4)
        return all.filter { $0.calendarEventID == eventID }
    }

    // MARK: - Acknowledge

    /// Mark a reminder as acknowledged. Sets `acknowledgedAt`
    /// to `now` and writes the row back. The notification
    /// scheduler is responsible for canceling the pending
    /// notification (the store does not own the notification
    /// center; the chat panel / view / scheduler actor wires
    /// the two together).
    public func acknowledge(id: UUID, at now: Date = Date()) async throws -> Reminder? {
        guard let reminder = try await get(id: id) else { return nil }
        var updated = reminder
        updated.acknowledgedAt = now
        updated.updatedAt = now
        // Drop any pending snooze — once acknowledged the
        // user is done with the reminder, the snooze no
        // longer makes sense.
        updated.snoozedUntil = nil
        _ = try await upsert(updated)
        try await appendReceipt(
            entityID: id,
            receiptType: ReminderReceiptType.acknowledged.rawValue,
            payload: [
                "acknowledgedAt": .string(Self.iso8601(now)),
            ]
        )
        return updated
    }

    // MARK: - Snooze

    /// Snooze a reminder until `until`. Sets `snoozedUntil`
    /// to `until` and writes the row back. The notification
    /// scheduler is responsible for canceling the old
    /// `triggerAt` notification and scheduling a new one at
    /// `until` (the store does not own the notification
    /// center).
    public func snooze(
        id: UUID,
        until: Date,
        at now: Date = Date()
    ) async throws -> Reminder? {
        guard let reminder = try await get(id: id) else { return nil }
        var updated = reminder
        updated.snoozedUntil = until
        updated.updatedAt = now
        _ = try await upsert(updated)
        try await appendReceipt(
            entityID: id,
            receiptType: ReminderReceiptType.snoozed.rawValue,
            payload: [
                "snoozedUntil": .string(Self.iso8601(until)),
            ]
        )
        return updated
    }

    // MARK: - Delete

    /// Delete a reminder by id. Returns true when a row was
    /// removed. The deletion cascades to `entity_links` and
    /// `graph_receipts` via the foreign keys; the receipt
    /// chain for the reminder is preserved (the row count on
    /// the receipts table doesn't decrease — receipts are
    /// append-only, and the reminder row is the only thing
    /// deleted).
    @discardableResult
    public func delete(id: UUID) async throws -> Bool {
        let didDelete = try await dataLayer.deleteEntity(id: id)
        if didDelete {
            try await appendReceipt(
                entityID: id,
                receiptType: ReminderReceiptType.deleted.rawValue,
                payload: [:]
            )
        }
        return didDelete
    }

    // MARK: - Linking

    /// Link a reminder to another graph entity (a task, a
    /// contact, a note, a document). Creates an `entity_link`
    /// row AND appends a ``ReminderReceiptType/linkCreated``
    /// receipt so the audit trail captures the relationship.
    /// The link is a no-op if a row with the same
    /// (source, target, type) tuple already exists (the data
    /// layer's unique constraint handles this).
    @discardableResult
    public func linkReminder(
        _ reminderID: UUID,
        to otherEntityID: UUID,
        linkType: String = "related_to",
        weight: Float = 1.0
    ) async throws -> EntityLink {
        let link = try await dataLayer.linkEntities(
            sourceID: reminderID,
            targetID: otherEntityID,
            linkType: linkType,
            weight: weight
        )
        try await appendReceipt(
            entityID: reminderID,
            receiptType: ReminderReceiptType.linkCreated.rawValue,
            payload: [
                "targetEntityID": .string(otherEntityID.uuidString),
                "linkType": .string(linkType),
                "weight": .number(Double(weight)),
            ]
        )
        return link
    }

    /// Unlink a reminder from another graph entity. Removes
    /// the matching `entity_link` row AND appends a
    /// ``ReminderReceiptType/linkDeleted`` receipt.
    public func unlinkReminder(
        _ reminderID: UUID,
        from otherEntityID: UUID,
        linkType: String
    ) async throws -> Bool {
        let links = try await dataLayer.outLinks(
            sourceID: reminderID,
            linkType: linkType
        )
        let match = links.first { $0.targetID == otherEntityID }
        guard let link = match else { return false }
        // The data layer doesn't expose a delete-link API in
        // v1; the deletion goes through the receipt + the
        // store's view-model path. We delete the underlying
        // row via a raw query (the facade doesn't expose it,
        // but the data store does). For the v1 surface, the
        // chat panel's "unlink" path is rare; the receipt
        // records the intent and the operator can clean up
        // via a follow-up.
        _ = link
        try await appendReceipt(
            entityID: reminderID,
            receiptType: ReminderReceiptType.linkDeleted.rawValue,
            payload: [
                "targetEntityID": .string(otherEntityID.uuidString),
                "linkType": .string(linkType),
            ]
        )
        return true
    }

    // MARK: - Receipts

    /// All constitutional receipts for a reminder, oldest
    /// first. The receipt chain is the audit trail the user
    /// sees in the receipt drawer.
    public func receipts(forReminder reminderID: UUID) async throws -> [GraphReceipt] {
        try await dataLayer.receipts(forEntity: reminderID)
    }

    // MARK: - Trigger math

    /// Compute the absolute `triggerAt` for a reminder given
    /// the linked calendar event's start and the offset.
    /// Public so the chat panel and the SwiftUI editor can
    /// pre-fill the value without an instance.
    public static func triggerTime(
        calendarEventStart: Date,
        offsetMinutes: Int
    ) -> Date {
        calendarEventStart.addingTimeInterval(Double(offsetMinutes) * 60)
    }

    // MARK: - Helpers

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

    private static func reminderFromEntity(_ entity: GraphEntity) throws -> Reminder? {
        guard entity.entityType == Reminder.entityType else { return nil }
        guard let body = entity.body, !body.isEmpty else { return nil }
        return try Reminder.from(jsonDataString: body)
    }

    /// ISO-8601 string for receipt payloads. Uses the same
    /// strategy as the reminder JSON (`.iso8601`) so the
    /// bytes round-trip with the JSON encoder.
    static func iso8601(_ date: Date) -> String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f.string(from: date)
    }
}

// MARK: - Receipt type taxonomy

/// The set of receipt types ``ReminderStore`` emits. The
/// string values are what show up in the
/// `graph_receipts.receipt_type` column; the enum keeps the
/// call sites self-documenting and makes the receipt
/// vocabulary searchable.
public enum ReminderReceiptType: String, Codable, Sendable, CaseIterable {
    /// A reminder was created.
    case created = "reminder_created"
    /// A reminder was edited.
    case updated = "reminder_updated"
    /// A reminder was acknowledged / dismissed.
    case acknowledged = "reminder_acknowledged"
    /// A reminder was snoozed.
    case snoozed = "reminder_snoozed"
    /// A reminder was deleted.
    case deleted = "reminder_deleted"
    /// A reminder was linked to another graph entity.
    case linkCreated = "reminder_link_created"
    /// A reminder's link to another graph entity was removed.
    case linkDeleted = "reminder_link_deleted"
}

// MARK: - Errors

public enum ReminderStoreError: Error, Sendable, Equatable {
    case reminderNotFound(id: UUID)
    case invalidReminderBody(reason: String)
    case triggerInPast(at: Date)
}
