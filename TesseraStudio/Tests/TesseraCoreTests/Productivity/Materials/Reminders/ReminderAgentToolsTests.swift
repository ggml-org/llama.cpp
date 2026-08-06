import XCTest
@testable import TesseraCore

/// Tests for the chat panel's reminder tools
/// (``ReminderCreateTool``, ``ReminderListTool``,
/// ``ReminderDismissTool``, ``ReminderSnoozeTool``). Uses
/// an in-memory ``ReminderStoring`` mock so the tests don't
/// need Postgres / Valkey. The end-to-end flow against a
/// real data layer lives in
/// ``ReminderStoreIntegrationTests`` (env-gated).
final class ReminderAgentToolsTests: XCTestCase {

    // MARK: - Mock store

    /// In-memory ``ReminderStoring`` for tool tests. Mimics
    /// the production store's receipt + idempotency
    /// semantics just enough to exercise the tool logic.
    actor InMemoryReminderStore: ReminderStoring {
        var reminders: [UUID: Reminder] = [:]
        var receipts: [UUID: [GraphReceipt]] = [:]
        private var nextReceiptID = 0

        func upsert(_ reminder: Reminder) async throws -> Reminder {
            reminders[reminder.id] = reminder
            return reminder
        }

        func get(id: UUID) async throws -> Reminder? {
            reminders[id]
        }

        func list(limit: Int) async throws -> [Reminder] {
            Array(reminders.values).sorted { $0.triggerAt < $1.triggerAt }.prefix(limit).map { $0 }
        }

        func listForCalendarEvent(_ eventID: UUID, limit: Int) async throws -> [Reminder] {
            Array(reminders.values.filter { $0.calendarEventID == eventID })
                .sorted { $0.triggerAt < $1.triggerAt }
                .prefix(limit).map { $0 }
        }

        func acknowledge(id: UUID, at now: Date) async throws -> Reminder? {
            guard var r = reminders[id] else { return nil }
            r.acknowledgedAt = now
            r.snoozedUntil = nil
            r.updatedAt = now
            reminders[id] = r
            await appendReceipt(entityID: id, type: "reminder_acknowledged")
            return r
        }

        func snooze(id: UUID, until: Date, at now: Date) async throws -> Reminder? {
            guard var r = reminders[id] else { return nil }
            r.snoozedUntil = until
            r.updatedAt = now
            reminders[id] = r
            await appendReceipt(entityID: id, type: "reminder_snoozed")
            return r
        }

        func delete(id: UUID) async throws -> Bool {
            let existed = reminders[id] != nil
            reminders[id] = nil
            return existed
        }

        func receipts(forReminder reminderID: UUID) async throws -> [GraphReceipt] {
            receipts[reminderID] ?? []
        }

        private func appendReceipt(entityID: UUID, type: String) {
            nextReceiptID += 1
            let r = GraphReceipt(
                id: UUID(),
                entityID: entityID,
                receiptType: type,
                payload: [:],
                signature: nil,
                witnessedAt: Date()
            )
            receipts[entityID, default: []].append(r)
            _ = nextReceiptID
        }
    }

    // MARK: - Create

    func testCreateHappyPath() async throws {
        let store = InMemoryReminderStore()
        let tool = ReminderCreateTool(store: store)
        let eventID = UUID()
        let trigger = Date().addingTimeInterval(3600)
        let result = try await tool.execute(arguments: [
            "title": .string("15 min before Q3 review"),
            "calendar_event_id": .string(eventID.uuidString),
            "offset_minutes": .number(-15),
            "trigger_at": .string(Self.iso(trigger)),
        ])
        XCTAssertTrue(result.success)
        XCTAssertNotNil(result.data?["reminder_id"])
        let rid = result.data?["reminder_id"]?.stringValue
        XCTAssertNotNil(rid)
        let saved = try await store.get(id: UUID(uuidString: rid!)!)
        XCTAssertEqual(saved?.title, "15 min before Q3 review")
        XCTAssertEqual(saved?.calendarEventID, eventID)
        XCTAssertEqual(saved?.offsetMinutes, -15)
    }

    func testCreateMissingTitle() async throws {
        let tool = ReminderCreateTool(store: InMemoryReminderStore())
        let r = try await tool.execute(arguments: [
            "calendar_event_id": .string(UUID().uuidString),
            "offset_minutes": .number(-15),
            "trigger_at": .string(Self.iso(Date())),
        ])
        XCTAssertFalse(r.success)
        XCTAssertTrue(r.error?.contains("title") ?? false)
    }

    func testCreateInvalidEventID() async throws {
        let tool = ReminderCreateTool(store: InMemoryReminderStore())
        let r = try await tool.execute(arguments: [
            "title": .string("t"),
            "calendar_event_id": .string("not-a-uuid"),
            "offset_minutes": .number(-15),
            "trigger_at": .string(Self.iso(Date())),
        ])
        XCTAssertFalse(r.success)
        XCTAssertTrue(r.error?.contains("UUID") ?? false)
    }

    func testCreateInvalidTrigger() async throws {
        let tool = ReminderCreateTool(store: InMemoryReminderStore())
        let r = try await tool.execute(arguments: [
            "title": .string("t"),
            "calendar_event_id": .string(UUID().uuidString),
            "offset_minutes": .number(-15),
            "trigger_at": .string("not a date"),
        ])
        XCTAssertFalse(r.success)
        XCTAssertTrue(r.error?.contains("ISO") ?? false)
    }

    func testCreatePriorityDefaultsToNone() async throws {
        let store = InMemoryReminderStore()
        let tool = ReminderCreateTool(store: store)
        let r = try await tool.execute(arguments: [
            "title": .string("t"),
            "calendar_event_id": .string(UUID().uuidString),
            "offset_minutes": .number(-15),
            "trigger_at": .string(Self.iso(Date())),
        ])
        XCTAssertTrue(r.success)
        let rid = UUID(uuidString: r.data?["reminder_id"]?.stringValue ?? "")!
        let saved = try await store.get(id: rid)
        XCTAssertNotNil(saved)
        XCTAssertEqual(saved?.priority, TesseraTaskPriority.none)
    }

    // MARK: - List

    func testListEmpty() async throws {
        let tool = ReminderListTool(store: InMemoryReminderStore())
        let r = try await tool.execute(arguments: [:])
        XCTAssertTrue(r.success)
        XCTAssertEqual(r.data?["count"]?.numberValue, 0)
        XCTAssertTrue(r.output.contains("No reminders"))
    }

    func testListRendersTable() async throws {
        let store = InMemoryReminderStore()
        let tool = ReminderCreateTool(store: store)
        let r = try await tool.execute(arguments: [
            "title": .string("Q3 review"),
            "calendar_event_id": .string(UUID().uuidString),
            "offset_minutes": .number(-15),
            "trigger_at": .string(Self.iso(Date().addingTimeInterval(900))),
        ])
        XCTAssertTrue(r.success)
        let listTool = ReminderListTool(store: store)
        let out = try await listTool.execute(arguments: ["filter": .string("all")])
        XCTAssertTrue(out.output.contains("Q3 review"))
        XCTAssertEqual(out.data?["count"]?.numberValue, 1)
    }

    func testListFilterScopedToEvent() async throws {
        let store = InMemoryReminderStore()
        let eventA = UUID()
        let eventB = UUID()
        let tool = ReminderCreateTool(store: store)
        // Use a future triggerAt so the default "upcoming"
        // filter doesn't drop them on the list call.
        let future = Date().addingTimeInterval(3600)
        _ = try await tool.execute(arguments: [
            "title": .string("A"),
            "calendar_event_id": .string(eventA.uuidString),
            "offset_minutes": .number(-15),
            "trigger_at": .string(Self.iso(future)),
        ])
        _ = try await tool.execute(arguments: [
            "title": .string("B"),
            "calendar_event_id": .string(eventB.uuidString),
            "offset_minutes": .number(-15),
            "trigger_at": .string(Self.iso(future.addingTimeInterval(60))),
        ])
        let listTool = ReminderListTool(store: store)
        let out = try await listTool.execute(arguments: [
            "calendar_event_id": .string(eventA.uuidString),
        ])
        XCTAssertTrue(out.output.contains("A"))
        XCTAssertFalse(out.output.contains("B"))
    }

    // MARK: - Dismiss

    func testDismissHappyPath() async throws {
        let store = InMemoryReminderStore()
        let c = ReminderCreateTool(store: store)
        let created = try await c.execute(arguments: [
            "title": .string("Q3 review"),
            "calendar_event_id": .string(UUID().uuidString),
            "offset_minutes": .number(-15),
            "trigger_at": .string(Self.iso(Date().addingTimeInterval(900))),
        ])
        let rid = UUID(uuidString: created.data?["reminder_id"]?.stringValue ?? "")!
        let d = ReminderDismissTool(store: store)
        let r = try await d.execute(arguments: [
            "reminder_id": .string(rid.uuidString),
        ])
        XCTAssertTrue(r.success)
        let saved = try await store.get(id: rid)
        XCTAssertNotNil(saved?.acknowledgedAt)
    }

    func testDismissNotFound() async throws {
        let tool = ReminderDismissTool(store: InMemoryReminderStore())
        let r = try await tool.execute(arguments: [
            "reminder_id": .string(UUID().uuidString),
        ])
        XCTAssertFalse(r.success)
        XCTAssertTrue(r.error?.contains("not found") ?? false)
    }

    // MARK: - Snooze

    func testSnoozeHappyPath() async throws {
        let store = InMemoryReminderStore()
        let c = ReminderCreateTool(store: store)
        let created = try await c.execute(arguments: [
            "title": .string("Q3 review"),
            "calendar_event_id": .string(UUID().uuidString),
            "offset_minutes": .number(-15),
            "trigger_at": .string(Self.iso(Date().addingTimeInterval(900))),
        ])
        let rid = UUID(uuidString: created.data?["reminder_id"]?.stringValue ?? "")!
        let s = ReminderSnoozeTool(store: store)
        let r = try await s.execute(arguments: [
            "reminder_id": .string(rid.uuidString),
            "snooze_minutes": .number(10),
        ])
        XCTAssertTrue(r.success)
        let saved = try await store.get(id: rid)
        XCTAssertNotNil(saved?.snoozedUntil)
    }

    func testSnoozeRejectsOverOneDay() async throws {
        let store = InMemoryReminderStore()
        let c = ReminderCreateTool(store: store)
        let created = try await c.execute(arguments: [
            "title": .string("Q3 review"),
            "calendar_event_id": .string(UUID().uuidString),
            "offset_minutes": .number(-15),
            "trigger_at": .string(Self.iso(Date().addingTimeInterval(900))),
        ])
        let rid = UUID(uuidString: created.data?["reminder_id"]?.stringValue ?? "")!
        let s = ReminderSnoozeTool(store: store)
        let r = try await s.execute(arguments: [
            "reminder_id": .string(rid.uuidString),
            "snooze_minutes": .number(2000),
        ])
        XCTAssertFalse(r.success)
        XCTAssertTrue(r.error?.contains("1440") ?? false)
    }

    // MARK: - Helpers

    private static func iso(_ d: Date) -> String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f.string(from: d)
    }
}
