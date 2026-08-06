import XCTest
@testable import TesseraCore

/// Tests for the ``Reminder`` value type: JSON round-trip,
/// offset math, filter state derivation, display formatting,
/// and the receipt type taxonomy.
///
/// The store-level flows (upsert / fetch / receipts) need a
/// real Postgres + Valkey to exercise and live in
/// ``ReminderStoreIntegrationTests`` (env-gated on
/// `TESSERA_DB_INTEGRATION=1`).
final class ReminderTests: XCTestCase {

    // MARK: - Receipt type strings (schema migration pins)

    func testReceiptTypesAreStable() {
        // The string values are persisted to
        // graph_receipts.receipt_type; changing them is
        // a schema migration. Pin them here.
        XCTAssertEqual(ReminderReceiptType.created.rawValue, "reminder_created")
        XCTAssertEqual(ReminderReceiptType.updated.rawValue, "reminder_updated")
        XCTAssertEqual(ReminderReceiptType.acknowledged.rawValue, "reminder_acknowledged")
        XCTAssertEqual(ReminderReceiptType.snoozed.rawValue, "reminder_snoozed")
        XCTAssertEqual(ReminderReceiptType.deleted.rawValue, "reminder_deleted")
        XCTAssertEqual(ReminderReceiptType.linkCreated.rawValue, "reminder_link_created")
        XCTAssertEqual(ReminderReceiptType.linkDeleted.rawValue, "reminder_link_deleted")
        XCTAssertEqual(Reminder.entityType, "reminder")
    }

    // MARK: - Store error equality

    func testStoreErrorEquality() {
        let id = UUID()
        XCTAssertEqual(
            ReminderStoreError.reminderNotFound(id: id),
            ReminderStoreError.reminderNotFound(id: id)
        )
        XCTAssertNotEqual(
            ReminderStoreError.reminderNotFound(id: id),
            ReminderStoreError.reminderNotFound(id: UUID())
        )
        XCTAssertEqual(
            ReminderStoreError.invalidReminderBody(reason: "x"),
            ReminderStoreError.invalidReminderBody(reason: "x")
        )
        XCTAssertNotEqual(
            ReminderStoreError.invalidReminderBody(reason: "x"),
            ReminderStoreError.invalidReminderBody(reason: "y")
        )
    }

    // MARK: - Notification error equality

    func testNotificationErrorEquality() {
        let id = UUID()
        let now = Date()
        XCTAssertEqual(
            ReminderNotificationError.triggerInPast(at: now, reminderID: id),
            ReminderNotificationError.triggerInPast(at: now, reminderID: id)
        )
        XCTAssertNotEqual(
            ReminderNotificationError.triggerInPast(at: now, reminderID: id),
            ReminderNotificationError.triggerInPast(at: now.addingTimeInterval(1), reminderID: id)
        )
        XCTAssertEqual(
            ReminderNotificationError.notificationCenterFailed(reason: "x"),
            ReminderNotificationError.notificationCenterFailed(reason: "x")
        )
        XCTAssertNotEqual(
            ReminderNotificationError.triggerInPast(at: now, reminderID: id),
            ReminderNotificationError.notificationCenterFailed(reason: "x")
        )
    }

    // MARK: - JSON round-trip

    func testReminderJSONStringRoundTrip() throws {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let later = now.addingTimeInterval(60 * 30) // 30 min later
        let r = Reminder(
            id: UUID(),
            title: "Q3 review meeting",
            notes: "Prep slides",
            calendarEventID: UUID(),
            offsetMinutes: -15,
            triggerAt: later,
            priority: .high,
            createdAt: now,
            updatedAt: now
        )
        let body = try r.jsonDataString()
        XCTAssertTrue(body.contains("Q3 review meeting"))
        XCTAssertTrue(body.contains("Prep slides"))
        let parsed = try Reminder.from(jsonDataString: body)
        XCTAssertEqual(parsed, r)
    }

    func testReminderJSONHandlesOptionals() throws {
        let now = Date(timeIntervalSince1970: 2_000_000)
        let r = Reminder(
            id: UUID(),
            title: "Bare",
            notes: "",
            calendarEventID: UUID(),
            offsetMinutes: 0,
            triggerAt: now,
            acknowledgedAt: nil,
            snoozedUntil: nil,
            priority: .none,
            createdAt: now,
            updatedAt: now
        )
        let body = try r.jsonDataString()
        let parsed = try Reminder.from(jsonDataString: body)
        XCTAssertNil(parsed.acknowledgedAt)
        XCTAssertNil(parsed.snoozedUntil)
        XCTAssertEqual(parsed.priority, .none)
    }

    func testInvalidJSONRejected() {
        let bad = "not json at all"
        XCTAssertThrowsError(try Reminder.from(jsonDataString: bad)) { _ in
            // expected
        }
    }

    // MARK: - Offset math

    func testFormatOffsetBeforeMinutes() {
        XCTAssertEqual(Reminder.formatOffset(-15), "15 min before")
        XCTAssertEqual(Reminder.formatOffset(-1), "1 min before")
        XCTAssertEqual(Reminder.formatOffset(-45), "45 min before")
    }

    func testFormatOffsetAfterMinutes() {
        XCTAssertEqual(Reminder.formatOffset(15), "15 min after")
        XCTAssertEqual(Reminder.formatOffset(1), "1 min after")
    }

    func testFormatOffsetHours() {
        XCTAssertEqual(Reminder.formatOffset(-60), "1 hour before")
        XCTAssertEqual(Reminder.formatOffset(-120), "2 hours before")
        XCTAssertEqual(Reminder.formatOffset(60), "1 hour after")
        XCTAssertEqual(Reminder.formatOffset(180), "3 hours after")
    }

    func testFormatOffsetZero() {
        XCTAssertEqual(Reminder.formatOffset(0), "at start")
    }

    // MARK: - triggerAt computation

    func testTriggerTimeOffset() {
        let eventStart = Date(timeIntervalSince1970: 1_000_000)
        let triggerMinus15 = ReminderStore.triggerTime(
            calendarEventStart: eventStart,
            offsetMinutes: -15
        )
        XCTAssertEqual(
            triggerMinus15,
            eventStart.addingTimeInterval(-15 * 60)
        )
        let triggerPlus60 = ReminderStore.triggerTime(
            calendarEventStart: eventStart,
            offsetMinutes: 60
        )
        XCTAssertEqual(
            triggerPlus60,
            eventStart.addingTimeInterval(60 * 60)
        )
    }

    // MARK: - State accessors

    func testIsUpcoming() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let future = now.addingTimeInterval(3600)
        let past = now.addingTimeInterval(-3600)
        XCTAssertTrue(Reminder(title: "a", calendarEventID: UUID(), offsetMinutes: 0, triggerAt: future).isUpcoming(now: now))
        XCTAssertFalse(Reminder(title: "a", calendarEventID: UUID(), offsetMinutes: 0, triggerAt: past).isUpcoming(now: now))
        var ack = Reminder(title: "a", calendarEventID: UUID(), offsetMinutes: 0, triggerAt: future)
        ack.acknowledgedAt = now
        XCTAssertFalse(ack.isUpcoming(now: now))
        var snoozed = Reminder(title: "a", calendarEventID: UUID(), offsetMinutes: 0, triggerAt: future)
        snoozed.snoozedUntil = now.addingTimeInterval(120)
        XCTAssertFalse(snoozed.isUpcoming(now: now))
    }

    func testIsSnoozed() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        var r = Reminder(title: "a", calendarEventID: UUID(), offsetMinutes: 0, triggerAt: now)
        XCTAssertFalse(r.isSnoozed(now: now))
        r.snoozedUntil = now.addingTimeInterval(120)
        XCTAssertTrue(r.isSnoozed(now: now))
        r.snoozedUntil = now.addingTimeInterval(-120) // past snooze
        XCTAssertFalse(r.isSnoozed(now: now))
    }

    func testIsAcknowledged() {
        var r = Reminder(title: "a", calendarEventID: UUID(), offsetMinutes: 0, triggerAt: Date())
        XCTAssertFalse(r.isAcknowledged())
        r.acknowledgedAt = Date()
        XCTAssertTrue(r.isAcknowledged())
    }

    // MARK: - Display

    func testDisplayLine() {
        let r = Reminder(
            title: "Q3 review",
            calendarEventID: UUID(),
            offsetMinutes: -15,
            triggerAt: Date()
        )
        XCTAssertEqual(
            r.displayLine(calendarEventTitle: "Q3 review meeting"),
            "15 min before Q3 review meeting"
        )
        XCTAssertEqual(
            r.displayLine(calendarEventTitle: nil),
            "15 min before event"
        )
    }

    // MARK: - Priority ordering

    func testPriorityOrdering() {
        // Higher priority sorts FIRST (`.high` is "less"
        // than `.none` in Comparable terms, so it appears
        // first in the sorted list).
        XCTAssertTrue(TesseraTaskPriority.high < .medium)
        XCTAssertTrue(TesseraTaskPriority.medium < .low)
        XCTAssertTrue(TesseraTaskPriority.low < .none)
        XCTAssertEqual(
            [TesseraTaskPriority.none, .high, .low, .medium].sorted(),
            [.high, .medium, .low, .none]
        )
    }

    func testPriorityShortLabel() {
        XCTAssertEqual(TesseraTaskPriority.none.shortLabel, "")
        XCTAssertEqual(TesseraTaskPriority.low.shortLabel, "·")
        XCTAssertEqual(TesseraTaskPriority.medium.shortLabel, "•")
        XCTAssertEqual(TesseraTaskPriority.high.shortLabel, "!")
    }

    // MARK: - Equatable / Hashable

    func testEqualityIsValueBased() throws {
        let id = UUID()
        let now = Date(timeIntervalSince1970: 1_000_000)
        let a = Reminder(
            id: id, title: "t", notes: "n",
            calendarEventID: UUID(),
            offsetMinutes: -15, triggerAt: now,
            createdAt: now, updatedAt: now
        )
        let b = Reminder(
            id: id, title: "t", notes: "n",
            calendarEventID: a.calendarEventID,
            offsetMinutes: -15, triggerAt: now,
            createdAt: now, updatedAt: now
        )
        XCTAssertEqual(a, b)
        var c = a
        c.title = "other"
        XCTAssertNotEqual(a, c)
    }

    func testHashableValueBased() {
        // Two reminders with identical fields hash to the
        // same bucket and compare equal — Set<Reminder>
        // lookups work as the user expects.
        let id = UUID()
        let date = Date(timeIntervalSince1970: 1_000_000)
        let a = Reminder(
            id: id, title: "t", notes: "",
            calendarEventID: UUID(),
            offsetMinutes: 0, triggerAt: date,
            createdAt: date, updatedAt: date
        )
        let b = Reminder(
            id: id, title: "t", notes: "",
            calendarEventID: a.calendarEventID,
            offsetMinutes: 0, triggerAt: date,
            createdAt: date, updatedAt: date
        )
        XCTAssertEqual(a, b)
        XCTAssertEqual(a.hashValue, b.hashValue)
        let set: Set<Reminder> = [a]
        XCTAssertTrue(set.contains(b))
    }
}
