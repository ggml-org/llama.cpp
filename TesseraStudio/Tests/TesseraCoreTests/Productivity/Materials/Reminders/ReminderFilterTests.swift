import XCTest
@testable import TesseraCore

/// Tests for ``ReminderFilter`` and the list view-model
/// pure helpers (relative time, filtered result, badge
/// counts).
final class ReminderFilterTests: XCTestCase {

    private let now = Date(timeIntervalSince1970: 1_700_000_000)

    // MARK: - Filter cases

    func testFilterMetadata() {
        XCTAssertEqual(ReminderFilter.allCases.count, 4)
        XCTAssertEqual(ReminderFilter.upcoming.displayName, "Upcoming")
        XCTAssertEqual(ReminderFilter.acknowledged.displayName, "Acknowledged")
        XCTAssertEqual(ReminderFilter.snoozed.displayName, "Snoozed")
        XCTAssertEqual(ReminderFilter.all.displayName, "All")
        XCTAssertEqual(ReminderFilter.upcoming.systemImage, "bell")
        XCTAssertEqual(ReminderFilter.acknowledged.systemImage, "checkmark.circle")
        XCTAssertEqual(ReminderFilter.snoozed.systemImage, "moon.zzz")
        XCTAssertEqual(ReminderFilter.all.systemImage, "tray")
    }

    // MARK: - Bucketing

    private func makeReminder(
        trigger: Date,
        acknowledged: Date? = nil,
        snoozed: Date? = nil,
        title: String = "r"
    ) -> Reminder {
        Reminder(
            id: UUID(),
            title: title,
            calendarEventID: UUID(),
            offsetMinutes: 0,
            triggerAt: trigger,
            acknowledgedAt: acknowledged,
            snoozedUntil: snoozed,
            createdAt: now,
            updatedAt: now
        )
    }

    func testUpcomingKeepsOnlyFutureAndUnacknowledged() {
        let future = makeReminder(trigger: now.addingTimeInterval(3600), title: "future")
        let past = makeReminder(trigger: now.addingTimeInterval(-3600), title: "past")
        let ack = makeReminder(
            trigger: now.addingTimeInterval(3600),
            acknowledged: now,
            title: "ack"
        )
        let result = ReminderFilter.upcoming.apply(
            to: [future, past, ack], referenceDate: now
        )
        XCTAssertEqual(result.map(\.title), ["future"])
    }

    func testUpcomingIsSortedByTrigger() {
        let r1 = makeReminder(trigger: now.addingTimeInterval(7200), title: "later")
        let r2 = makeReminder(trigger: now.addingTimeInterval(60), title: "sooner")
        let r3 = makeReminder(trigger: now.addingTimeInterval(600), title: "middle")
        let result = ReminderFilter.upcoming.apply(
            to: [r1, r2, r3], referenceDate: now
        )
        XCTAssertEqual(result.map(\.title), ["sooner", "middle", "later"])
    }

    func testAcknowledgedFilter() {
        let ack1 = makeReminder(
            trigger: now.addingTimeInterval(-7200),
            acknowledged: now.addingTimeInterval(-100),
            title: "ack1"
        )
        let ack2 = makeReminder(
            trigger: now.addingTimeInterval(-3600),
            acknowledged: now.addingTimeInterval(-50),
            title: "ack2"
        )
        let result = ReminderFilter.acknowledged.apply(
            to: [ack1, ack2], referenceDate: now
        )
        // Sorted by acknowledgedAt descending — most recent ack first.
        XCTAssertEqual(result.map(\.title), ["ack2", "ack1"])
    }

    func testSnoozedFilter() {
        let snoozedFuture = makeReminder(
            trigger: now.addingTimeInterval(3600),
            snoozed: now.addingTimeInterval(1800),
            title: "snoozed"
        )
        let snoozedPast = makeReminder(
            trigger: now.addingTimeInterval(3600),
            snoozed: now.addingTimeInterval(-1800),
            title: "stale"
        )
        let result = ReminderFilter.snoozed.apply(
            to: [snoozedFuture, snoozedPast], referenceDate: now
        )
        XCTAssertEqual(result.map(\.title), ["snoozed"])
    }

    func testAllFilterReturnsEverythingSorted() {
        let r1 = makeReminder(trigger: now.addingTimeInterval(3600), title: "later")
        let r2 = makeReminder(trigger: now.addingTimeInterval(60), title: "sooner")
        let result = ReminderFilter.all.apply(
            to: [r1, r2], referenceDate: now
        )
        XCTAssertEqual(result.map(\.title), ["sooner", "later"])
    }
}
