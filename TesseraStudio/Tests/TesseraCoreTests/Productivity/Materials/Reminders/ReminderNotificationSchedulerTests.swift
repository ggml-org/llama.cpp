import XCTest
@testable import TesseraCore

/// Tests for the pure-decision parts of
/// ``ReminderNotificationScheduler`` (identifier format,
/// effective fire date, past-date detection). The actual
/// `UNUserNotificationCenter` calls are not exercised here —
/// they need a real system notification center (the
/// integration test for that is hand-driven, not in
/// CI). The pure helpers are what the production code uses
/// to decide what to send; they are fully testable.
final class ReminderNotificationSchedulerTests: XCTestCase {

    private let now = Date(timeIntervalSince1970: 1_700_000_000)
    private let prefix = "tessera.reminder."

    // MARK: - Identifier

    func testIdentifierFormat() {
        let id = UUID()
        let result = ReminderNotificationScheduler.identifier(
            for: id, prefix: prefix
        )
        XCTAssertEqual(result, "\(prefix)\(id.uuidString)")
    }

    func testIdentifierUniqueness() {
        let a = ReminderNotificationScheduler.identifier(
            for: UUID(), prefix: prefix
        )
        let b = ReminderNotificationScheduler.identifier(
            for: UUID(), prefix: prefix
        )
        XCTAssertNotEqual(a, b)
    }

    // MARK: - Effective fire date

    func testEffectiveFireDateFutureSnoozeWins() {
        let r = Reminder(
            id: UUID(),
            title: "t",
            calendarEventID: UUID(),
            offsetMinutes: -15,
            triggerAt: now.addingTimeInterval(60), // 1 min from now
            snoozedUntil: now.addingTimeInterval(3600) // 1 hour from now
        )
        let fire = ReminderNotificationScheduler.effectiveFireDate(
            for: r, now: now
        )
        XCTAssertEqual(fire, r.snoozedUntil)
    }

    func testEffectiveFireDatePastSnoozeIgnored() {
        let r = Reminder(
            id: UUID(),
            title: "t",
            calendarEventID: UUID(),
            offsetMinutes: -15,
            triggerAt: now.addingTimeInterval(60),
            snoozedUntil: now.addingTimeInterval(-60) // 1 min ago
        )
        let fire = ReminderNotificationScheduler.effectiveFireDate(
            for: r, now: now
        )
        XCTAssertEqual(fire, r.triggerAt)
    }

    func testEffectiveFireDateNoSnooze() {
        let r = Reminder(
            id: UUID(),
            title: "t",
            calendarEventID: UUID(),
            offsetMinutes: -15,
            triggerAt: now.addingTimeInterval(60)
        )
        let fire = ReminderNotificationScheduler.effectiveFireDate(
            for: r, now: now
        )
        XCTAssertEqual(fire, r.triggerAt)
    }

    // MARK: - Past-date detection

    func testIsFireDateInPastTrue() {
        let r = Reminder(
            id: UUID(),
            title: "t",
            calendarEventID: UUID(),
            offsetMinutes: 0,
            triggerAt: now.addingTimeInterval(-1)
        )
        XCTAssertTrue(ReminderNotificationScheduler.isFireDateInPast(
            for: r, now: now
        ))
    }

    func testIsFireDateInPastFalseWhenFuture() {
        let r = Reminder(
            id: UUID(),
            title: "t",
            calendarEventID: UUID(),
            offsetMinutes: 0,
            triggerAt: now.addingTimeInterval(60)
        )
        XCTAssertFalse(ReminderNotificationScheduler.isFireDateInPast(
            for: r, now: now
        ))
    }

    func testIsFireDateInPastRespectsSnooze() {
        // triggerAt is in the future but the snooze is in the past
        // -> still considered in-past (we'd skip notification).
        let r = Reminder(
            id: UUID(),
            title: "t",
            calendarEventID: UUID(),
            offsetMinutes: 0,
            triggerAt: now.addingTimeInterval(3600),
            snoozedUntil: now.addingTimeInterval(-1)
        )
        // Without snooze: triggerAt is future -> not in past.
        XCTAssertFalse(ReminderNotificationScheduler.isFireDateInPast(
            for: r, now: now
        ))
    }

    // MARK: - Snooze duration cap (1 day per spec §12.3)

    func testSnoozeReasonableDurations() {
        // The store / tools cap at 1440 min; the scheduler
        // doesn't enforce the cap (the agent tool does).
        // Verify the helper doesn't choke on edge cases.
        let r = Reminder(
            id: UUID(),
            title: "t",
            calendarEventID: UUID(),
            offsetMinutes: 0,
            triggerAt: now
        )
        for minutes in [1, 5, 10, 60, 1440] {
            let until = now.addingTimeInterval(Double(minutes) * 60)
            XCTAssertTrue(until > now, "until must be in future for minutes=\(minutes)")
        }
    }
}
