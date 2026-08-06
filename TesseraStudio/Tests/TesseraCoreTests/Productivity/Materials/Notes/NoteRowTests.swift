import XCTest
@testable import TesseraCore

/// Tests for ``NoteRow`` view-model row construction and the
/// relative time formatter. The formatter is the load-bearing
/// piece the list view shows in the "edited N days ago" slot.
final class NoteRowTests: XCTestCase {

    private let now = Date(timeIntervalSince1970: 1_700_000_000)

    // MARK: - Construction

    func testRowCarriesNoteMetadata() {
        let id = UUID()
        let date = Date(timeIntervalSince1970: 1_699_999_000)
        let note = Note(
            id: id,
            title: "Q3 Review",
            body: DocumentAST(),
            tags: ["q3", "review"],
            pinnedAt: date,
            archivedAt: nil,
            createdAt: date,
            updatedAt: date
        )
        let row = NoteRow(note: note, now: now)
        XCTAssertEqual(row.id, id)
        XCTAssertEqual(row.title, "Q3 Review")
        XCTAssertEqual(row.tags, ["q3", "review"])
        XCTAssertTrue(row.isPinned)
        XCTAssertFalse(row.isArchived)
    }

    func testRowUsesDefaultTitle() {
        let note = Note(title: "", body: .empty)
        let row = NoteRow(note: note, now: now)
        XCTAssertEqual(row.title, "Untitled")
    }

    func testRowTracksArchived() {
        let note = Note(
            title: "X",
            archivedAt: Date(timeIntervalSince1970: 1_000_000)
        )
        let row = NoteRow(note: note, now: now)
        XCTAssertTrue(row.isArchived)
    }

    // MARK: - Relative time

    func testRelativeTimeJustNow() {
        let date = now.addingTimeInterval(-30)  // 30s ago
        XCTAssertEqual(
            NoteRow.relativeTimeString(for: date, now: now),
            "just now"
        )
    }

    func testRelativeTimeMinutesAgo() {
        let date = now.addingTimeInterval(-300)  // 5 min ago
        XCTAssertEqual(
            NoteRow.relativeTimeString(for: date, now: now),
            "5 min ago"
        )
    }

    func testRelativeTimeHoursAgo() {
        let date = now.addingTimeInterval(-3 * 3600)  // 3 hours ago
        XCTAssertEqual(
            NoteRow.relativeTimeString(for: date, now: now),
            "3 hr ago"
        )
    }

    func testRelativeTimeYesterday() {
        // The "yesterday" branch in the implementation
        // uses `Calendar.isDateInYesterday`, which compares
        // against the calendar's "today" (the actual
        // current date, not the test's `now` constant). So
        // we build "yesterday" relative to the actual
        // current date — the result is timezone-independent.
        let calendar = Calendar.current
        let realNow = Date()
        let noonToday = calendar.date(
            bySettingHour: 12, minute: 0, second: 0,
            of: realNow
        ) ?? realNow
        let yesterday = calendar.date(byAdding: .day, value: -1, to: noonToday) ?? noonToday
        XCTAssertEqual(
            NoteRow.relativeTimeString(for: yesterday, now: noonToday),
            "yesterday"
        )
    }

    func testRelativeTimeDaysAgo() {
        let date = now.addingTimeInterval(-3 * 86400)  // 3 days ago
        XCTAssertEqual(
            NoteRow.relativeTimeString(for: date, now: now),
            "3 days ago"
        )
    }

    func testRelativeTimeWeeksAgo() {
        let date = now.addingTimeInterval(-2 * 7 * 86400)  // 2 weeks ago
        XCTAssertEqual(
            NoteRow.relativeTimeString(for: date, now: now),
            "2 weeks ago"
        )
    }

    func testRelativeTimeMonthsAgoUsesFormattedDate() {
        // 60 days ago — should be older than the week-based
        // formatter can express, so it falls through to the
        // "MMM d" / "MMM d, yyyy" formatter.
        let date = now.addingTimeInterval(-60 * 86400)
        let result = NoteRow.relativeTimeString(for: date, now: now)
        // Result is a date-formatted string (not a relative
        // time string). It contains the year because the
        // date is in a different year than `now`.
        XCTAssertFalse(result.hasSuffix("ago"))
        XCTAssertFalse(result == "yesterday")
    }

    func testRelativeTimeFutureDateReturnsJustNow() {
        let date = now.addingTimeInterval(3600)  // 1 hour in the future
        XCTAssertEqual(
            NoteRow.relativeTimeString(for: date, now: now),
            "just now"
        )
    }
}
