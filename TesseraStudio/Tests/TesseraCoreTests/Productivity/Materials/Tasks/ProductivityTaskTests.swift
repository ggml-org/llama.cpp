import XCTest
@testable import TesseraCore

/// Unit tests for the ``ProductivityTask`` model: JSON
/// round-trip, list/priority serialization, due-date helpers,
/// list auto-classification, JSON helpers.
final class ProductivityTaskTests: XCTestCase {

    // MARK: - JSON round-trip

    func testProductivityTaskRoundTripsJSON() throws {
        let id = UUID()
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let due = Date(timeIntervalSince1970: 1_700_086_400)
        let original = ProductivityTask(
            id: id,
            title: "Review Q3 report",
            notes: "Read sections 1-3 first",
            dueAt: due,
            completedAt: nil,
            priority: .high,
            tags: ["work", "quarterly"],
            list: .today,
            linkedEntityIDs: [UUID(), UUID()],
            sourceURL: "https://example.com/q3-report",
            createdAt: now,
            updatedAt: now
        )
        let data = try original.jsonData()
        let decoded = try ProductivityTask.from(jsonData: data)
        XCTAssertEqual(decoded, original)
        XCTAssertEqual(decoded.title, "Review Q3 report")
        XCTAssertEqual(decoded.list, .today)
        XCTAssertEqual(decoded.priority, .high)
        XCTAssertEqual(decoded.linkedEntityIDs.count, 2)
    }

    func testEmptyProductivityTaskRoundTrips() throws {
        let fixedDate = Date(timeIntervalSince1970: 1_700_000_000)
        let original = ProductivityTask(
            title: "buy milk",
            createdAt: fixedDate,
            updatedAt: fixedDate
        )
        let data = try original.jsonData()
        let decoded = try ProductivityTask.from(jsonData: data)
        XCTAssertEqual(decoded, original)
        XCTAssertEqual(decoded.title, "buy milk")
        XCTAssertEqual(decoded.list, .inbox)
        XCTAssertEqual(decoded.priority, .none)
        XCTAssertNil(decoded.dueAt)
    }

    func testCompletedProductivityTaskRoundTrips() throws {
        let fixedDate = Date(timeIntervalSince1970: 1_700_000_000)
        let original = ProductivityTask(
            title: "send invoice",
            completedAt: fixedDate,
            priority: .low,
            list: .anytime,
            createdAt: fixedDate,
            updatedAt: fixedDate
        )
        let data = try original.jsonData()
        let decoded = try ProductivityTask.from(jsonData: data)
        XCTAssertEqual(decoded, original)
        XCTAssertNotNil(decoded.completedAt)
        XCTAssertTrue(decoded.isCompleted)
    }

    // MARK: - List + priority serialization

    func testEachListSerializesDistinctly() throws {
        for list in ProductivityTask.List.allCases {
            let task = ProductivityTask(title: "x", list: list)
            let data = try task.jsonData()
            let s = String(data: data, encoding: .utf8) ?? ""
            XCTAssertTrue(
                s.contains("\"\(list.rawValue)\""),
                "list \(list.rawValue) should appear in JSON: \(s)"
            )
        }
    }

    func testEachPrioritySerializesDistinctly() throws {
        for priority in ProductivityTask.Priority.allCases {
            let task = ProductivityTask(title: "x", priority: priority)
            let data = try task.jsonData()
            let s = String(data: data, encoding: .utf8) ?? ""
            XCTAssertTrue(
                s.contains("\"\(priority.rawValue)\""),
                "priority \(priority.rawValue) should appear in JSON: \(s)"
            )
        }
    }

    func testPriorityIsComparable() {
        XCTAssertLessThan(ProductivityTask.Priority.none, .low)
        XCTAssertLessThan(ProductivityTask.Priority.low, .medium)
        XCTAssertLessThan(ProductivityTask.Priority.medium, .high)
        XCTAssertGreaterThan(ProductivityTask.Priority.high, .none)
    }

    // MARK: - Due date helpers

    func testIsOverdueWhenDueInPast() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let past = now.addingTimeInterval(-3600)
        let task = ProductivityTask(title: "x", dueAt: past)
        XCTAssertTrue(task.isOverdue(asOf: now))
    }

    func testIsNotOverdueWhenCompleted() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let past = now.addingTimeInterval(-3600)
        let task = ProductivityTask(
            title: "x",
            dueAt: past,
            completedAt: now
        )
        XCTAssertFalse(task.isOverdue(asOf: now))
    }

    func testIsNotOverdueWhenDueInFuture() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let future = now.addingTimeInterval(3600)
        let task = ProductivityTask(title: "x", dueAt: future)
        XCTAssertFalse(task.isOverdue(asOf: now))
    }

    func testIsDueWithin24Hours() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let in3h = now.addingTimeInterval(3 * 3600)
        let in2d = now.addingTimeInterval(2 * 24 * 3600)
        XCTAssertTrue(ProductivityTask(title: "x", dueAt: in3h).isDueWithin24Hours(asOf: now))
        XCTAssertFalse(ProductivityTask(title: "x", dueAt: in2d).isDueWithin24Hours(asOf: now))
    }

    func testIsDueWithin7DaysButNotToday() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let cal = Calendar.current
        let tomorrow = cal.date(byAdding: .day, value: 1, to: cal.startOfDay(for: now))!
        let in2d = cal.date(byAdding: .day, value: 2, to: cal.startOfDay(for: now))!
        let in3h = now.addingTimeInterval(3 * 3600)
        XCTAssertTrue(ProductivityTask(title: "x", dueAt: tomorrow).isDueWithin7DaysButNotToday(asOf: now))
        XCTAssertTrue(ProductivityTask(title: "x", dueAt: in2d).isDueWithin7DaysButNotToday(asOf: now))
        XCTAssertFalse(ProductivityTask(title: "x", dueAt: in3h).isDueWithin7DaysButNotToday(asOf: now))
    }

    // MARK: - List auto-classification

    func testAutoClassifyTodayFor24h() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let due = now.addingTimeInterval(2 * 3600)
        let task = ProductivityTask(title: "x", dueAt: due)
        XCTAssertEqual(task.autoClassifiedList(asOf: now), .today)
    }

    func testAutoClassifyUpcomingFor7Days() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let due = now.addingTimeInterval(4 * 24 * 3600)
        let task = ProductivityTask(title: "x", dueAt: due)
        XCTAssertEqual(task.autoClassifiedList(asOf: now), .upcoming)
    }

    func testAutoClassifyAnytimeForNoDueDate() {
        let task = ProductivityTask(title: "x", dueAt: nil)
        XCTAssertEqual(task.autoClassifiedList(), .anytime)
    }

    // MARK: - Display helpers

    func testListDisplayNames() {
        XCTAssertEqual(ProductivityTask.List.inbox.displayName, "Inbox")
        XCTAssertEqual(ProductivityTask.List.today.displayName, "Today")
        XCTAssertEqual(ProductivityTask.List.upcoming.displayName, "Upcoming")
        XCTAssertEqual(ProductivityTask.List.anytime.displayName, "Anytime")
        XCTAssertEqual(ProductivityTask.List.someday.displayName, "Someday")
    }

    func testPriorityDisplayNames() {
        XCTAssertEqual(ProductivityTask.Priority.none.displayName, "no priority")
        XCTAssertEqual(ProductivityTask.Priority.high.displayName, "high")
    }

    func testNotesPreviewTruncatesLongNotes() {
        let long = String(repeating: "a", count: 100)
        let task = ProductivityTask(title: "x", notes: long)
        XCTAssertEqual(task.notesPreview.count, 61) // 60 + ellipsis
        XCTAssertTrue(task.notesPreview.hasSuffix("…"))
    }

    func testNotesPreviewEmpty() {
        let task = ProductivityTask(title: "x", notes: "")
        XCTAssertEqual(task.notesPreview, "")
    }

    // MARK: - JSON helpers

    func testJSONDataStringRoundTrip() throws {
        let fixedDate = Date(timeIntervalSince1970: 1_700_000_000)
        let task = ProductivityTask(
            title: "x",
            priority: .high,
            createdAt: fixedDate,
            updatedAt: fixedDate
        )
        let s = try task.jsonDataString()
        let data = s.data(using: .utf8)!
        let decoded = try ProductivityTask.from(jsonData: data)
        XCTAssertEqual(decoded, task)
    }

    func testJSONDataStringRoundTripViaStringInit() throws {
        let fixedDate = Date(timeIntervalSince1970: 1_700_000_000)
        let task = ProductivityTask(
            title: "x",
            priority: .medium,
            list: .today,
            createdAt: fixedDate,
            updatedAt: fixedDate
        )
        let s = try task.jsonDataString()
        let decoded = try ProductivityTask.from(jsonDataString: s)
        XCTAssertEqual(decoded, task)
    }
}
