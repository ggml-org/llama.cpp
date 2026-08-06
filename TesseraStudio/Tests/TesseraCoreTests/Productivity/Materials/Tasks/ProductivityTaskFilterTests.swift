import XCTest
@testable import TesseraCore

/// Tests for the in-memory list-filter logic. The data layer
/// is async; these tests exercise the same predicates
/// against a `ProductivityTaskFilter` helper to lock in the
/// Today / Upcoming / Anytime / Someday semantics without
/// needing a Postgres connection.
final class ProductivityTaskFilterTests: XCTestCase {

    // MARK: - Filter predicates

    func testInboxFilterMatchesListOnly() {
        let inbox = ProductivityTask(title: "x", list: .inbox)
        let today = ProductivityTask(title: "y", list: .today)
        XCTAssertTrue(ProductivityTaskFilter.isIn(inbox, list: .inbox))
        XCTAssertFalse(ProductivityTaskFilter.isIn(today, list: .inbox))
    }

    func testTodayFilterMatchesDueIn24h() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let due = now.addingTimeInterval(2 * 3600)
        let task = ProductivityTask(title: "x", dueAt: due, list: .anytime)
        XCTAssertTrue(ProductivityTaskFilter.isIn(task, list: .today, asOf: now))
    }

    func testTodayFilterSkipsCompleted() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let due = now.addingTimeInterval(2 * 3600)
        let task = ProductivityTask(
            title: "x",
            dueAt: due,
            completedAt: now,
            list: .anytime
        )
        XCTAssertFalse(ProductivityTaskFilter.isIn(task, list: .today, asOf: now))
    }

    func testTodayFilterIncludesOverdue() {
        // Overdue tasks (due in the past) belong in Today
        // (per the spec's "Today auto-populates from due
        // date, including overdue" rule). The task shows
        // with a red "overdue" badge in the list view.
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let due = now.addingTimeInterval(-30 * 24 * 3600)
        let task = ProductivityTask(title: "x", dueAt: due, list: .anytime)
        XCTAssertTrue(ProductivityTaskFilter.isIn(task, list: .today, asOf: now))
    }

    func testUpcomingFilterMatchesDueIn7Days() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let cal = Calendar.current
        let in2d = cal.date(byAdding: .day, value: 2, to: cal.startOfDay(for: now))!
        let task = ProductivityTask(title: "x", dueAt: in2d, list: .anytime)
        XCTAssertTrue(ProductivityTaskFilter.isIn(task, list: .upcoming, asOf: now))
    }

    func testUpcomingFilterSkipsToday() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let in3h = now.addingTimeInterval(3 * 3600)
        let task = ProductivityTask(title: "x", dueAt: in3h, list: .anytime)
        XCTAssertFalse(ProductivityTaskFilter.isIn(task, list: .upcoming, asOf: now))
    }

    func testAnytimeFilterMatchesByList() {
        let task = ProductivityTask(title: "x", list: .anytime)
        XCTAssertTrue(ProductivityTaskFilter.isIn(task, list: .anytime))
    }

    func testSomedayFilterMatchesByList() {
        let task = ProductivityTask(title: "x", list: .someday)
        XCTAssertTrue(ProductivityTaskFilter.isIn(task, list: .someday))
    }

    // MARK: - Sort

    func testSortTodayByDueAscending() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let a = ProductivityTask(title: "a", dueAt: now.addingTimeInterval(3600), list: .anytime)
        let b = ProductivityTask(title: "b", dueAt: now.addingTimeInterval(1800), list: .anytime)
        let c = ProductivityTask(title: "c", dueAt: now.addingTimeInterval(7200), list: .anytime)
        let sorted: [ProductivityTask] = ProductivityTaskFilter.sortForList([a, b, c], list: .today, asOf: now)
        XCTAssertEqual(sorted.map { $0.title }, ["b", "a", "c"])
    }

    func testSortCompletedNewestFirst() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let older = ProductivityTask(title: "old", completedAt: now.addingTimeInterval(-3600), list: .anytime)
        let newer = ProductivityTask(title: "new", completedAt: now, list: .anytime)
        let sorted: [ProductivityTask] = ProductivityTaskFilter.sortForList([older, newer], list: .anytime, asOf: now)
        XCTAssertEqual(sorted.map { $0.title }, ["new", "old"])
    }

    func testSortByPriorityThenTitle() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let low = ProductivityTask(title: "alpha", priority: .low, list: .anytime)
        let high = ProductivityTask(title: "beta", priority: .high, list: .anytime)
        let none = ProductivityTask(title: "gamma", priority: .none, list: .anytime)
        let sorted: [ProductivityTask] = ProductivityTaskFilter.sortForList([low, high, none], list: .anytime, asOf: now)
        XCTAssertEqual(sorted.map { $0.title }, ["beta", "alpha", "gamma"])
    }
}

