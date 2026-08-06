import XCTest
@testable import TesseraCore

/// Unit tests for ``ProductivityTaskStore`` parts that don't
/// need a real Postgres connection. The end-to-end
/// upsert -> receipt -> fetch flow is exercised by the
/// integration test (``ProductivityTaskStoreIntegrationTests``)
/// which is env-gated on `TESSERA_DB_INTEGRATION=1`.
final class ProductivityTaskStoreTests: XCTestCase {

    // MARK: - Receipt type taxonomy

    func testReceiptTypesAreStable() {
        // The receipt type strings are persisted to
        // graph_receipts.receipt_type; changing them is
        // a schema migration. Pin them here.
        XCTAssertEqual(ProductivityTaskReceiptType.upsert.rawValue, "task_upsert")
        XCTAssertEqual(ProductivityTaskReceiptType.delete.rawValue, "task_delete")
        XCTAssertEqual(ProductivityTaskReceiptType.completed.rawValue, "task_completed")
        XCTAssertEqual(ProductivityTaskReceiptType.reopened.rawValue, "task_reopened")
        XCTAssertEqual(ProductivityTaskReceiptType.moved.rawValue, "task_moved")
        XCTAssertEqual(ProductivityTaskReceiptType.priorityChanged.rawValue, "task_priority_changed")
        XCTAssertEqual(ProductivityTaskReceiptType.dueDateChanged.rawValue, "task_due_date_changed")
        XCTAssertEqual(ProductivityTaskReceiptType.linkCreated.rawValue, "task_link_created")
        XCTAssertEqual(ProductivityTaskReceiptType.linkDeleted.rawValue, "task_link_deleted")
        XCTAssertEqual(ProductivityTaskReceiptType.createdFromChat.rawValue, "task_created_from_chat")
        XCTAssertEqual(ProductivityTaskReceiptType.createdFromNLU.rawValue, "task_created_from_nlu")
    }

    // MARK: - Entity type constant

    func testEntityTypeIsTask() {
        // The graph view's "by type" filter and the
        // migration's partial index predicate both rely on
        // this constant. Pin it.
        XCTAssertEqual(ProductivityTask.entityType, "task")
    }

    func testSubtypeStringMatchesList() {
        let task = ProductivityTask(title: "x", list: .today)
        XCTAssertEqual(task.subtypeString, "today")
    }

    // MARK: - Errors

    func testStoreErrorEquality() {
        let id = UUID()
        XCTAssertEqual(
            ProductivityTaskStoreError.taskNotFound(id: id),
            ProductivityTaskStoreError.taskNotFound(id: id)
        )
        XCTAssertNotEqual(
            ProductivityTaskStoreError.invalidTaskBody(reason: "x"),
            ProductivityTaskStoreError.invalidTaskBody(reason: "y")
        )
    }

    // MARK: - isCompleted / isOverdue behavior

    func testIsCompletedWhenCompletedAtSet() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let task = ProductivityTask(title: "x", completedAt: now)
        XCTAssertTrue(task.isCompleted)
    }

    func testNotCompletedWhenCompletedAtNil() {
        let task = ProductivityTask(title: "x")
        XCTAssertFalse(task.isCompleted)
    }

    // MARK: - JSON helpers

    func testJSONHelpersRoundTripUTF8() throws {
        let fixedDate = Date(timeIntervalSince1970: 1_700_000_000)
        let task = ProductivityTask(
            title: "x",
            priority: .high,
            list: .today,
            createdAt: fixedDate,
            updatedAt: fixedDate
        )
        let s = try task.jsonDataString()
        XCTAssertFalse(s.isEmpty)
        let decoded = try ProductivityTask.from(jsonDataString: s)
        XCTAssertEqual(decoded, task)
    }

    // MARK: - All lists present

    func testAllListsAreCases() {
        let lists: [ProductivityTask.List] = [.inbox, .today, .upcoming, .anytime, .someday]
        XCTAssertEqual(lists.count, ProductivityTask.List.allCases.count)
    }

    // MARK: - All priorities present

    func testAllPrioritiesAreCases() {
        let priorities: [ProductivityTask.Priority] = [.none, .low, .medium, .high]
        XCTAssertEqual(priorities.count, ProductivityTask.Priority.allCases.count)
    }
}
