import XCTest
@testable import TesseraCore

/// Tests for the per-document chat queue data model: state
/// transitions, ordering, superseding, serialization.
final class ChatQueueItemTests: XCTestCase {

    // MARK: - State transitions

    func testInitialStateIsPending() {
        let item = ChatQueueItem(
            documentID: UUID(),
            order: 0,
            message: "hi",
            actor: .user(UUID())
        )
        XCTAssertEqual(item.state, .pending)
    }

    func testStartTransition() {
        let queue = ChatQueue(items: [
            ChatQueueItem(
                documentID: UUID(),
                order: 0,
                message: "test",
                actor: .user(UUID())
            )
        ])
        let id = queue.items[0].id
        let updated = queue.starting(itemID: id)
        XCTAssertEqual(updated.items[0].state, .inProgress)
    }

    func testFinishTransition() {
        let queue = ChatQueue(items: [
            ChatQueueItem(
                documentID: UUID(),
                order: 0,
                message: "test",
                actor: .user(UUID())
            )
        ])
        let id = queue.items[0].id
        let receiptID = UUID()
        let updated = queue.finishing(itemID: id, with: receiptID)
        XCTAssertEqual(updated.items[0].state, .applied)
        XCTAssertEqual(updated.items[0].producedReceiptID, receiptID)
    }

    func testFailTransition() {
        let queue = ChatQueue(items: [
            ChatQueueItem(
                documentID: UUID(),
                order: 0,
                message: "test",
                actor: .user(UUID())
            )
        ])
        let id = queue.items[0].id
        let updated = queue.failing(itemID: id)
        XCTAssertEqual(updated.items[0].state, .failed)
    }

    func testFullLifecycle() {
        let item = ChatQueueItem(
            documentID: UUID(),
            order: 0,
            message: "x",
            actor: .user(UUID())
        )
        var queue = ChatQueue(items: [item])
        queue = queue.starting(itemID: item.id)
        XCTAssertEqual(queue.items[0].state, .inProgress)
        let receiptID = UUID()
        queue = queue.finishing(itemID: item.id, with: receiptID)
        XCTAssertEqual(queue.items[0].state, .applied)
        XCTAssertEqual(queue.items[0].producedReceiptID, receiptID)
    }

    // MARK: - Ordering

    func testOrderedItemsByOrderField() {
        let a = ChatQueueItem(documentID: UUID(), order: 1, message: "a", actor: .user(UUID()))
        let b = ChatQueueItem(documentID: UUID(), order: 0, message: "b", actor: .user(UUID()))
        let c = ChatQueueItem(documentID: UUID(), order: 2, message: "c", actor: .user(UUID()))
        let queue = ChatQueue(items: [a, b, c])
        XCTAssertEqual(queue.orderedItems.map { $0.message }, ["b", "a", "c"])
    }

    func testInsertAtFrontPushesOthersBack() {
        let a = ChatQueueItem(documentID: UUID(), order: 0, message: "a", actor: .user(UUID()))
        let b = ChatQueueItem(documentID: UUID(), order: 1, message: "b", actor: .user(UUID()))
        let queue = ChatQueue(items: [a, b])
        let newItem = ChatQueueItem(
            documentID: a.documentID,
            order: 99,  // order will be overwritten
            message: "new",
            actor: .user(UUID())
        )
        let updated = queue.insertingAtFront(newItem)
        XCTAssertEqual(updated.orderedItems.map { $0.message }, ["new", "a", "b"])
        XCTAssertEqual(updated.orderedItems[0].order, 0)
        XCTAssertEqual(updated.orderedItems[1].order, 1)
        XCTAssertEqual(updated.orderedItems[2].order, 2)
    }

    func testReorderItem() {
        let a = ChatQueueItem(documentID: UUID(), order: 0, message: "a", actor: .user(UUID()))
        let b = ChatQueueItem(documentID: UUID(), order: 1, message: "b", actor: .user(UUID()))
        let c = ChatQueueItem(documentID: UUID(), order: 2, message: "c", actor: .user(UUID()))
        let queue = ChatQueue(items: [a, b, c])
        // Move `a` to the end.
        let updated = queue.reordering(itemID: a.id, to: 2)
        XCTAssertEqual(updated.orderedItems.map { $0.message }, ["b", "c", "a"])
    }

    func testReorderUnknownItemIsNoOp() {
        let a = ChatQueueItem(documentID: UUID(), order: 0, message: "a", actor: .user(UUID()))
        let queue = ChatQueue(items: [a])
        let updated = queue.reordering(itemID: UUID(), to: 0)
        XCTAssertEqual(updated.orderedItems.map { $0.message }, ["a"])
    }

    // MARK: - Superseding

    func testSupersede() {
        let a = ChatQueueItem(documentID: UUID(), order: 0, message: "a", actor: .user(UUID()))
        let b = ChatQueueItem(documentID: UUID(), order: 1, message: "b", actor: .user(UUID()))
        let queue = ChatQueue(items: [a, b])
        let updated = queue.superseding(itemID: a.id, by: b.id)
        let itemA = updated.items.first { $0.id == a.id }
        XCTAssertEqual(itemA?.supersededByID, b.id)
    }

    // MARK: - Serialization

    func testQueueSerialization() throws {
        let queue = ChatQueue(items: [
            ChatQueueItem(
                documentID: UUID(),
                order: 0,
                message: "first",
                state: .applied,
                actor: .user(UUID()),
                producedReceiptID: UUID()
            ),
            ChatQueueItem(
                documentID: UUID(),
                order: 1,
                message: "second",
                state: .pending,
                actor: .agent(UUID(), model: "x", promptHash: "y")
            )
        ])
        let data = try JSONEncoder().encode(queue)
        let decoded = try JSONDecoder().decode(ChatQueue.self, from: data)
        XCTAssertEqual(decoded.items.count, 2)
        XCTAssertEqual(decoded.items[0].state, .applied)
        XCTAssertEqual(decoded.items[1].state, .pending)
    }

    func testEmptyQueueSerialization() throws {
        let queue = ChatQueue.empty
        let data = try JSONEncoder().encode(queue)
        let decoded = try JSONDecoder().decode(ChatQueue.self, from: data)
        XCTAssertTrue(decoded.items.isEmpty)
    }

    // MARK: - State enum

    func testAllStatesPresent() {
        let expected: Set<ChatQueueItem.State> = [.pending, .inProgress, .applied, .failed]
        XCTAssertEqual(Set(ChatQueueItem.State.allCases), expected)
    }
}
