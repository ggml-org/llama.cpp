import XCTest
import CryptoKit
@testable import TesseraCore

/// Tests for ``ChatPanelViewModel``. The view-model is
/// the SwiftUI bridge between the actor-based
/// ``ChatPanelStateMachine`` and the `@MainActor`
/// view layer. The tests focus on the polling/refresh
/// behavior and the user-action forwarding (submit,
/// reorder, hold, etc.).
@MainActor
final class ChatPanelViewModelTests: XCTestCase {

    actor InMemoryChatQueueStore: ChatQueueStoring {
        let documentID: UUID
        init(documentID: UUID) { self.documentID = documentID }
        func loadChatQueue(documentID: UUID) async throws -> ChatQueue { .empty }
        func saveChatQueue(_ queue: ChatQueue, documentID: UUID) async throws {}
        func loadDocument(id: UUID) async throws -> DocumentAST { .empty }
        func history(of documentID: UUID, limit: Int) async throws -> [Receipt] { [] }
    }

    private func makeModel() async -> (ChatPanelViewModel, ChatPanelStateMachine) {
        let docID = UUID()
        let store = await InMemoryChatQueueStore(documentID: docID)
        let machine = ChatPanelStateMachine(documentID: docID, store: store)
        let model = ChatPanelViewModel(
            documentID: docID,
            documentTitle: "Test Doc",
            stateMachine: machine
        )
        return (model, machine)
    }

    func testStartLoadsAndRefreshes() async {
        let (model, _) = await makeModel()
        await model.start()
        XCTAssertTrue(model.isLoaded)
        XCTAssertEqual(model.items.count, 0)
        model.stop()
    }

    func testSubmitEnqueuesAndUpdatesItems() async {
        let (model, _) = await makeModel()
        await model.start()
        model.inputText = "summarize section 2"
        await model.submit()
        XCTAssertEqual(model.items.count, 1)
        XCTAssertEqual(model.items.first?.message, "summarize section 2")
        XCTAssertEqual(model.items.first?.item.state, .pending)
        XCTAssertTrue(model.inputText.isEmpty)
        model.stop()
    }

    func testReorderMovesItem() async {
        let (model, _) = await makeModel()
        await model.start()
        model.inputText = "first"
        await model.submit()
        model.inputText = "second"
        await model.submit()
        // Front: "second", Back: "first"
        XCTAssertEqual(model.items.first?.message, "second")
        // Move the front to the back.
        if let firstID = model.items.first?.item.id {
            await model.reorder(itemID: firstID, to: 1)
        }
        XCTAssertEqual(model.items.first?.message, "first")
        model.stop()
    }

    func testHoldYourHorsesChangesHoldMode() async {
        let (model, _) = await makeModel()
        await model.start()
        XCTAssertEqual(model.holdMode, .running)
        await model.holdYourHorses()
        XCTAssertEqual(model.holdMode, .holdRequested)
        XCTAssertNotNil(model.holdDialog)
        await model.resume()
        XCTAssertEqual(model.holdMode, .running)
        XCTAssertNil(model.holdDialog)
        model.stop()
    }

    func testDeleteRemovesItem() async {
        let (model, _) = await makeModel()
        await model.start()
        model.inputText = "delete me"
        await model.submit()
        let id = model.items.first!.item.id
        await model.delete(itemID: id)
        XCTAssertEqual(model.items.count, 0)
        model.stop()
    }

    func testDraggableItemsIncludesPending() async {
        let (model, _) = await makeModel()
        await model.start()
        model.inputText = "test"
        await model.submit()
        let firstID = model.items.first!.item.id
        // Pending items are draggable.
        let draggable = model.draggableItems
        XCTAssertTrue(draggable.contains(where: { $0.item.id == firstID }))
        model.stop()
    }
}
