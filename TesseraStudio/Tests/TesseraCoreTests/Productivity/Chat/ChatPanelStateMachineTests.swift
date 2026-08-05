import XCTest
import CryptoKit
@testable import TesseraCore

/// Tests for ``ChatPanelStateMachine``. The tests use an
/// in-memory `ChatQueueStoring` mock so they don't need a
/// real Postgres + Keychain setup. The mock supports the
/// chat-queue load/save path and a stub document history.
final class ChatPanelStateMachineTests: XCTestCase {

    // MARK: - Mocks

    /// In-memory mock of ``ChatQueueStoring``. The
    /// `history` and `loadDocument` methods return empty
    /// results; the state machine's tests don't exercise
    /// those paths.
    actor InMemoryChatQueueStore: ChatQueueStoring {
        let documentID: UUID
        var queue: ChatQueue

        init(documentID: UUID, queue: ChatQueue = .empty) {
            self.documentID = documentID
            self.queue = queue
        }

        func loadChatQueue(documentID: UUID) async throws -> ChatQueue {
            guard documentID == self.documentID else {
                throw DocumentStoreError.documentNotFound(id: documentID)
            }
            return queue
        }

        func saveChatQueue(_ queue: ChatQueue, documentID: UUID) async throws {
            guard documentID == self.documentID else {
                throw DocumentStoreError.documentNotFound(id: documentID)
            }
            self.queue = queue
        }

        func loadDocument(id: UUID) async throws -> DocumentAST {
            return .empty
        }

        func history(of documentID: UUID, limit: Int) async throws -> [Receipt] {
            return []
        }
    }

    // MARK: - Test scaffolding

    private func makeMachine(
        queue: ChatQueue = .empty,
        documentID: UUID = UUID()
    ) async -> (ChatPanelStateMachine, InMemoryChatQueueStore) {
        let store = InMemoryChatQueueStore(documentID: documentID, queue: queue)
        let machine = ChatPanelStateMachine(
            documentID: documentID,
            store: store
        )
        return (machine, store)
    }

    // MARK: - Load

    func testLoadEmptyQueue() async throws {
        let (machine, _) = await makeMachine()
        let result = try await machine.load()
        if case .empty = result { } else {
            XCTFail("expected .empty, got \(result)")
        }
    }

    func testLoadExistingQueue() async throws {
        let existing = ChatQueue(items: [
            ChatQueueItem(
                documentID: UUID(),
                order: 0,
                message: "first",
                actor: .user(UUID())
            )
        ])
        let (machine, _) = await makeMachine(queue: existing)
        let result = try await machine.load()
        if case .loaded(let count) = result {
            XCTAssertEqual(count, 1)
        } else {
            XCTFail("expected .loaded")
        }
    }

    // MARK: - Enqueue

    func testEnqueueEmptyMessageThrows() async throws {
        let (machine, _) = await makeMachine()
        _ = try await machine.load()
        do {
            _ = try await machine.enqueue(message: "   ")
            XCTFail("expected empty message to throw")
        } catch ChatPanelStateMachineError.emptyMessage {
            // expected
        }
    }

    func testEnqueueInsertsAtFront() async throws {
        let (machine, store) = await makeMachine()
        _ = try await machine.load()
        _ = try await machine.enqueue(message: "first")
        _ = try await machine.enqueue(message: "second")
        let queue = await machine.queue
        XCTAssertEqual(queue.orderedItems.map { $0.message }, ["second", "first"])
        // Persisted?
        let persisted = await store.queue
        XCTAssertEqual(persisted.orderedItems.map { $0.message }, ["second", "first"])
    }

    // MARK: - State transitions

    func testStartNextPendingTransitionsToInProgress() async throws {
        let (machine, _) = await makeMachine()
        _ = try await machine.load()
        let item = try await machine.enqueue(message: "do the thing")
        let started = try await machine.startNextPending()
        XCTAssertEqual(started?.id, item.id)
        let queue = await machine.queue
        XCTAssertEqual(queue.items.first?.state, .inProgress)
    }

    func testStartNextPendingReturnsNilWhenPaused() async throws {
        let (machine, _) = await makeMachine()
        _ = try await machine.load()
        _ = try await machine.enqueue(message: "paused")
        try await machine.holdYourHorses()
        try await machine.forceHold()  // skip the transient state
        let started = try await machine.startNextPending()
        XCTAssertNil(started)
    }

    func testStartNextPendingReturnsNilWhenEmpty() async throws {
        let (machine, _) = await makeMachine()
        _ = try await machine.load()
        let started = try await machine.startNextPending()
        XCTAssertNil(started)
    }

    func testMarkAppliedTransitionsAndIncrementsCount() async throws {
        let (machine, _) = await makeMachine()
        let docID = await machine.documentID
        _ = try await machine.load()
        let item = try await machine.enqueue(message: "apply me")
        let receipt = makeReceipt(for: docID)
        try await machine.markApplied(itemID: item.id, receipt: receipt)
        let queue = await machine.queue
        XCTAssertEqual(queue.items.first?.state, .applied)
        XCTAssertEqual(queue.items.first?.producedReceiptID, receipt.id)
        let count = await machine.currentReceiptCount
        XCTAssertEqual(count, 1)
    }

    func testMarkAppliedRejectsMismatchedDocument() async throws {
        let (machine, _) = await makeMachine()
        _ = try await machine.load()
        let item = try await machine.enqueue(message: "x")
        let receipt = Receipt(
            documentID: UUID(),  // different doc id
            actor: .user(UUID()),
            mutations: [],
            priorReceiptID: nil,
            signature: Data(repeating: 0, count: 64),
            summary: "wrong doc"
        )
        do {
            try await machine.markApplied(itemID: item.id, receipt: receipt)
            XCTFail("expected documentMismatch")
        } catch ChatPanelStateMachineError.documentMismatch {
            // expected
        }
    }

    func testMarkFailedStoresFailureNote() async throws {
        let (machine, _) = await makeMachine()
        _ = try await machine.load()
        let item = try await machine.enqueue(message: "fail me")
        try await machine.markFailed(itemID: item.id, error: TestError.boom)
        let queue = await machine.queue
        XCTAssertEqual(queue.items.first?.state, .failed)
        let meta = await machine.currentMeta()
        let note = meta[item.id]?.failureNote
        XCTAssertNotNil(note)
        XCTAssertTrue(note!.contains("boom"))
    }

    // MARK: - Hold mode

    func testHoldYourHorsesTransitionsToHoldRequested() async throws {
        let (machine, _) = await makeMachine()
        _ = try await machine.load()
        try await machine.holdYourHorses()
        let mode = await machine.holdMode
        XCTAssertEqual(mode, .holdRequested)
    }

    func testForceHoldSkipsRequested() async throws {
        let (machine, _) = await makeMachine()
        _ = try await machine.load()
        try await machine.forceHold()
        let mode = await machine.holdMode
        XCTAssertEqual(mode, .hold)
    }

    func testResumeTransitionsToResumingThenRunning() async throws {
        let (machine, _) = await makeMachine()
        _ = try await machine.load()
        try await machine.forceHold()
        try await machine.resume()
        // After resume, the mode is back to .running
        // (the .resuming state is transient).
        let mode = await machine.holdMode
        XCTAssertEqual(mode, .running)
    }

    func testHoldIsIdempotent() async throws {
        let (machine, _) = await makeMachine()
        _ = try await machine.load()
        try await machine.forceHold()
        try await machine.forceHold()  // idempotent
        let mode = await machine.holdMode
        XCTAssertEqual(mode, .hold)
    }

    // MARK: - Reorder

    func testReorderMovesItem() async throws {
        let (machine, _) = await makeMachine()
        _ = try await machine.load()
        let a = try await machine.enqueue(message: "a")
        let b = try await machine.enqueue(message: "b")
        let c = try await machine.enqueue(message: "c")
        // Front is c (newest first). Move c to the end.
        try await machine.reorder(itemID: c.id, toNewIndex: 2)
        let queue = await machine.queue
        XCTAssertEqual(queue.orderedItems.map { $0.message }, ["b", "a", "c"])
        // Identities are preserved.
        XCTAssertEqual(Set(queue.items.map { $0.id }), Set([a.id, b.id, c.id]))
    }

    // MARK: - Supersession

    func testSupersedeMarksOriginal() async throws {
        let (machine, _) = await makeMachine()
        _ = try await machine.load()
        let a = try await machine.enqueue(message: "a")
        let b = try await machine.enqueue(message: "b")
        try await machine.supersede(oldItemID: a.id, by: b.id)
        let queue = await machine.queue
        XCTAssertEqual(queue.items.first(where: { $0.id == a.id })?.supersededByID, b.id)
    }

    func testUnsupersedeClearsMarker() async throws {
        let (machine, _) = await makeMachine()
        _ = try await machine.load()
        let a = try await machine.enqueue(message: "a")
        let b = try await machine.enqueue(message: "b")
        try await machine.supersede(oldItemID: a.id, by: b.id)
        try await machine.unsupersede(itemID: a.id)
        let queue = await machine.queue
        XCTAssertNil(queue.items.first(where: { $0.id == a.id })?.supersededByID)
    }

    // MARK: - Delete

    func testDeletePendingItem() async throws {
        let (machine, _) = await makeMachine()
        _ = try await machine.load()
        let a = try await machine.enqueue(message: "delete me")
        try await machine.delete(itemID: a.id)
        let queue = await machine.queue
        XCTAssertTrue(queue.items.isEmpty)
    }

    func testDeleteAppliedThrows() async throws {
        let (machine, _) = await makeMachine()
        let docID = await machine.documentID
        _ = try await machine.load()
        let item = try await machine.enqueue(message: "applied")
        let receipt = makeReceipt(for: docID)
        try await machine.markApplied(itemID: item.id, receipt: receipt)
        do {
            try await machine.delete(itemID: item.id)
            XCTFail("expected cannotDeleteApplied")
        } catch ChatPanelStateMachineError.cannotDeleteApplied {
            // expected
        }
    }

    // MARK: - Persistence round-trip

    func testPersistenceRoundTrip() async throws {
        let docID = UUID()
        let (machine, store) = await makeMachine(documentID: docID)
        _ = try await machine.load()
        _ = try await machine.enqueue(message: "round-trip 1")
        _ = try await machine.enqueue(message: "round-trip 2")
        // Reload.
        let machine2 = ChatPanelStateMachine(
            documentID: docID,
            store: store
        )
        _ = try await machine2.load()
        let queue = await machine2.queue
        XCTAssertEqual(queue.orderedItems.map { $0.message }, ["round-trip 2", "round-trip 1"])
    }

    func testPersistenceRoundTrip1000ItemsUnder1Second() async throws {
        let docID = UUID()
        let (machine, store) = await makeMachine(documentID: docID)
        _ = try await machine.load()
        for i in 0..<1000 {
            _ = try await machine.enqueue(message: "item \(i)")
        }
        let start = Date()
        let machine2 = ChatPanelStateMachine(
            documentID: docID,
            store: store
        )
        _ = try await machine2.load()
        let elapsed = Date().timeIntervalSince(start)
        let queue = await machine2.queue
        XCTAssertEqual(queue.items.count, 1000)
        XCTAssertLessThan(elapsed, 1.0, "1000-item load should take < 1s, took \(elapsed)s")
    }

    // MARK: - Helpers

    private enum TestError: Error {
        case boom
    }

    private func makeReceipt(for documentID: UUID) -> Receipt {
        // The receipt's actor is the user; the signature
        // is a dummy 64-byte value (we don't verify in
        // these tests — the state machine only checks
        // the document id).
        return Receipt(
            documentID: documentID,
            actor: .user(UUID()),
            mutations: [],
            priorReceiptID: nil,
            signature: Data(repeating: 0, count: 64),
            summary: "test receipt"
        )
    }
}
