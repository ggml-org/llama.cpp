import XCTest
@testable import TesseraCore

/// Tests for ``CrossDocumentChatRegistry``. The tests use
/// the real ``ChatPanelStateMachine`` with an in-memory
/// `ChatQueueStoring` mock so they don't need a real
/// Postgres + Keychain setup. The registry's
/// `pauseAll()` exercises the real `forceHold()` path;
/// the state machine transitions to `.hold` and the
/// registry reports the new state correctly.
final class CrossDocumentChatRegistryTests: XCTestCase {

    actor InMemoryChatQueueStore: ChatQueueStoring {
        let documentID: UUID
        init(documentID: UUID) { self.documentID = documentID }
        func loadChatQueue(documentID: UUID) async throws -> ChatQueue { .empty }
        func saveChatQueue(_ queue: ChatQueue, documentID: UUID) async throws {}
        func loadDocument(id: UUID) async throws -> DocumentAST { .empty }
        func history(of documentID: UUID, limit: Int) async throws -> [Receipt] { [] }
    }

    private func makeMachine(documentID: UUID) async -> ChatPanelStateMachine {
        let store = await InMemoryChatQueueStore(documentID: documentID)
        return ChatPanelStateMachine(documentID: documentID, store: store)
    }

    // MARK: - Register / unregister

    func testRegisterAddsDocument() async {
        let registry = CrossDocumentChatRegistry()
        let docID = UUID()
        let machine = await makeMachine(documentID: docID)
        await registry.register(machine, for: docID, title: "Doc A")
        let count = await registry.registrationCount
        XCTAssertEqual(count, 1)
    }

    func testUnregisterRemovesDocument() async {
        let registry = CrossDocumentChatRegistry()
        let docID = UUID()
        let machine = await makeMachine(documentID: docID)
        await registry.register(machine, for: docID, title: "Doc A")
        await registry.unregister(documentID: docID)
        let count = await registry.registrationCount
        XCTAssertEqual(count, 0)
    }

    func testUnregisterIsIdempotent() async {
        let registry = CrossDocumentChatRegistry()
        let docID = UUID()
        await registry.unregister(documentID: docID)  // no-op
        let count = await registry.registrationCount
        XCTAssertEqual(count, 0)
    }

    // MARK: - Active documents

    func testActiveDocumentsListsRegistered() async {
        let registry = CrossDocumentChatRegistry()
        let a = UUID()
        let b = UUID()
        let m1 = await makeMachine(documentID: a)
        let m2 = await makeMachine(documentID: b)
        await registry.register(m1, for: a, title: "Doc A")
        await registry.register(m2, for: b, title: "Doc B")
        await registry.setCurrent(documentID: a)
        let docs = await registry.activeDocuments()
        XCTAssertEqual(docs.count, 2)
        // Current is first.
        XCTAssertEqual(docs.first?.documentID, a)
        XCTAssertTrue(docs.first?.isCurrent ?? false)
    }

    func testActiveDocumentsSortedByTitle() async {
        let registry = CrossDocumentChatRegistry()
        let a = UUID()
        let b = UUID()
        let m1 = await makeMachine(documentID: a)
        let m2 = await makeMachine(documentID: b)
        await registry.register(m1, for: a, title: "B")
        await registry.register(m2, for: b, title: "A")
        let docs = await registry.activeDocuments()
        XCTAssertEqual(docs.map { $0.title }, ["A", "B"])
    }

    // MARK: - Pause all

    func testPauseAllCallsForceHoldOnEach() async {
        let registry = CrossDocumentChatRegistry()
        let a = UUID()
        let b = UUID()
        let m1 = await makeMachine(documentID: a)
        let m2 = await makeMachine(documentID: b)
        await registry.register(m1, for: a, title: "A")
        await registry.register(m2, for: b, title: "B")
        await registry.pauseAll()
        let mode1 = await m1.holdMode
        let mode2 = await m2.holdMode
        XCTAssertEqual(mode1, .hold)
        XCTAssertEqual(mode2, .hold)
    }

    // MARK: - In-flight tracking

    func testSetInFlightCountUpdatesActiveDocuments() async {
        let registry = CrossDocumentChatRegistry()
        let a = UUID()
        let m1 = await makeMachine(documentID: a)
        await registry.register(m1, for: a, title: "A")
        await registry.setInFlightCount(3, for: a)
        let docs = await registry.activeDocuments()
        XCTAssertEqual(docs.first?.inFlightItemCount, 3)
    }

    // MARK: - Lookup

    func testTitleLookup() async {
        let registry = CrossDocumentChatRegistry()
        let a = UUID()
        let m1 = await makeMachine(documentID: a)
        await registry.register(m1, for: a, title: "My Doc")
        let title = await registry.title(for: a)
        XCTAssertEqual(title, "My Doc")
        let missing = await registry.title(for: UUID())
        XCTAssertNil(missing)
    }

    func testCurrentDocument() async {
        let registry = CrossDocumentChatRegistry()
        let a = UUID()
        let m1 = await makeMachine(documentID: a)
        await registry.register(m1, for: a, title: "A")
        await registry.setCurrent(documentID: a)
        let current = await registry.currentDocument
        XCTAssertEqual(current, a)
    }
}
