import Foundation

// MARK: - ChatQueueStoring

/// The persistence seam the ``ChatPanelStateMachine`` uses
/// for the chat queue. A protocol so the state machine can
/// be tested without a real Postgres + Keychain setup.
///
/// The production implementation is
/// ``DocumentStoreChatQueueStore``, which delegates to
/// ``DocumentStore`` (the Phase 1 wrapper around
/// ``TesseraDataLayer``). Tests can use any in-memory
/// implementation that satisfies the protocol.
///
/// **Why a protocol.** The state machine's persistence is
/// just a load + save (the queue is read once on document
/// open, written on every state transition). The state
/// machine doesn't need the rest of ``DocumentStore``'s
/// surface (the receipt chain, the AST, the search). The
/// protocol is the minimum seam.
public protocol ChatQueueStoring: Sendable {
    func loadChatQueue(documentID: UUID) async throws -> ChatQueue
    func saveChatQueue(_ queue: ChatQueue, documentID: UUID) async throws
    func loadDocument(id: UUID) async throws -> DocumentAST
    func history(of documentID: UUID, limit: Int) async throws -> [Receipt]
}

// MARK: - DocumentStoreChatQueueStore

/// The production ``ChatQueueStoring`` implementation.
/// Wraps the Phase 1 ``DocumentStore`` and exposes only
/// the methods the state machine needs.
public struct DocumentStoreChatQueueStore: ChatQueueStoring {
    private let documentStore: DocumentStore

    public init(documentStore: DocumentStore) {
        self.documentStore = documentStore
    }

    public func loadChatQueue(documentID: UUID) async throws -> ChatQueue {
        try await documentStore.loadChatQueue(documentID: documentID)
    }

    public func saveChatQueue(_ queue: ChatQueue, documentID: UUID) async throws {
        try await documentStore.saveChatQueue(queue, documentID: documentID)
    }

    public func loadDocument(id: UUID) async throws -> DocumentAST {
        try await documentStore.loadDocument(id: id)
    }

    public func history(of documentID: UUID, limit: Int) async throws -> [Receipt] {
        try await documentStore.history(of: documentID, limit: limit)
    }
}
