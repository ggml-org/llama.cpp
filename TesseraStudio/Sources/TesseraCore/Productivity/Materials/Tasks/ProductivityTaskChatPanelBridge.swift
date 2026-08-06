import Foundation

// MARK: - ProductivityTaskChatPanelBridge

/// The bridge between the chat panel (Phase 3) and the
/// Tasks surface (Phase 5). The chat panel emits
/// ``ChatQueueItem`` values when the user types "add a
/// task to …"; the bridge turns the queue item into a
/// ``ProductivityTask``, calls ``ProductivityTaskStore`` to
/// persist it, and reports the new task id back so the
/// chat panel can show the receipt chip.
///
/// **Why a separate type:** the chat panel's existing
/// command path emits a `Mutation` (for documents); tasks
/// are a different kind of mutation, so we route them
/// through this bridge. The chat panel's view-model
/// recognises "task" as a command keyword and dispatches
/// here.
///
/// **Receipt integration:** every task creation through
/// this bridge produces a `task_created_from_chat`
/// receipt, distinct from the `task_upsert` receipt. The
/// receipt payload includes the chat item id so the
/// audit trail can answer "which chat command created
/// this task".
public struct ProductivityTaskChatPanelBridge: Sendable {

    private let store: ProductivityTaskStore

    public init(store: ProductivityTaskStore) {
        self.store = store
    }

    /// Create a task from a chat queue item. The
    /// `message` is the raw user text; the `parser` (if
    /// provided) extracts title, due date, priority, and
    /// linked entities. The `actor` is the chat user (the
    /// human, not the agent — the agent processes the
    /// queue item, but the human initiated the action).
    @discardableResult
    public func createTaskFromChat(
        chatItemID: UUID,
        documentID: UUID,
        message: String,
        actor: Actor = .user(UUID()),
        parser: ProductivityTaskNLUParser? = nil
    ) async throws -> ProductivityTask {
        let p = parser ?? ProductivityTaskNLUParser()
        let parsed = p.parse(message)
        let task = parsed.toTask()
        let saved = try await store.upsert(task, actor: actor)
        // Distinct receipt so the audit trail shows the
        // chat provenance.
        _ = try? await store.exposedDataLayer.appendReceipt(
            entityID: saved.id,
            receiptType: ProductivityTaskReceiptType.createdFromChat.rawValue,
            payload: [
                "chatItemID": .string(chatItemID.uuidString),
                "documentID": .string(documentID.uuidString),
                "rawMessage": .string(message),
            ]
        )
        return saved
    }
}
