import Foundation

// MARK: - ChatQueueItem

/// One item in a document's chat queue. The chat panel (Phase 3)
/// is a per-document command queue, not a chat history: each
/// item transitions `pending -> inProgress -> applied` (or
/// `failed`) and carries a natural-language description plus
/// metadata about who produced it and what state it's in.
///
/// The queue is stored in the data layer (the `chat_queues` table
/// from migration `0002_productivity_receipts.sql`); Phase 1
/// provides the data model and serialization, Phase 3 wires the
/// SwiftUI list.
public struct ChatQueueItem: Codable, Sendable, Identifiable, Hashable {
    public let id: UUID
    public let documentID: UUID
    /// Position in the queue. Lower = closer to the front. The
    /// queue is NOT a strict array; the chat panel can reorder
    /// items by drag, and the order field is what the agent's
    /// context window reads.
    public var order: Int
    /// The natural-language description ("summarize section 2",
    /// "you edited paragraph 3"). For user-typed items, this is
    /// the raw text the user typed. For agent-typed items, this
    /// is the prompt that drove the agent's mutations.
    public var message: String
    /// Current state in the lifecycle (see ``State``).
    public var state: State
    /// The actor that produced this item.
    public var actor: Actor
    /// The mutation that produced this item (for user edits via
    /// the editor; the chat panel uses the message text directly).
    /// nil for chat-typed items.
    public var sourceMutation: Mutation?
    /// The receipt produced when this item was applied. nil while
    /// `state` is `.pending` or `.inProgress`.
    public var producedReceiptID: UUID?
    /// Creation time. Used to break ties when two items have the
    /// same `order` (e.g., the match-and-supersede rewrite).
    public var createdAt: Date
    /// If this item was superseded by another (the spec's
    /// match-and-supersede check), the id of the superseding
    /// item. nil when the item is still active.
    public var supersededByID: UUID?

    public init(
        id: UUID = UUID(),
        documentID: UUID,
        order: Int,
        message: String,
        state: State = .pending,
        actor: Actor,
        sourceMutation: Mutation? = nil,
        producedReceiptID: UUID? = nil,
        createdAt: Date = Date(),
        supersededByID: UUID? = nil
    ) {
        self.id = id
        self.documentID = documentID
        self.order = order
        self.message = message
        self.state = state
        self.actor = actor
        self.sourceMutation = sourceMutation
        self.producedReceiptID = producedReceiptID
        self.createdAt = createdAt
        self.supersededByID = supersededByID
    }

    public enum State: String, Codable, Sendable, Hashable, CaseIterable {
        /// User (or agent) just created the item. The agent has
        /// seen it in its context but has not started.
        case pending
        /// The agent is actively producing mutations from this item.
        case inProgress
        /// The agent finished and produced a receipt.
        case applied
        /// The agent failed (model error, validation failure, ...).
        /// The chat panel shows a retry button.
        case failed
    }

    // MARK: - Convenience

    /// True iff this item has been superseded by another (the
    /// spec's match-and-supersede check marked it). Superseded
    /// items stay in the queue (visible but dimmed) so the user
    /// can see the history of intent.
    public var isSuperseded: Bool { supersededByID != nil }

    /// The position the user sees in the chat panel: front of
    /// the queue is position 1 (one-based), so the first item is
    /// #1. Used by the "replaces #N" badge on superseded items.
    public func displayPosition(among items: [ChatQueueItem]) -> Int? {
        let ordered = items.sorted { lhs, rhs in
            if lhs.order != rhs.order { return lhs.order < rhs.order }
            return lhs.createdAt < rhs.createdAt
        }
        return ordered.firstIndex(where: { $0.id == self.id }).map { $0 + 1 }
    }
}

// MARK: - ChatQueue

/// The per-document chat queue. The `items` are kept in `order`
/// ascending (lowest order = front of queue). The data layer
/// stores the items as JSONB; this wrapper gives a typed
/// view that handles reordering, superseding, and state
/// transitions.
///
/// Phase 1 keeps this in memory; Phase 3 will surface it to
/// SwiftUI. The data layer persists the full `ChatQueue` (or
/// just `items` -- the brief's migration uses a separate table
/// keyed by `document_id`).
public struct ChatQueue: Codable, Sendable, Hashable {
    public var items: [ChatQueueItem]

    public init(items: [ChatQueueItem] = []) {
        self.items = items
    }

    public static let empty = ChatQueue()

    /// Items in queue order (front of queue first).
    public var orderedItems: [ChatQueueItem] {
        items.sorted { lhs, rhs in
            if lhs.order != rhs.order { return lhs.order < rhs.order }
            return lhs.createdAt < rhs.createdAt
        }
    }

    /// Insert a new item at the front of the queue, pushing all
    /// existing items back by 1. Returns the updated queue.
    public func insertingAtFront(_ item: ChatQueueItem) -> ChatQueue {
        // Shift everything by 1.
        let shifted = items.map { existing -> ChatQueueItem in
            var copy = existing
            copy.order += 1
            return copy
        }
        var inserted = item
        inserted.order = 0
        return ChatQueue(items: shifted + [inserted])
    }

    /// Reorder an item to a new position. The item's `order` is
    /// rewritten to land it at `newIndex` (0-based) in the
    /// ordered list. Other items' order fields are NOT
    /// renormalized -- the new orders may have gaps, but the
    /// relative order is preserved.
    public func reordering(itemID: UUID, to newIndex: Int) -> ChatQueue {
        guard let movingItemIdx = items.firstIndex(where: { $0.id == itemID }) else {
            return self
        }
        let ordered = orderedItems
        guard let currentIdx = ordered.firstIndex(where: { $0.id == itemID }) else {
            return self
        }
        let safeIndex = max(0, min(newIndex, ordered.count - 1))
        // Rebuild the ordered list with the item moved.
        var newOrdered = ordered
        let item = newOrdered.remove(at: currentIdx)
        newOrdered.insert(item, at: safeIndex)
        // Renumber compactly 0..n-1.
        let renumbered: [ChatQueueItem] = newOrdered.enumerated().map { pair in
            var item = pair.element
            item.order = pair.offset
            return item
        }
        return ChatQueue(items: renumbered)
    }

    /// Mark an item as superseded by another. The original item
    /// stays in the queue (visible but dimmed) so the user can
    /// see the history of intent.
    public func superseding(itemID: UUID, by supersederID: UUID) -> ChatQueue {
        var newItems = items
        if let idx = newItems.firstIndex(where: { $0.id == itemID }) {
            newItems[idx].supersededByID = supersederID
        }
        return ChatQueue(items: newItems)
    }

    /// Mark an item as in-progress. Throws nothing; the caller
    /// is expected to check that the item exists.
    public func starting(itemID: UUID) -> ChatQueue {
        var newItems = items
        if let idx = newItems.firstIndex(where: { $0.id == itemID }) {
            newItems[idx].state = .inProgress
        }
        return ChatQueue(items: newItems)
    }

    /// Mark an item as applied, recording the receipt id.
    public func finishing(itemID: UUID, with receiptID: UUID) -> ChatQueue {
        var newItems = items
        if let idx = newItems.firstIndex(where: { $0.id == itemID }) {
            newItems[idx].state = .applied
            newItems[idx].producedReceiptID = receiptID
        }
        return ChatQueue(items: newItems)
    }

    /// Mark an item as failed.
    public func failing(itemID: UUID) -> ChatQueue {
        var newItems = items
        if let idx = newItems.firstIndex(where: { $0.id == itemID }) {
            newItems[idx].state = .failed
        }
        return ChatQueue(items: newItems)
    }
}
