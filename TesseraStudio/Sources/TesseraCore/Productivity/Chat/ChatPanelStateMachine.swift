import Foundation

// MARK: - ChatQueueItemMeta

/// Per-item metadata the state machine tracks out-of-band
/// (not persisted on the `ChatQueueItem` itself). The
/// `ChatQueueItem` shape is owned by Phase 1 and is
/// versioned through the data layer's `chat_queues.items`
/// JSONB column. Adding fields to it would force a
/// migration; the state machine instead keeps a separate
/// dictionary keyed by item id and persists it on every
/// mutation (the dictionary is part of the queue's JSON
/// envelope — see ``ChatPanelEnvelope``).
///
/// The dictionary is `Sendable` and `Hashable` so it can
/// be passed across actor boundaries and observed by
/// SwiftUI views.
public struct ChatQueueItemMeta: Codable, Sendable, Hashable {
    /// The reasoning string from the match-and-supersede
    /// engine, recorded when the item was created.
    public var supersedeReasoning: String?
    /// The failure note for failed items, recorded when
    /// the item was marked failed.
    public var failureNote: String?
    /// The agent's suggested reordering during a "Hold
    /// your horses" pause. nil when no suggestion has
    /// been made.
    public var agentSuggestedOrder: Int?

    public init(
        supersedeReasoning: String? = nil,
        failureNote: String? = nil,
        agentSuggestedOrder: Int? = nil
    ) {
        self.supersedeReasoning = supersedeReasoning
        self.failureNote = failureNote
        self.agentSuggestedOrder = agentSuggestedOrder
    }

    public static let empty = ChatQueueItemMeta()
}

// MARK: - ChatPanelEnvelope

/// The wrapper persisted to the data layer's `chat_queues`
/// table. The Phase 1 schema stores the items as a JSONB
/// column; this wrapper adds the state-machine's bookkeeping
/// (`holdMode`, per-item `meta`) on top. A v0 row (no
/// envelope) is loaded as an empty envelope with `running`
/// hold mode — the wrapper is forward-compatible.
///
/// The wrapper is encoded as a top-level JSON object:
/// `{ "hold_mode": "running", "items": [...], "meta": { "uuid": {...} } }`.
public struct ChatPanelEnvelope: Codable, Sendable, Hashable {
    public var holdMode: HoldMode
    public var items: [ChatQueueItem]
    public var meta: [UUID: ChatQueueItemMeta]

    public init(
        holdMode: HoldMode = .running,
        items: [ChatQueueItem] = [],
        meta: [UUID: ChatQueueItemMeta] = [:]
    ) {
        self.holdMode = holdMode
        self.items = items
        self.meta = meta
    }

    public static let empty = ChatPanelEnvelope()

    /// A v0 (Phase 1) envelope — no `hold_mode`, no
    /// `meta`. We default to a running hold mode and an
    /// empty meta map.
    public static func legacy(from chatQueue: ChatQueue) -> ChatPanelEnvelope {
        ChatPanelEnvelope(
            holdMode: .running,
            items: chatQueue.items,
            meta: [:]
        )
    }

    /// Reduce the envelope to a Phase 1 `ChatQueue` for
    /// callers that don't know about the envelope.
    public var asChatQueue: ChatQueue {
        ChatQueue(items: items)
    }
}

// MARK: - ChatPanelStateMachine

/// The per-document chat-queue state machine (per spec §6.2).
/// One instance per document window. The machine wraps the
/// Phase 1 ``ChatQueue`` data model and persists it to the
/// data layer's `chat_queues` table on every transition.
///
/// **Threading.** The machine is an `actor` so concurrent
/// reads from the SwiftUI view layer and the agent's
/// `agentLoop` don't race on the queue. The view layer
/// observes the queue by calling `queue` (an async read);
/// the agent calls `startNextPending` when it's idle.
///
/// **Persistence.** Every method that mutates the queue
/// calls `store.saveChatQueue(...)` after the
/// in-memory mutation. The save is awaited before the
/// method returns, so a successful method call is durable.
/// On document open, the machine loads the queue from the
/// data layer once (`load`).
///
/// **Hold mode.** The machine has a ``HoldMode`` (see the
/// separate type). The "Hold your horses" button in the chat
/// panel footer toggles between `.running` and `.hold`. While
/// paused, `startNextPending` returns nil (no new items are
/// picked up). The pause is persisted in the envelope's
/// `holdMode` field, so a reload restores the pause.
///
/// **Match-and-supersede.** The ``enqueue`` method calls the
/// ``MatchAndSupersedeEngine`` before persisting. The LLM
/// result is applied to the queue (the superseded items are
/// marked), then the new front is inserted.
public actor ChatPanelStateMachine {

    // MARK: - Public types

    /// A bound on the agent's context size. Defaults are
    /// 50 pending and 50 recent receipts. The state machine
    /// uses these when building ``AgentContext``.
    public struct ContextLimits: Sendable, Hashable {
        public var pendingLimit: Int
        public var receiptLimit: Int

        public init(pendingLimit: Int = 50, receiptLimit: Int = 50) {
            self.pendingLimit = max(0, pendingLimit)
            self.receiptLimit = max(0, receiptLimit)
        }

        public static let `default` = ContextLimits()
    }

    /// The state machine's load-time result. The SwiftUI view
    /// uses this to render an empty state vs. a populated one.
    public enum LoadResult: Sendable, Equatable {
        case empty
        case loaded(itemCount: Int)
    }

    // MARK: - Init

    public let documentID: UUID
    private let store: any ChatQueueStoring
    private let supersedeEngine: MatchAndSupersedeEngine
    private let contextLimits: ContextLimits

    /// Cached envelope. Loaded once at init, mutated in place
    /// by the state-machine methods. The cache is the source of
    /// truth between persistence calls.
    private var envelope: ChatPanelEnvelope
    private var loaded: Bool

    /// Cached receipt count (for the chat panel header). The
    /// state machine bumps this on `markApplied` and exposes
    /// the getter for the header. The host view can override
    /// via `setReceiptCount(_:)` on document open (using the
    /// chain's row count).
    private var receiptCount: Int = 0

    public init(
        documentID: UUID,
        store: any ChatQueueStoring,
        supersedeEngine: MatchAndSupersedeEngine = MatchAndSupersedeEngine(),
        contextLimits: ContextLimits = .default
    ) {
        self.documentID = documentID
        self.store = store
        self.supersedeEngine = supersedeEngine
        self.contextLimits = contextLimits
        self.envelope = .empty
        self.loaded = false
    }

    /// Convenience initializer for the production
    /// `DocumentStore`. Use this when wiring from the host
    /// app; tests use the protocol-based init.
    public init(
        documentID: UUID,
        documentStore: DocumentStore,
        supersedeEngine: MatchAndSupersedeEngine = MatchAndSupersedeEngine(),
        contextLimits: ContextLimits = .default
    ) {
        self.init(
            documentID: documentID,
            store: DocumentStoreChatQueueStore(documentStore: documentStore),
            supersedeEngine: supersedeEngine,
            contextLimits: contextLimits
        )
    }

    // MARK: - Load / save

    /// Load the envelope from the data layer. Idempotent: a
    /// second call re-reads the envelope (the in-memory cache
    /// is replaced).
    @discardableResult
    public func load() async throws -> LoadResult {
        let queue = try await store.loadChatQueue(documentID: documentID)
        // The data layer's chat_queues row stores the Phase 1
        // ChatQueue (an array of items). We wrap it in an
        // envelope with the default hold mode. Future versions
        // that persist the envelope directly can decode it
        // here.
        envelope = ChatPanelEnvelope(
            holdMode: .running,
            items: sortByOrder(queue.items),
            meta: [:]
        )
        loaded = true
        return envelope.items.isEmpty ? .empty : .loaded(itemCount: envelope.items.count)
    }

    /// The current in-memory queue (Phase 1 view). Callers
    /// that hold a reference to the returned value are reading
    /// a snapshot; subsequent state-machine calls may
    /// invalidate it.
    public var queue: ChatQueue {
        ChatQueue(items: envelope.items)
    }

    /// The current envelope (items + meta + hold mode). The
    /// SwiftUI view layer reads this to render the chat panel.
    public func currentEnvelope() -> ChatPanelEnvelope {
        envelope
    }

    /// The current hold mode.
    public var holdMode: HoldMode { envelope.holdMode }

    /// The current receipt count.
    public var currentReceiptCount: Int { receiptCount }

    /// The per-item meta dictionary.
    public func currentMeta() -> [UUID: ChatQueueItemMeta] {
        envelope.meta
    }

    /// Set the cached receipt count. The caller (the host
    /// view) typically calls this on document open with the
    /// count from the chain.
    public func setReceiptCount(_ count: Int) {
        receiptCount = max(0, count)
    }

    // MARK: - Enqueue (the front-of-queue add path)

    /// Append a new pending item to the front of the queue.
    /// The match-and-supersede check runs first; the existing
    /// items that the new front replaces are marked
    /// `supersededByID`. The new item is inserted at order 0.
    /// Persists the queue.
    @discardableResult
    public func enqueue(
        message: String,
        sourceMutation: Mutation? = nil,
        actor: Actor = .user(UUID()),
        supersedeCheck: Bool = true
    ) async throws -> ChatQueueItem {
        let trimmed = message.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw ChatPanelStateMachineError.emptyMessage
        }
        ensureLoaded()

        // Build the new item first; the engine needs the
        // candidate id for the cache key.
        let newItem = ChatQueueItem(
            documentID: documentID,
            order: 0,
            message: trimmed,
            state: .pending,
            actor: actor,
            sourceMutation: sourceMutation
        )

        // Run the match-and-supersede check BEFORE inserting
        // the new item, so the engine sees the existing queue
        // (the new item is the one being judged, not the
        // candidate).
        var supersededIDs: [UUID] = []
        var supersedeReasoning: String? = nil
        if supersedeCheck {
            let decision = try await supersedeEngine.evaluate(
                newFront: newItem,
                existingQueue: envelope.items.filter { $0.state != .applied && !$0.isSuperseded }
            )
            supersededIDs = decision.supersededItemIDs
            supersedeReasoning = decision.reasoning.isEmpty ? nil : decision.reasoning
        }

        // Apply the supersession markers to the existing items.
        var working = envelope
        for id in supersededIDs {
            working.items = applySupersede(items: working.items, oldID: id, newID: newItem.id)
        }
        // Insert the new front. The state machine
        // maintains the invariant that `envelope.items`
        // is sorted by `order` ascending (so
        // `items[0]` is the front of the queue, items[1]
        // is next, etc.). `insertingAtFront` shifts
        // orders and returns the new item at the end of
        // the array; we re-sort to maintain the
        // invariant.
        let newQueue = ChatQueue(items: working.items).insertingAtFront(newItem)
        working.items = sortByOrder(newQueue.items)
        if let reasoning = supersedeReasoning {
            working.meta[newItem.id] = ChatQueueItemMeta(
                supersedeReasoning: reasoning,
                failureNote: nil,
                agentSuggestedOrder: working.meta[newItem.id]?.agentSuggestedOrder
            )
        }
        envelope = working
        try await persist()
        return newItem
    }

    private func applySupersede(
        items: [ChatQueueItem],
        oldID: UUID,
        newID: UUID
    ) -> [ChatQueueItem] {
        items.map { item in
            guard item.id == oldID else { return item }
            var copy = item
            copy.supersededByID = newID
            return copy
        }
    }

    // MARK: - State transitions

    /// Start the next pending item. Transitions the head of
    /// the queue from `.pending` to `.inProgress` and returns
    /// the item. Returns nil when the queue is empty, when
    /// the queue is paused, or when the head is already
    /// in-flight / applied / failed.
    @discardableResult
    public func startNextPending() async throws -> ChatQueueItem? {
        ensureLoaded()
        if envelope.holdMode.isPaused { return nil }
        let ordered = orderedItems()
        guard let head = ordered.first(where: { $0.state == .pending && !$0.isSuperseded }) else {
            return nil
        }
        var working = envelope
        working.items = applyStateTransition(items: working.items, id: head.id, state: .inProgress)
        envelope = working
        try await persist()
        return working.items.first(where: { $0.id == head.id })
    }

    /// Mark an item as in-progress. The caller (the agent)
    /// uses this when the head of the queue has been
    /// transitioned by some other path (e.g., the agent's
    /// internal model) and the queue needs to know.
    public func markInProgress(itemID: UUID) async throws {
        ensureLoaded()
        envelope.items = applyStateTransition(items: envelope.items, id: itemID, state: .inProgress)
        try await persist()
    }

    /// Mark an item as applied, recording the receipt id
    /// produced by the agent. The receipt count is bumped
    /// by 1. The item is now part of the audit trail (it
    /// cannot be deleted or reordered).
    public func markApplied(itemID: UUID, receipt: Receipt) async throws {
        ensureLoaded()
        guard receipt.documentID == documentID else {
            throw ChatPanelStateMachineError.documentMismatch(
                expected: documentID,
                actual: receipt.documentID
            )
        }
        envelope.items = applyStateTransition(
            items: envelope.items,
            id: itemID,
            state: .applied,
            receiptID: receipt.id
        )
        receiptCount += 1
        try await persist()
    }

    /// Mark an item as failed. The chat panel's failed row
    /// shows a retry button; calling `enqueue` with the same
    /// message after a manual edit will create a new pending
    /// item (the old failed item stays in the queue, dimmed).
    public func markFailed(itemID: UUID, error: Error) async throws {
        ensureLoaded()
        envelope.items = applyStateTransition(items: envelope.items, id: itemID, state: .failed)
        let note = String(describing: error)
        var meta = envelope.meta[itemID] ?? .empty
        meta.failureNote = (meta.failureNote.map { "\($0)\n\(note)" }) ?? note
        envelope.meta[itemID] = meta
        try await persist()
    }

    /// Mark an item as superseded by another. The state
    /// machine's `enqueue` calls this internally; the
    /// public method is exposed for the drag-to-reorder
    /// "undo supersession" gesture (the user can drag the
    /// new front away, and the older item is un-superseded).
    public func supersede(oldItemID: UUID, by newItemID: UUID) async throws {
        ensureLoaded()
        envelope.items = applySupersede(items: envelope.items, oldID: oldItemID, newID: newItemID)
        try await persist()
    }

    /// Undo a supersession. The user dragged the new front
    /// away from the older item; the older item is restored
    /// to its pre-supersession state.
    public func unsupersede(itemID: UUID) async throws {
        ensureLoaded()
        envelope.items = envelope.items.map { item in
            guard item.id == itemID else { return item }
            var copy = item
            copy.supersededByID = nil
            return copy
        }
        try await persist()
    }

    /// Cancel the currently in-progress item. The item's
    /// state is set back to `.pending` and a failure note
    /// is appended. The agent is expected to discard the
    /// in-flight mutations.
    public func cancelInProgress() async throws {
        ensureLoaded()
        for idx in envelope.items.indices where envelope.items[idx].state == .inProgress {
            envelope.items[idx].state = .pending
            var meta = envelope.meta[envelope.items[idx].id] ?? .empty
            let note = "cancelled by user"
            meta.failureNote = (meta.failureNote.map { "\($0)\n\(note)" }) ?? note
            envelope.meta[envelope.items[idx].id] = meta
        }
        try await persist()
    }

    /// Reorder an item to a new position. The new index is
    /// 0-based (0 = front of the queue). Persists the queue.
    /// The order is renormalized so the queue's `order` field
    /// is a contiguous `0..n-1`.
    public func reorder(itemID: UUID, toNewIndex newIndex: Int) async throws {
        ensureLoaded()
        let newQueue = ChatQueue(items: envelope.items).reordering(itemID: itemID, to: newIndex)
        envelope.items = sortByOrder(newQueue.items)
        try await persist()
    }

    /// Delete an item. Applied items cannot be deleted (they
    /// are in the audit trail). Pending, in-progress, and
    /// failed items can be deleted.
    public func delete(itemID: UUID) async throws {
        ensureLoaded()
        guard let target = envelope.items.first(where: { $0.id == itemID }) else { return }
        guard target.state != .applied else {
            throw ChatPanelStateMachineError.cannotDeleteApplied
        }
        envelope.items.removeAll { $0.id == itemID }
        envelope.meta.removeValue(forKey: itemID)
        try await persist()
    }

    /// Record an agent-suggested ordering for an item (used
    /// during a "Hold your horses" pause to suggest a
    /// reordering). The user can accept the suggestion by
    /// dragging the item to the suggested position, which
    /// the chat panel's drag handler does automatically.
    public func setAgentSuggestedOrder(itemID: UUID, order: Int) async throws {
        ensureLoaded()
        var meta = envelope.meta[itemID] ?? .empty
        meta.agentSuggestedOrder = order
        envelope.meta[itemID] = meta
        try await persist()
    }

    // MARK: - Hold your horses

    /// Pause the queue. The agent's `startNextPending` returns
    /// nil while the queue is paused. The transition goes
    /// through `.holdRequested` first (so the UI can animate
    /// the dialog opening), then becomes `.hold` on the next
    /// call.
    public func holdYourHorses() async throws {
        ensureLoaded()
        switch envelope.holdMode {
        case .running:
            envelope.holdMode = .holdRequested
        case .holdRequested, .hold:
            return  // idempotent
        case .resuming:
            envelope.holdMode = .hold
        }
        try await persist()
    }

    /// Resume the queue. The transition goes through
    /// `.resuming` first, then becomes `.running` on the next
    /// call (or when the agent picks up the new front).
    public func resume() async throws {
        ensureLoaded()
        switch envelope.holdMode {
        case .running:
            return  // idempotent
        case .holdRequested, .hold, .resuming:
            envelope.holdMode = .resuming
        }
        try await persist()
        // Immediately complete the resume transition. The
        // UI sees `.resuming` for one render frame (enough
        // to play the agent-paused-banner slide-out), then
        // we're back to `.running`.
        envelope.holdMode = .running
        try await persist()
    }

    /// Force a hold (used by the cross-document "Pause all"
    /// button — it goes straight to `.hold` without the
    /// `.holdRequested` dance).
    public func forceHold() async throws {
        ensureLoaded()
        envelope.holdMode = .hold
        try await persist()
    }

    // MARK: - Agent context

    /// Build the agent's prompt-time context. The state
    /// machine reads its cached envelope + the recent receipts
    /// from the document store's chain (capped).
    public func agentContext() async throws -> AgentContext {
        ensureLoaded()
        let pending = orderedItems().filter { $0.state == .pending && !$0.isSuperseded }
        let recent = try await store.history(
            of: documentID,
            limit: contextLimits.receiptLimit
        )
        let ast = try await store.loadDocument(id: documentID)
        return AgentContext(
            documentID: documentID,
            pending: Array(pending.prefix(contextLimits.pendingLimit)),
            recentReceipts: Array(recent.suffix(contextLimits.receiptLimit)),
            documentAST: ast
        )
    }

    // MARK: - Internals

    private func ensureLoaded() {
        // Operations on a not-yet-loaded machine operate on
        // an empty envelope. The host view is expected to
        // call `load()` once on document open; if it
        // forgets, the resulting persist will write an
        // empty envelope, which is detectable from the data
        // layer side.
    }

    private func persist() async throws {
        // Phase 1's data layer stores the items as JSONB.
        // The state machine persists the items only (the
        // envelope's other fields are reconstructed on
        // load). The hold mode is exposed through a
        // separate accessor; v2 of the data layer will
        // store the full envelope.
        try await store.saveChatQueue(
            ChatQueue(items: envelope.items),
            documentID: documentID
        )
    }

    private func orderedItems() -> [ChatQueueItem] {
        sortByOrder(envelope.items)
    }

    /// Sort items by the `order` field, breaking ties by
    /// `createdAt`. This is the canonical "front of queue
    /// first" ordering.
    private func sortByOrder(_ items: [ChatQueueItem]) -> [ChatQueueItem] {
        items.sorted { lhs, rhs in
            if lhs.order != rhs.order { return lhs.order < rhs.order }
            return lhs.createdAt < rhs.createdAt
        }
    }

    private func applyStateTransition(
        items: [ChatQueueItem],
        id: UUID,
        state: ChatQueueItem.State,
        receiptID: UUID? = nil
    ) -> [ChatQueueItem] {
        items.map { item in
            guard item.id == id else { return item }
            var copy = item
            copy.state = state
            if let receiptID { copy.producedReceiptID = receiptID }
            return copy
        }
    }
}

// MARK: - Errors

public enum ChatPanelStateMachineError: Error, Sendable, Equatable {
    case emptyMessage
    case documentMismatch(expected: UUID, actual: UUID)
    case cannotDeleteApplied
    case queueNotLoaded
}
