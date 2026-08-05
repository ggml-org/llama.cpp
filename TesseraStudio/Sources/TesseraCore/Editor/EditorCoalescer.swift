import Foundation

// MARK: - EditorCoalescer

/// Coalesces a burst of user edits into a single
/// `Mutation` batch (per spec §5.5). The user typing ten
/// characters in a second produces ten text-view change
/// events; the coalescer aggregates them into one
/// `Mutation` batch so the chat queue only sees one
/// "you edited paragraph 3" item.
///
/// **Window.** A burst is a sequence of edits that arrive
/// within `coalesceWindow` seconds of each other. The
/// default is 1.5s; the user can configure 0.5-5.0s. When
/// the window expires (no new edits within `coalesceWindow`
/// seconds of the last one), the coalescer flushes its
/// pending mutations and emits a `ChatQueueItem`.
///
/// **Coalescing rules.** The coalescer does not concatenate
/// mutations across edits — the editor's text view already
/// produces a final state, so the reducer converts that
/// final state to a single `Mutation` (typically
/// `setBlockContent` for a typing burst, or
/// `setInlineAnnotation` for a format change). The
/// coalescer's job is to:
///   1. Hold the burst in memory until the window expires.
///   2. When the window expires, build a single `Mutation`
///      that represents the **cumulative** effect of the
///      burst (NOT the concatenation of the per-edit
///      mutations).
///   3. Emit a `ChatQueueItem` with the source mutation
///      attached.
///
/// **State.** The coalescer is a final class because the
/// coalescing window and the pending batch are mutable
/// state. The class is `Sendable` because the state is
/// guarded by a lock — coalescing runs on the main thread
/// (text-view change events arrive on the main thread) but
/// the API is `async`-friendly so the chat-queue write can
/// happen off the main thread.
public final class EditorCoalescer: @unchecked Sendable {

    public struct Settings: Codable, Sendable, Hashable {
        public var coalesceWindow: TimeInterval
        public init(coalesceWindow: TimeInterval = 1.5) {
            // Clamp to the architect-specified range.
            self.coalesceWindow = min(max(coalesceWindow, 0.5), 5.0)
        }
        public static let `default` = Settings()
    }

    /// One coalesced burst. The caller persists the
    /// `mutations` via `DocumentStore.applyBatch` and the
    /// `queueItem` via `DocumentStore.saveChatQueue` after
    /// the burst flushes.
    public struct CoalescedBurst: Sendable {
        public let documentID: UUID
        public let blockID: UUID
        public let mutations: [Mutation]
        public let queueItem: ChatQueueItem
        public let startedAt: Date
        public let flushedAt: Date
    }

    /// The coalescer's notification. Posted on the main
    /// thread when a burst flushes. The notification's
    /// `userInfo["burst"]` carries the `CoalescedBurst`.
    public static let didFlushNotification = Notification.Name("TesseraStudio.EditorCoalescer.didFlush")

    // MARK: - State

    private let lock = NSLock()
    private var settings: Settings
    private var documentID: UUID?
    private var pendingBlockID: UUID?
    private var pendingMutations: [Mutation] = []
    private var pendingStart: Date?
    private var pendingLastUpdate: Date?
    private var pendingQueueMessage: String?
    /// Timer for the coalescing window. The coalescer uses
    /// a wall-clock timer (`DispatchSourceTimer`) rather than
    /// `Task.sleep` because the editor's text-view events
    /// arrive on the main run loop.
    private var flushTimer: DispatchSourceTimer?
    private let queue = DispatchQueue(label: "tessera.editor.coalescer", qos: .userInteractive)

    public init(settings: Settings = .default) {
        self.settings = settings
    }

    public var coalesceWindow: TimeInterval {
        lock.lock(); defer { lock.unlock() }
        return settings.coalesceWindow
    }

    public func updateSettings(_ new: Settings) {
        lock.lock(); defer { lock.unlock() }
        settings = new
    }

    // MARK: - Append

    /// Append a new edit to the pending burst. The coalescer
    /// decides whether to coalesce (the edit is in the same
    /// block as the pending burst and within the window) or
    /// start a fresh burst.
    ///
    /// - Parameters:
    ///   - mutation: the new mutation from the editor.
    ///   - blockID: the block the mutation is in.
    ///   - documentID: the document the block is in.
    ///   - queueMessage: the natural-language description
    ///     that will be attached to the resulting
    ///     `ChatQueueItem` ("You edited paragraph 3").
    public func append(
        mutation: Mutation,
        blockID: UUID,
        documentID: UUID,
        queueMessage: String
    ) {
        lock.lock()
        let now = Date()
        let isSameDocument = self.documentID == documentID
        let isSameBlock = self.pendingBlockID == blockID
        let isWithinWindow: Bool
        if let last = pendingLastUpdate {
            isWithinWindow = now.timeIntervalSince(last) <= settings.coalesceWindow
        } else {
            isWithinWindow = false
        }
        if isSameDocument && isSameBlock && isWithinWindow {
            // Coalesce: keep the most recent mutation (which
            // represents the post-edit state). The previous
            // pending mutations are stale — the editor's text
            // view has already applied them; the most recent
            // one captures the final state.
            pendingMutations = [mutation]
            pendingLastUpdate = now
            pendingQueueMessage = queueMessage
        } else {
            // Flush the previous burst (if any) and start a
            // new one. We do this synchronously so the
            // caller sees a consistent state.
            if let _ = pendingStart {
                lock.unlock()
                flush()
                lock.lock()
            }
            self.documentID = documentID
            self.pendingBlockID = blockID
            self.pendingMutations = [mutation]
            self.pendingStart = now
            self.pendingLastUpdate = now
            self.pendingQueueMessage = queueMessage
        }
        // (Re)arm the flush timer.
        scheduleFlushLocked()
        lock.unlock()
    }

    // MARK: - Flush

    /// Force-flush the pending burst. The caller uses this
    /// when the editor loses focus (so the burst isn't lost
    /// if the user closes the document) or when the user
    /// switches documents.
    @discardableResult
    public func flush() -> CoalescedBurst? {
        lock.lock()
        guard let documentID,
              let pendingBlockID,
              let startedAt = pendingStart,
              !pendingMutations.isEmpty,
              let queueMessage = pendingQueueMessage else {
            cancelTimerLocked()
            clearPendingLocked()
            lock.unlock()
            return nil
        }
        let mutations = pendingMutations
        let now = Date()
        let item = ChatQueueItem(
            documentID: documentID,
            order: 0,
            message: queueMessage,
            state: .applied,
            actor: .user(UUID()),
            sourceMutation: mutations.first,
            createdAt: now
        )
        let burst = CoalescedBurst(
            documentID: documentID,
            blockID: pendingBlockID,
            mutations: mutations,
            queueItem: item,
            startedAt: startedAt,
            flushedAt: now
        )
        cancelTimerLocked()
        clearPendingLocked()
        lock.unlock()
        NotificationCenter.default.post(
            name: Self.didFlushNotification,
            object: self,
            userInfo: ["burst": burst]
        )
        return burst
    }

    /// True iff the coalescer has a pending burst. Used by
    /// the editor to know whether to flush on focus loss.
    public var hasPending: Bool {
        lock.lock(); defer { lock.unlock() }
        return pendingStart != nil && !pendingMutations.isEmpty
    }

    // MARK: - Internals

    private func scheduleFlushLocked() {
        cancelTimerLocked()
        let window = settings.coalesceWindow
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + window)
        timer.setEventHandler { [weak self] in
            self?.flush()
        }
        timer.resume()
        flushTimer = timer
    }

    private func cancelTimerLocked() {
        flushTimer?.cancel()
        flushTimer = nil
    }

    private func clearPendingLocked() {
        documentID = nil
        pendingBlockID = nil
        pendingMutations = []
        pendingStart = nil
        pendingLastUpdate = nil
        pendingQueueMessage = nil
    }
}
