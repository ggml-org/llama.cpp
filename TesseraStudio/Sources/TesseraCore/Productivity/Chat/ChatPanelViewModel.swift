import Foundation
import SwiftUI

// MARK: - ChatPanelViewModel

/// The SwiftUI view-model for the chat panel. The model
/// bridges the actor-based `ChatPanelStateMachine` to the
/// `@MainActor` view layer: it polls the state machine
/// for the current envelope and exposes the data as
/// `@Published` properties that SwiftUI can observe.
///
/// The view-model is `@MainActor` so all `@Published`
/// updates happen on the main thread (SwiftUI's
/// expectation). The model is intentionally thin: it does
/// no business logic, it just reflects the state machine's
/// state and forwards user actions.
///
/// **Polling vs observation.** The state machine is an
/// actor; SwiftUI can't observe it directly. The model
/// polls on a `Timer.publish` (default 200ms) and also
/// polls on every `refresh()` call (which the views call
/// after a user action). For v1 this is fine — the chat
/// panel is a low-frequency view, and the polling cost is
/// negligible. v2 may add a `AsyncStream` from the state
/// machine for true event-driven updates.
@MainActor
public final class ChatPanelViewModel: ObservableObject {

    // MARK: - Published state

    @Published public private(set) var items: [ChatQueueItemDisplay] = []
    @Published public private(set) var holdMode: HoldMode = .running
    @Published public private(set) var receiptCount: Int = 0
    @Published public private(set) var inFlightItemID: UUID?
    @Published public private(set) var isLoaded: Bool = false
    @Published public private(set) var lastError: String?

    /// The current pending message text in the input field.
    /// The view binds to this; the model's `submit()`
    /// enqueues it.
    @Published public var inputText: String = ""

    /// The hold-your-horses dialog state. When non-nil, the
    /// dialog is shown.
    @Published public var holdDialog: HoldDialogState?

    public struct HoldDialogState: Sendable, Hashable {
        public let title: String
        public let message: String
        public init(title: String = "Hold your horses", message: String) {
            self.title = title
            self.message = message
        }
        public static let defaultMessage = "Is something wrong? Would you like me to reframe and approach things differently?"
    }

    // MARK: - Dependencies

    public let documentID: UUID
    public let documentTitle: String
    private let stateMachine: ChatPanelStateMachine
    private let crossDocRegistry: CrossDocumentChatRegistry?
    private let coordinator: ReceiptsCoordinator?

    private var refreshTask: Task<Void, Never>?

    public init(
        documentID: UUID,
        documentTitle: String,
        stateMachine: ChatPanelStateMachine,
        crossDocRegistry: CrossDocumentChatRegistry? = nil,
        coordinator: ReceiptsCoordinator? = nil
    ) {
        self.documentID = documentID
        self.documentTitle = documentTitle
        self.stateMachine = stateMachine
        self.crossDocRegistry = crossDocRegistry
        self.coordinator = coordinator
    }

    // MARK: - Lifecycle

    /// Load the chat queue from the data layer. Called once
    /// by the host view on appear.
    public func start() async {
        do {
            _ = try await stateMachine.load()
            isLoaded = true
            await refresh()
        } catch {
            lastError = "Failed to load chat queue: \(error)"
        }
    }

    /// Stop the polling task. Called by the host view on
    /// disappear.
    public func stop() {
        refreshTask?.cancel()
        refreshTask = nil
    }

    /// Refresh the model from the state machine. The view
    /// layer calls this after every user action; the
    /// background timer also calls it periodically.
    public func refresh() async {
        let envelope = await stateMachine.currentEnvelope()
        let count = await stateMachine.currentReceiptCount
        let inFlight = envelope.items.first(where: { $0.state == .inProgress })?.id
        let displays = envelope.items.map { item in
            ChatQueueItemDisplay.display(
                for: item,
                in: envelope.items,
                meta: envelope.meta[item.id] ?? .empty
            )
        }
        self.items = displays
        self.holdMode = envelope.holdMode
        self.receiptCount = count
        self.inFlightItemID = inFlight
    }

    /// Set the receipt count (called by the host view on
    /// document open, using the chain's row count).
    public func setReceiptCount(_ count: Int) async {
        await stateMachine.setReceiptCount(count)
        await refresh()
    }

    // MARK: - User actions

    /// Submit the current `inputText` as a new pending item.
    /// The text is trimmed; an empty message is a no-op.
    public func submit() async {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        inputText = ""
        do {
            _ = try await stateMachine.enqueue(message: text)
            await refresh()
        } catch {
            lastError = "Failed to enqueue: \(error)"
        }
    }

    /// Cancel the currently in-progress item. The agent
    /// is expected to discard its in-flight mutations.
    public func cancelInProgress() async {
        do {
            try await stateMachine.cancelInProgress()
            await refresh()
        } catch {
            lastError = "Failed to cancel: \(error)"
        }
    }

    /// Reorder an item to a new index (0-based).
    public func reorder(itemID: UUID, to newIndex: Int) async {
        do {
            try await stateMachine.reorder(itemID: itemID, toNewIndex: newIndex)
            await refresh()
        } catch {
            lastError = "Failed to reorder: \(error)"
        }
    }

    /// Delete an item.
    public func delete(itemID: UUID) async {
        do {
            try await stateMachine.delete(itemID: itemID)
            await refresh()
        } catch {
            lastError = "Failed to delete: \(error)"
        }
    }

    /// Undo a supersession (called by the drag-to-reorder
    /// override).
    public func unsupersede(itemID: UUID) async {
        do {
            try await stateMachine.unsupersede(itemID: itemID)
            await refresh()
        } catch {
            lastError = "Failed to unsupersede: \(error)"
        }
    }

    // MARK: - Hold your horses

    /// Pause the queue. Sets the hold mode to
    /// `.holdRequested`; the dialog is shown by the view.
    public func holdYourHorses() async {
        do {
            try await stateMachine.holdYourHorses()
            holdDialog = HoldDialogState(message: HoldDialogState.defaultMessage)
            await refresh()
        } catch {
            lastError = "Failed to pause: \(error)"
        }
    }

    /// Resume the queue. Closes the dialog.
    public func resume() async {
        do {
            try await stateMachine.resume()
            holdDialog = nil
            await refresh()
        } catch {
            lastError = "Failed to resume: \(error)"
        }
    }

    /// Pause all registered documents (the cross-doc
    /// "Pause all" button).
    public func pauseAll() async {
        guard let registry = crossDocRegistry else { return }
        await registry.pauseAll()
        await refresh()
    }

    // MARK: - Receipts

    /// Open a receipt in the drawer. Called by the chat
    /// panel's "applied" rows when the user taps the
    /// receipt chip.
    public func openReceiptInDrawer(receiptID: UUID, fromChatItem itemID: UUID? = nil) async {
        guard let coordinator = coordinator else { return }
        await coordinator.openReceiptInDrawer(receiptID, fromChatItem: itemID)
    }

    // MARK: - Convenience

    /// The list of items the user can drag to reorder.
    /// Applied items are excluded (they're in the audit
    /// trail and can't be reordered).
    public var draggableItems: [ChatQueueItemDisplay] {
        items.filter { $0.item.state != .applied }
    }

    /// The list of items the user can delete.
    public var deletableItems: [ChatQueueItemDisplay] {
        items.filter { $0.item.state != .applied }
    }

    /// The list of background documents (other than the
    /// current one) that have an active chat queue.
    public func backgroundDocuments() async -> [ActiveDocumentInfo] {
        guard let registry = crossDocRegistry else { return [] }
        let docs = await registry.activeDocuments()
        return docs.filter { $0.documentID != documentID }
    }
}
