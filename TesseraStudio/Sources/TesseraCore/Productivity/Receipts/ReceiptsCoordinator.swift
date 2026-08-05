import Foundation

// MARK: - ReceiptsFocus

/// The current focus of the receipts drawer (per spec §7.3
/// + the cross-surface coordination in §15). The focus is
/// the receipt (or graph entity) the drawer is currently
/// showing in its detail view. The chat panel reads the
/// focus to decide whether to highlight the corresponding
/// applied item.
public enum ReceiptsFocus: Sendable, Hashable, Codable {
    /// No focus. The drawer is showing the chain list
    /// without a selected receipt.
    case none
    /// A receipt is focused. The id is the receipt's own
    /// id (which matches a `ChatQueueItem.producedReceiptID`
    /// for the corresponding chat-panel row).
    case receipt(UUID)
    /// A graph entity is focused. This is the Phase 6 hook
    /// — the Graph view takes over from the drawer.
    case graphEntity(UUID)

    public var receiptID: UUID? {
        if case .receipt(let id) = self { return id }
        return nil
    }

    public var graphEntityID: UUID? {
        if case .graphEntity(let id) = self { return id }
        return nil
    }
}

// MARK: - ReceiptsCoordinator

/// Cross-surface navigation state for the receipts drawer
/// + chat panel + (eventually) Graph view. The coordinator
/// is an `actor` for the cross-surface state, with an
/// `ObservableObject` companion (`ReceiptsCoordinatorBridge`)
/// that SwiftUI views observe on the main actor.
///
/// **Three navigation paths:**
/// - **Chat → drawer.** The chat panel's "applied" rows tap
///   a receipt chip; the chip fires `openReceiptInDrawer`,
///   the drawer's detail view appears.
/// - **Drawer → chat.** The drawer's detail view has a
///   "Show in chat" button that scrolls the chat panel to
///   the corresponding applied item.
/// - **Drawer → graph.** The drawer's detail view has a
///   "Show in graph" button that hands off to the Phase 6
///   Graph view (placeholder for now; the navigation
///   surface is wired but the Graph view itself is a later
///   phase).
///
/// The coordinator is also the registry the chat panel
/// uses to look up which chat item corresponds to a given
/// receipt (the `chatItem(forReceipt:)` path). This is a
/// per-document lookup; the coordinator is fed a lookup
/// function from the host view.
public actor ReceiptsCoordinator {

    // MARK: - State

    private var focus: ReceiptsFocus = .none
    private var drawerVisible: Bool = true
    private var scrollTarget: UUID?  // chat item id
    private var openRequest: OpenRequest?

    /// A pending "open" request that the host view will
    /// observe. The request is set when the user taps a
    /// receipt chip in the chat panel; the drawer's
    /// container view consumes the request and clears it
    /// once the drawer has shown the receipt.
    public struct OpenRequest: Sendable, Hashable {
        public let receiptID: UUID
        public let fromChatItemID: UUID?
        public let timestamp: Date
        public init(receiptID: UUID, fromChatItemID: UUID?, timestamp: Date = Date()) {
            self.receiptID = receiptID
            self.fromChatItemID = fromChatItemID
            self.timestamp = timestamp
        }
    }

    public init() {}

    // MARK: - Drawer visibility

    public var isDrawerVisible: Bool { drawerVisible }

    public func setDrawerVisible(_ visible: Bool) {
        drawerVisible = visible
    }

    public func toggleDrawerVisibility() {
        drawerVisible.toggle()
    }

    // MARK: - Open / scroll / focus

    /// Open a receipt in the drawer. Called by the chat
    /// panel when the user taps a chip. The drawer's
    /// container view observes `pendingOpenRequest()` to
    /// know when to show the receipt.
    public func openReceiptInDrawer(
        _ receiptID: UUID,
        fromChatItem itemID: UUID? = nil
    ) {
        focus = .receipt(receiptID)
        drawerVisible = true
        openRequest = OpenRequest(
            receiptID: receiptID,
            fromChatItemID: itemID
        )
    }

    /// Scroll the chat panel to the chat item that
    /// corresponds to a given receipt id. The host view
    /// is expected to provide a `chatItemLookup` so the
    /// coordinator can resolve the receipt id to a chat
    /// item id (the chat item's `producedReceiptID`
    /// matches the receipt id).
    public func showInChat(receiptID: UUID) async -> UUID? {
        // The host view is expected to wire the lookup; we
        // record the request and the host resolves it.
        // The lookup is async because the chat panel's
        // view-model is `@MainActor`-isolated.
        if let lookup = chatItemLookup, let resolved = await lookup(receiptID) {
            scrollTarget = resolved
            return resolved
        }
        scrollTarget = nil
        return nil
    }

    /// Hand off to the Graph view (Phase 6). The drawer
    /// container observes the focus and navigates.
    public func showInGraph(entityID: UUID) {
        focus = .graphEntity(entityID)
    }

    /// Clear the focus (e.g., when the drawer is closed).
    public func clearFocus() {
        focus = .none
        openRequest = nil
    }

    /// The current focus. The drawer's detail view reads
    /// this to know which receipt to show.
    public var currentFocus: ReceiptsFocus { focus }

    /// The pending open request. The drawer's container
    /// view consumes this when it appears.
    public func consumeOpenRequest() -> OpenRequest? {
        defer { openRequest = nil }
        return openRequest
    }

    /// The current scroll target. The chat panel reads
    /// this to know which item to scroll to.
    public var currentScrollTarget: UUID? { scrollTarget }

    /// Clear the scroll target. Called by the chat panel
    /// after it has scrolled.
    public func clearScrollTarget() {
        scrollTarget = nil
    }

    // MARK: - Lookup hook

    /// A function the host view provides to map a receipt
    /// id to the corresponding chat item id. The chat
    /// panel's view layer sets this on appearance and
    /// clears it on disappearance. The closure is async
    /// because the lookup may need to hop to the main
    /// actor to read the chat panel's `@Published` items.
    public var chatItemLookup: (@Sendable (UUID) async -> UUID?)?

    public func setChatItemLookup(_ lookup: (@Sendable (UUID) async -> UUID?)?) {
        chatItemLookup = lookup
    }
}

// MARK: - ReceiptsCoordinatorBridge

/// The SwiftUI-friendly companion to ``ReceiptsCoordinator``.
/// The coordinator is an actor; SwiftUI can't observe it
/// directly. The bridge polls the coordinator on a timer
/// and republishes the state as `@Published` properties.
///
/// The bridge is `@MainActor` so all `@Published` updates
/// happen on the main thread.
@MainActor
public final class ReceiptsCoordinatorBridge: ObservableObject {

    @Published public private(set) var isDrawerVisible: Bool = true
    @Published public private(set) var focus: ReceiptsFocus = .none
    @Published public private(set) var scrollTarget: UUID?
    @Published public private(set) var pendingOpenRequest: ReceiptsCoordinator.OpenRequest?

    private let coordinator: ReceiptsCoordinator
    private var refreshTask: Task<Void, Never>?

    public init(coordinator: ReceiptsCoordinator) {
        self.coordinator = coordinator
    }

    public func start() async {
        await refresh()
        refreshTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 200_000_000)
                await self?.refresh()
            }
        }
    }

    public func stop() {
        refreshTask?.cancel()
        refreshTask = nil
    }

    public func refresh() async {
        let visible = await coordinator.isDrawerVisible
        let focus = await coordinator.currentFocus
        let target = await coordinator.currentScrollTarget
        let open = await coordinator.consumeOpenRequest()
        self.isDrawerVisible = visible
        self.focus = focus
        self.scrollTarget = target
        if let open { self.pendingOpenRequest = open }
    }

    // MARK: - Action forwarding

    public func toggleDrawer() async {
        await coordinator.toggleDrawerVisibility()
        await refresh()
    }

    public func openReceipt(_ receiptID: UUID, fromChatItem itemID: UUID? = nil) async {
        await coordinator.openReceiptInDrawer(receiptID, fromChatItem: itemID)
        await refresh()
    }

    public func showInChat(receiptID: UUID) async -> UUID? {
        let result = await coordinator.showInChat(receiptID: receiptID)
        await refresh()
        return result
    }

    public func showInGraph(entityID: UUID) async {
        await coordinator.showInGraph(entityID: entityID)
        await refresh()
    }

    public func clearScrollTarget() async {
        await coordinator.clearScrollTarget()
        await refresh()
    }
}
