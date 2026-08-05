import SwiftUI
import AppKit
import TesseraCore

// MARK: - ProductivitySurfaceView

/// The Phase 3 productivity surface host. Composes the
/// editor (Phase 2), the chat panel, and the receipt
/// drawer into a three-column layout. Phase 5 will wire
/// the per-Materials wrappers (Documents / Notes / Code)
/// around this host.
///
/// **macOS layout.** A `NavigationSplitView` with three
/// columns: surfaces | editor | chat. The receipt drawer
/// is a sibling inspector pane inside the chat column.
///
/// **Keyboard shortcuts.** Cmd-2 focuses the chat panel
/// (per spec §6.1, it does NOT toggle visibility).
/// Cmd-Option-2 toggles the receipts drawer (per spec
/// §7.3). Both shortcuts are wired through the focused
/// scene value so they work from the menu bar.
///
/// **Per-document wiring.** The host takes a
/// `ProductivitySurfaceModel` that owns the per-document
/// state machine, the receipt store, and the
/// cross-document registry. The model is a
/// `@MainActor` `ObservableObject` that the host view
/// observes.
public struct ProductivitySurfaceView: View {

    @ObservedObject public var model: ProductivitySurfaceModel
    public let documentID: UUID
    public let documentTitle: String

    @FocusState private var chatPanelFocused: Bool

    public init(
        model: ProductivitySurfaceModel,
        documentID: UUID,
        documentTitle: String
    ) {
        self.model = model
        self.documentID = documentID
        self.documentTitle = documentTitle
    }

    public var body: some View {
        NavigationSplitView {
            surfacesList
                .frame(minWidth: 180, idealWidth: 220)
        } content: {
            editorColumn
                .frame(minWidth: 360)
        } detail: {
            HStack(spacing: 0) {
                chatColumn
                if model.showReceiptsDrawer {
                    Divider()
                    receiptsColumn
                        .frame(minWidth: 320, idealWidth: 380)
                        .transition(.move(edge: .trailing))
                }
            }
        }
        .navigationTitle(documentTitle)
        .frame(minWidth: 900, minHeight: 560)
        .focusedSceneValue(\.productivityActions, ProductivityActions(
            focusChat: { focusChat() },
            toggleReceipts: { toggleReceipts() }
        ))
        .focusable(true)
        .onAppear { Task { await model.start() } }
        .onDisappear { model.stop() }
        .onChange(of: model.scrollTarget) { _, target in
            // The chat panel observes the scroll target
            // through the model; the focus is fired here so
            // the user can interact with the panel after
            // the chat panel forwards a request.
            chatPanelFocused = true
        }
    }

    // MARK: - Columns

    private var surfacesList: some View {
        List {
            Section("Surfaces") {
                Label("Documents", systemImage: "doc.text")
                Label("Notes", systemImage: "note.text")
                Label("Code", systemImage: "chevron.left.forwardslash.chevron.right")
            }
            Section("Current") {
                Label(documentTitle, systemImage: "doc")
            }
        }
        .listStyle(.sidebar)
    }

    private var editorColumn: some View {
        VStack(spacing: 0) {
            // The Phase 2 editor is rendered with a stub
            // binding for the demo; the production wiring
            // happens in Phase 5.
            TesseraEditorView(
                mode: .document,
                theme: .light,
                document: .constant(DocumentAST.empty),
                onMutationCommitted: { _, _ in }
            )
        }
    }

    private var chatColumn: some View {
        ChatPanelView(
            viewModel: model.chatPanelViewModel,
            canUndo: model.canUndo,
            canRedo: model.canRedo,
            onUndo: { model.undo() },
            onRedo: { model.redo() },
            onSwitchToDocument: { id in
                Task { await model.switchToDocument(id) }
            },
            onOpenReceipt: { id in
                Task { await model.openReceiptInDrawer(id) }
            }
        )
        .focused($chatPanelFocused)
    }

    private var receiptsColumn: some View {
        ReceiptsDrawerView(
            documentID: documentID,
            documentTitle: documentTitle,
            documentStore: model.documentStore,
            service: model.exportService,
            userID: model.userID,
            bridge: model.coordinatorBridge
        )
    }

    // MARK: - Actions

    private func focusChat() {
        chatPanelFocused = true
    }

    private func toggleReceipts() {
        withAnimation { model.showReceiptsDrawer.toggle() }
    }
}

// MARK: - ProductivityActions

/// The focused-scene action container for the
/// productivity surface. The `View > Focus Chat` and
/// `View > Toggle Receipts` menu items call these
/// closures.
public struct ProductivityActions: Sendable {
    public let focusChat: @MainActor () -> Void
    public let toggleReceipts: @MainActor () -> Void

    public init(
        focusChat: @escaping @MainActor () -> Void,
        toggleReceipts: @escaping @MainActor () -> Void
    ) {
        self.focusChat = focusChat
        self.toggleReceipts = toggleReceipts
    }
}

// MARK: - FocusedSceneValues extension

private struct ProductivityActionsKey: FocusedValueKey {
    typealias Value = ProductivityActions
}

extension FocusedValues {
    public var productivityActions: ProductivityActions? {
        get { self[ProductivityActionsKey.self] }
        set { self[ProductivityActionsKey.self] = newValue }
    }
}

// MARK: - ProductivitySurfaceModel

/// The view-model for the productivity surface. The
/// model owns the per-document state machine, the
/// receipt store, the cross-document registry, the
/// coordinator bridge, and the chat panel view-model.
///
/// The model is `@MainActor` so all `@Published`
/// updates happen on the main thread. The state machine
/// is the only piece that runs off the main thread.
@MainActor
public final class ProductivitySurfaceModel: ObservableObject {

    // MARK: - Published state

    @Published public var showReceiptsDrawer: Bool = false
    @Published public var canUndo: Bool = false
    @Published public var canRedo: Bool = false
    @Published public var scrollTarget: UUID?
    @Published public var activeDocumentID: UUID

    // MARK: - Dependencies (public for the host to wire)

    public let documentStore: DocumentStore
    public let dataLayer: TesseraDataLayer
    public let userID: UserID

    public let chatPanelViewModel: ChatPanelViewModel
    public let coordinator: ReceiptsCoordinator
    public let coordinatorBridge: ReceiptsCoordinatorBridge
    public let crossDocRegistry: CrossDocumentChatRegistry
    public let exportService: ReceiptExportService

    private let stateMachine: ChatPanelStateMachine
    private var refreshTask: Task<Void, Never>?

    // MARK: - Init

    public init(
        documentID: UUID,
        documentTitle: String,
        documentStore: DocumentStore,
        dataLayer: TesseraDataLayer,
        userID: UserID = UUID(),
        signer: ReceiptSigner = ReceiptSigner()
    ) {
        self.documentStore = documentStore
        self.dataLayer = dataLayer
        self.userID = userID
        self.activeDocumentID = documentID
        let coordinator = ReceiptsCoordinator()
        self.coordinator = coordinator
        self.coordinatorBridge = ReceiptsCoordinatorBridge(coordinator: coordinator)
        self.crossDocRegistry = CrossDocumentChatRegistry()
        self.stateMachine = ChatPanelStateMachine(
            documentID: documentID,
            documentStore: documentStore
        )
        // The state machine uses the DocumentStore-backed
        // wrapper (DocumentStoreChatQueueStore) for its
        // chat-queue load/save; the export service uses
        // the full DocumentStore directly.
        self.chatPanelViewModel = ChatPanelViewModel(
            documentID: documentID,
            documentTitle: documentTitle,
            stateMachine: stateMachine,
            crossDocRegistry: crossDocRegistry,
            coordinator: coordinator
        )
        self.exportService = ReceiptExportService(
            documentStore: documentStore,
            dataLayer: dataLayer,
            signer: signer
        )
    }

    // MARK: - Lifecycle

    public func start() async {
        await chatPanelViewModel.start()
        await crossDocRegistry.register(
            stateMachine,
            for: chatPanelViewModel.documentID,
            title: chatPanelViewModel.documentTitle
        )
        await crossDocRegistry.setCurrent(documentID: chatPanelViewModel.documentID)
        // Wire the chat item lookup so the coordinator can
        // resolve a receipt id to the corresponding chat item.
        // The chat panel is `@MainActor`-isolated; the
        // closure runs on the coordinator's actor. We hop
        // to the main actor for the read.
        let viewModel = chatPanelViewModel
        await coordinator.setChatItemLookup { receiptID in
            return await MainActor.run {
                viewModel.items.first(where: { $0.item.producedReceiptID == receiptID })?.item.id
            }
        }
        // Start the coordinator bridge polling.
        await coordinatorBridge.start()
        // Begin observing the bridge for scroll targets.
        beginObserving()
        // Initial receipt count: read the chain.
        do {
            let history = try await documentStore.history(of: chatPanelViewModel.documentID, limit: 1)
            await chatPanelViewModel.setReceiptCount(history.count)
        } catch {
            // Ignore — empty chain is fine.
        }
        refreshTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 200_000_000)
                await self?.chatPanelViewModel.refresh()
            }
        }
    }

    public func stop() {
        refreshTask?.cancel()
        refreshTask = nil
        Task { [crossDocRegistry, coordinatorBridge, chatPanelViewModel, documentID = chatPanelViewModel.documentID] in
            await crossDocRegistry.unregister(documentID: documentID)
            await coordinatorBridge.stop()
            await chatPanelViewModel.stop()
        }
    }

    // MARK: - User actions

    public func undo() {
        // Wired to the document store's ReceiptUndoManager
        // in Phase 5. For v1, this is a no-op (the chat
        // panel header shows undo/redo state but the
        // actual undo is handled by the editor's menu).
    }

    public func redo() {
        // Same as `undo` — wired in Phase 5.
    }

    public func switchToDocument(_ id: UUID) async {
        activeDocumentID = id
        await crossDocRegistry.setCurrent(documentID: id)
    }

    public func openReceiptInDrawer(_ id: UUID) async {
        await coordinator.openReceiptInDrawer(id, fromChatItem: nil)
        withAnimation { showReceiptsDrawer = true }
    }

    // MARK: - Internals

    private func beginObserving() {
        Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 100_000_000)
                await self?.observeCoordinator()
            }
        }
    }

    private func observeCoordinator() async {
        let target = await coordinator.currentScrollTarget
        if target != scrollTarget {
            scrollTarget = target
        }
        let visible = await coordinator.isDrawerVisible
        if visible != showReceiptsDrawer {
            showReceiptsDrawer = visible
        }
    }
}
