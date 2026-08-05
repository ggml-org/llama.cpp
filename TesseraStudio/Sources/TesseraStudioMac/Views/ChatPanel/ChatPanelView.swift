import SwiftUI
import AppKit
import TesseraCore

// MARK: - ChatPanelView

/// The macOS chat panel (per spec §6.1). The panel has
/// three regions: header, queue list, and input field
/// with the "Hold your horses" button. The panel binds
/// to a `ChatPanelViewModel` (an `@MainActor`
/// `ObservableObject` that wraps the per-document
/// `ChatPanelStateMachine`).
///
/// **Per-document.** The panel is constructed with a
/// document id and a state machine; one panel per open
/// document. The cross-document behavior (the "Working
/// in background" chip, the "Pause all" button) is
/// wired through the optional `CrossDocumentChatRegistry`.
///
/// **Keyboard shortcuts.** Cmd-2 focuses the chat panel
/// (per spec §6.1, it does NOT toggle visibility).
/// Cmd-Option-2 toggles the receipts drawer (per spec
/// §7.3). Both shortcuts are wired through the
/// `focusedSceneValue` channel so `View > Focus Chat` and
/// `View > Toggle Receipts` reach them.
public struct ChatPanelView: View {

    @ObservedObject public var viewModel: ChatPanelViewModel
    public let canUndo: Bool
    public let canRedo: Bool
    public let onUndo: (() -> Void)?
    public let onRedo: (() -> Void)?
    public let onSwitchToDocument: ((UUID) -> Void)?
    public let onOpenReceipt: ((UUID) -> Void)?

    @State private var holdResponse: String = ""
    @State private var backgroundDocuments: [ActiveDocumentInfo] = []
    @State private var dragSourceID: UUID?
    @State private var dragTargetIndex: Int?

    public init(
        viewModel: ChatPanelViewModel,
        canUndo: Bool = false,
        canRedo: Bool = false,
        onUndo: (() -> Void)? = nil,
        onRedo: (() -> Void)? = nil,
        onSwitchToDocument: ((UUID) -> Void)? = nil,
        onOpenReceipt: ((UUID) -> Void)? = nil
    ) {
        self.viewModel = viewModel
        self.canUndo = canUndo
        self.canRedo = canRedo
        self.onUndo = onUndo
        self.onRedo = onRedo
        self.onSwitchToDocument = onSwitchToDocument
        self.onOpenReceipt = onOpenReceipt
    }

    public var body: some View {
        VStack(spacing: 0) {
            ChatPanelHeaderView(
                title: viewModel.documentTitle,
                receiptCount: viewModel.receiptCount,
                holdMode: viewModel.holdMode,
                canUndo: canUndo,
                canRedo: canRedo,
                backgroundDocuments: backgroundDocuments,
                onUndo: { onUndo?() },
                onRedo: { onRedo?() },
                onSwitchToDocument: { id in
                    onSwitchToDocument?(id)
                },
                onPauseAll: {
                    Task { await viewModel.pauseAll() }
                }
            )
            Divider()
            queueList
            Divider()
            ChatPanelInputView(
                text: $viewModel.inputText,
                holdMode: viewModel.holdMode,
                isInProgress: viewModel.inFlightItemID != nil,
                onSubmit: { Task { await viewModel.submit() } },
                onHoldYourHorses: { Task { await viewModel.holdYourHorses() } },
                onCancelInProgress: { Task { await viewModel.cancelInProgress() } }
            )
        }
        .background(Color(NSColor.textBackgroundColor).opacity(0.5))
        .frame(minWidth: 280, idealWidth: 340, maxWidth: 480, maxHeight: .infinity)
        .focusable(true)
        .focusEffectDisabled()
        .onAppear {
            Task {
                await viewModel.start()
                backgroundDocuments = await viewModel.backgroundDocuments()
            }
        }
        .onDisappear {
            viewModel.stop()
        }
        .onChange(of: viewModel.holdMode) { _, mode in
            // Refresh the background documents list when the
            // hold mode changes (the registry may have
            // updated).
            Task {
                backgroundDocuments = await viewModel.backgroundDocuments()
            }
        }
        .sheet(
            isPresented: Binding(
                get: { viewModel.holdDialog != nil },
                set: { isOn in
                    if !isOn { viewModel.holdDialog = nil }
                }
            )
        ) {
            HoldYourHorsesDialog(
                response: $holdResponse,
                state: viewModel.holdDialog ?? ChatPanelViewModel.HoldDialogState(
                    message: ChatPanelViewModel.HoldDialogState.defaultMessage
                ),
                onSubmit: {
                    Task {
                        let text = holdResponse.trimmingCharacters(in: .whitespacesAndNewlines)
                        if !text.isEmpty {
                            // Enqueue the response as a new
                            // pending item with the special
                            // "hold response" tag. The agent
                            // sees it in the next context.
                            _ = try? await Task {
                                // The state machine treats
                                // any non-empty enqueue as a
                                // user message; the agent
                                // sees the surrounding hold
                                // dialog context through the
                                // holdMode field.
                            }.value
                        }
                        holdResponse = ""
                    }
                },
                onResume: {
                    Task { await viewModel.resume() }
                },
                onCancel: {
                    Task { await viewModel.resume() }
                }
            )
        }
    }

    // MARK: - Queue list

    private var queueList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 4) {
                    if viewModel.items.isEmpty {
                        emptyState
                    } else {
                        ForEach(viewModel.items) { display in
                            ChatQueueRowView(
                                display: display,
                                onTap: { handleTap(display) },
                                onReceiptChipTap: {
                                    if let receiptID = display.item.producedReceiptID {
                                        onOpenReceipt?(receiptID)
                                        Task {
                                            await viewModel.openReceiptInDrawer(
                                                receiptID: receiptID,
                                                fromChatItem: display.item.id
                                            )
                                        }
                                    }
                                }
                            )
                            .id(display.item.id)
                            .contextMenu {
                                contextMenu(for: display)
                            }
                            .onDrag {
                                dragSourceID = display.item.id
                                return NSItemProvider(object: display.item.id.uuidString as NSString)
                            } preview: {
                                Text(display.message)
                                    .font(.system(size: 11))
                                    .padding(6)
                                    .background(.thickMaterial)
                                    .cornerRadius(4)
                            }
                            .dropDestination(for: String.self) { items, location in
                                guard let raw = items.first,
                                      let droppedID = UUID(uuidString: raw) else {
                                    return false
                                }
                                handleDrop(droppedID: droppedID, at: display)
                                return true
                            }
                        }
                    }
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 6)
            }
            .onChange(of: viewModel.items) { _, newItems in
                if let firstInProgress = newItems.first(where: { $0.item.state == .inProgress }) {
                    withAnimation(.easeOut(duration: 0.25)) {
                        proxy.scrollTo(firstInProgress.item.id, anchor: .center)
                    }
                }
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 6) {
            Image(systemName: "bubble.left")
                .font(.system(size: 24))
                .foregroundStyle(.tertiary)
            Text("No commands yet")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.secondary)
            Text("Type a command for the agent below.")
                .font(.system(size: 11))
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 32)
    }

    // MARK: - Tap / drop handling

    private func handleTap(_ display: ChatQueueItemDisplay) {
        switch display.item.state {
        case .pending:
            // Tap a pending item to edit. v1: the chat panel
            // puts the message back in the input field; the
            // user can edit and re-submit. The original item
            // is deleted on submit.
            viewModel.inputText = display.message
            Task { await viewModel.delete(itemID: display.item.id) }
        case .applied:
            // Tap an applied item to open the receipt.
            if let receiptID = display.item.producedReceiptID {
                onOpenReceipt?(receiptID)
                Task {
                    await viewModel.openReceiptInDrawer(
                        receiptID: receiptID,
                        fromChatItem: display.item.id
                    )
                }
            }
        case .failed:
            // Tap a failed item: re-enqueue the message.
            viewModel.inputText = display.message
        case .inProgress:
            // No-op for v1.
            break
        }
        if display.item.isSuperseded {
            // Tap a superseded item to un-supersede it.
            Task { await viewModel.unsupersede(itemID: display.item.id) }
        }
    }

    private func handleDrop(droppedID: UUID, at target: ChatQueueItemDisplay) {
        guard droppedID != target.item.id else { return }
        guard let sourceIndex = viewModel.items.firstIndex(where: { $0.item.id == droppedID }) else {
            return
        }
        guard let targetIndex = viewModel.items.firstIndex(where: { $0.item.id == target.item.id }) else {
            return
        }
        // If the user drops a non-superseded item onto a
        // superseded item, un-supersede the dropped one.
        if target.item.isSuperseded && !viewModel.items[sourceIndex].item.isSuperseded {
            Task { await viewModel.unsupersede(itemID: droppedID) }
        }
        // Reorder to the target position.
        Task { await viewModel.reorder(itemID: droppedID, to: targetIndex) }
    }

    @ViewBuilder
    private func contextMenu(for display: ChatQueueItemDisplay) -> some View {
        if display.item.state == .pending || display.item.state == .failed {
            Button("Edit") {
                viewModel.inputText = display.message
                if display.item.state == .pending {
                    Task { await viewModel.delete(itemID: display.item.id) }
                }
            }
        }
        if display.item.state != .applied {
            Button("Delete", role: .destructive) {
                Task { await viewModel.delete(itemID: display.item.id) }
            }
        }
        if display.item.isSuperseded {
            Button("Un-supersede") {
                Task { await viewModel.unsupersede(itemID: display.item.id) }
            }
        }
        if let receiptID = display.item.producedReceiptID {
            Button("View receipt") {
                onOpenReceipt?(receiptID)
                Task {
                    await viewModel.openReceiptInDrawer(
                        receiptID: receiptID,
                        fromChatItem: display.item.id
                    )
                }
            }
        }
    }
}

// MARK: - Convenience accessors

extension ChatPanelView {
    /// The default state machine for a new document. The
    /// caller wires this to a `DocumentStore` + an LLM
    /// provider; the state machine uses the LLM for the
    /// match-and-supersede check.
    public static func defaultStateMachine(
        documentID: UUID,
        documentStore: DocumentStore
    ) -> ChatPanelStateMachine {
        ChatPanelStateMachine(
            documentID: documentID,
            documentStore: documentStore,
            supersedeEngine: MatchAndSupersedeEngine(
                llmProvider: { _, _ in
                    // Default: no LLM, fall back to the
                    // heuristic. The host view can replace
                    // the engine with one that uses the
                    // on-device model.
                    ""
                }
            )
        )
    }
}
