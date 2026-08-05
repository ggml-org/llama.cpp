#if os(iOS)
import SwiftUI
import UIKit
import TesseraCore

// MARK: - ChatPanelView_iOS

/// The iOS chat panel. Per spec §6.1 the iOS chat panel
/// is a bottom tab; the editor is the other tab. The
/// tab bar is persistent (the user can switch between
/// editor and chat without losing state).
///
/// **Layout.** The iOS view is functionally identical
/// to the macOS view (header, queue list, input). The
/// differences are the touch-optimized drag gesture
/// (long-press to lift) and the modal sheet for the
/// "Hold your horses" dialog (rather than a popover).
///
/// **Drag-to-reorder.** The iOS view uses SwiftUI's
/// `.onDrag` + `.dropDestination` with a long-press
/// gesture to start the drag. VoiceOver rotor is wired
/// via the `.accessibilityRotor` modifier.
public struct ChatPanelView_iOS: View {

    @ObservedObject public var viewModel: ChatPanelViewModel
    public let onOpenReceipt: ((UUID) -> Void)?

    @State private var holdResponse: String = ""
    @State private var showHoldSheet: Bool = false
    @State private var showReceiptsSheet: Bool = false
    @State private var backgroundDocuments: [ActiveDocumentInfo] = []

    public init(
        viewModel: ChatPanelViewModel,
        onOpenReceipt: ((UUID) -> Void)? = nil
    ) {
        self.viewModel = viewModel
        self.onOpenReceipt = onOpenReceipt
    }

    public var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ChatPanelHeaderView(
                    title: viewModel.documentTitle,
                    receiptCount: viewModel.receiptCount,
                    holdMode: viewModel.holdMode,
                    backgroundDocuments: backgroundDocuments
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
            .navigationTitle("Chat")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        showReceiptsSheet = true
                    } label: {
                        Image(systemName: "doc.text.magnifyingglass")
                    }
                }
            }
            .sheet(isPresented: $showHoldSheet) {
                if let dialog = viewModel.holdDialog {
                    HoldYourHorsesDialog_iOS(
                        response: $holdResponse,
                        state: dialog,
                        onSubmit: {
                            holdResponse = ""
                        },
                        onResume: {
                            Task { await viewModel.resume() }
                            showHoldSheet = false
                        },
                        onCancel: {
                            Task { await viewModel.resume() }
                            showHoldSheet = false
                        }
                    )
                }
            }
            .sheet(isPresented: $showReceiptsSheet) {
                ReceiptsDrawerSheet_iOS(
                    documentID: viewModel.documentID,
                    documentTitle: viewModel.documentTitle
                )
            }
        }
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
            showHoldSheet = (mode == .hold || mode == .holdRequested) && viewModel.holdDialog != nil
        }
    }

    // MARK: - Queue list

    private var queueList: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 6) {
                if viewModel.items.isEmpty {
                    emptyState
                } else {
                    ForEach(viewModel.items) { display in
                        ChatQueueRowView_iOS(
                            display: display,
                            onTap: { handleTap(display) },
                            onReceiptChipTap: {
                                if let receiptID = display.item.producedReceiptID {
                                    onOpenReceipt?(receiptID)
                                }
                            }
                        )
                    }
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
        }
        .accessibilityRotor("Pending items") {
            ForEach(viewModel.items.filter { $0.item.state == .pending }) { item in
                AccessibilityRotorEntry(item.id) {
                    Text(item.message)
                }
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "bubble.left")
                .font(.system(size: 32))
                .foregroundStyle(.tertiary)
            Text("No commands yet")
                .font(.headline)
            Text("Type a command for the agent below.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 60)
    }

    // MARK: - Tap handling

    private func handleTap(_ display: ChatQueueItemDisplay) {
        switch display.item.state {
        case .pending:
            viewModel.inputText = display.message
            Task { await viewModel.delete(itemID: display.item.id) }
        case .applied:
            if let receiptID = display.item.producedReceiptID {
                onOpenReceipt?(receiptID)
            }
        case .failed:
            viewModel.inputText = display.message
        case .inProgress:
            break
        }
        if display.item.isSuperseded {
            Task { await viewModel.unsupersede(itemID: display.item.id) }
        }
    }
}

// MARK: - iOS row variant

/// The iOS row variant of the chat queue row. Slightly
/// larger touch targets, slightly larger fonts. Uses the
/// shared `ChatQueueItemDisplay` so the per-state
/// treatment is the same as the macOS row.
public struct ChatQueueRowView_iOS: View {

    public let display: ChatQueueItemDisplay
    public let onTap: () -> Void
    public let onReceiptChipTap: () -> Void

    public init(
        display: ChatQueueItemDisplay,
        onTap: @escaping () -> Void,
        onReceiptChipTap: @escaping () -> Void
    ) {
        self.display = display
        self.onTap = onTap
        self.onReceiptChipTap = onReceiptChipTap
    }

    public var body: some View {
        Button(action: onTap) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: display.style.iconSystemName)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(display.style.iconTint)
                    .frame(width: 22, height: 22)
                    .thinkingPulse(isActive: display.style.pulseAnimation)
                VStack(alignment: .leading, spacing: 4) {
                    Text(display.message)
                        .font(.system(size: 14))
                        .italic(display.style.isItalic)
                        .lineLimit(3)
                        .multilineTextAlignment(.leading)
                    if display.style.state == .applied {
                        Button(action: onReceiptChipTap) {
                            Label("Receipt", systemImage: "doc.text")
                                .font(.system(size: 12))
                                .foregroundStyle(.green)
                        }
                        .buttonStyle(.plain)
                    }
                    if let badge = display.style.replaceBadge {
                        Text("replaces #\(badge)")
                            .font(.system(size: 10, weight: .medium))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Capsule().fill(Color.secondary.opacity(0.12)))
                            .foregroundStyle(.secondary)
                    }
                }
                Spacer(minLength: 0)
            }
            .padding(.vertical, 8)
            .padding(.horizontal, 12)
            .background(rowBackground)
            .opacity(display.style.opacity)
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private var rowBackground: some View {
        switch display.style.backgroundStyle {
        case .clear:
            Color.clear
        case .subtleHighlight:
            RoundedRectangle(cornerRadius: 8).fill(Color.yellow.opacity(0.10))
        case .redFlash:
            RoundedRectangle(cornerRadius: 8).fill(Color.red.opacity(0.10))
        }
    }
}

// MARK: - iOS hold dialog

struct HoldYourHorsesDialog_iOS: View {
    @Binding var response: String
    let state: ChatPanelViewModel.HoldDialogState
    let onSubmit: () -> Void
    let onResume: () -> Void
    let onCancel: () -> Void

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 16) {
                Image(systemName: "pause.circle.fill")
                    .font(.system(size: 36))
                    .foregroundStyle(.orange)
                Text(state.title)
                    .font(.title2.bold())
                Text(state.message)
                    .font(.body)
                    .foregroundStyle(.secondary)
                VStack(alignment: .leading, spacing: 6) {
                    Text("What's working? What's not?")
                        .font(.subheadline.weight(.medium))
                    TextEditor(text: $response)
                        .frame(minHeight: 100, maxHeight: 200)
                        .padding(8)
                        .background(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.secondary.opacity(0.3), lineWidth: 1)
                        )
                }
                Spacer()
                HStack {
                    Button("Cancel", action: onCancel)
                        .buttonStyle(.bordered)
                    Spacer()
                    Button("Resume", action: onResume)
                        .buttonStyle(.borderedProminent)
                }
            }
            .padding(20)
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

#endif
