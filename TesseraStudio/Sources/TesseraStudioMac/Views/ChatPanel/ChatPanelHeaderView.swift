import SwiftUI
import TesseraCore

// MARK: - ChatPanelHeaderView

/// The header region of the chat panel (per spec §6.1).
/// Shows the document title, the "Working in background"
/// chip (if any), undo/redo buttons, and the receipt
/// count. The header is a thin strip at the top of the
/// panel; the queue list is the main body, the input
/// field is the footer.
public struct ChatPanelHeaderView: View {

    public let title: String
    public let receiptCount: Int
    public let holdMode: HoldMode
    public let canUndo: Bool
    public let canRedo: Bool
    public let backgroundDocuments: [ActiveDocumentInfo]
    public let onUndo: (() -> Void)?
    public let onRedo: (() -> Void)?
    public let onSwitchToDocument: ((UUID) -> Void)?
    public let onPauseAll: (() -> Void)?

    public init(
        title: String,
        receiptCount: Int,
        holdMode: HoldMode,
        canUndo: Bool = false,
        canRedo: Bool = false,
        backgroundDocuments: [ActiveDocumentInfo] = [],
        onUndo: (() -> Void)? = nil,
        onRedo: (() -> Void)? = nil,
        onSwitchToDocument: ((UUID) -> Void)? = nil,
        onPauseAll: (() -> Void)? = nil
    ) {
        self.title = title
        self.receiptCount = receiptCount
        self.holdMode = holdMode
        self.canUndo = canUndo
        self.canRedo = canRedo
        self.backgroundDocuments = backgroundDocuments
        self.onUndo = onUndo
        self.onRedo = onRedo
        self.onSwitchToDocument = onSwitchToDocument
        self.onPauseAll = onPauseAll
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text(title)
                    .font(.system(size: 13, weight: .semibold))
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer(minLength: 4)
                undoRedoButtons
                receiptCountBadge
            }
            if !backgroundDocuments.isEmpty {
                backgroundChip
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(headerBackground)
    }

    // MARK: - Pieces

    private var undoRedoButtons: some View {
        HStack(spacing: 2) {
            Button(action: { onUndo?() }) {
                Image(systemName: "arrow.uturn.backward")
                    .font(.system(size: 11))
            }
            .buttonStyle(.borderless)
            .disabled(!canUndo)
            .help("Undo last receipt")
            Button(action: { onRedo?() }) {
                Image(systemName: "arrow.uturn.forward")
                    .font(.system(size: 11))
            }
            .buttonStyle(.borderless)
            .disabled(!canRedo)
            .help("Redo")
        }
    }

    private var receiptCountBadge: some View {
        HStack(spacing: 3) {
            Image(systemName: "doc.text")
                .font(.system(size: 9))
            Text("\(receiptCount)")
                .font(.system(size: 11, weight: .medium))
                .monospacedDigit()
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(
            Capsule().fill(Color.secondary.opacity(0.12))
        )
        .help("\(receiptCount) receipt\(receiptCount == 1 ? "" : "s") in the chain")
    }

    private var backgroundChip: some View {
        HStack(spacing: 6) {
            Image(systemName: "person.2.fill")
                .font(.system(size: 10))
                .foregroundStyle(.orange)
            Text(backgroundText)
                .font(.system(size: 11))
                .lineLimit(1)
            Spacer(minLength: 4)
            if let first = backgroundDocuments.first {
                Button("Switch") {
                    onSwitchToDocument?(first.documentID)
                }
                .buttonStyle(.borderless)
                .font(.system(size: 10, weight: .medium))
            }
            Button("Pause all") {
                onPauseAll?()
            }
            .buttonStyle(.borderless)
            .font(.system(size: 10, weight: .medium))
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(Color.orange.opacity(0.12))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(Color.orange.opacity(0.3), lineWidth: 1)
        )
    }

    private var backgroundText: String {
        if backgroundDocuments.count == 1, let first = backgroundDocuments.first {
            return "Agent is editing '\(first.title)'"
        }
        let names = backgroundDocuments.prefix(2).map { "'\($0.title)'" }.joined(separator: ", ")
        let suffix = backgroundDocuments.count > 2 ? " +\(backgroundDocuments.count - 2) more" : ""
        return "Agent is editing \(names)\(suffix)"
    }

    @ViewBuilder
    private var headerBackground: some View {
        if holdMode.isPaused {
            // The paused indicator stripe (orange, 4pt) at the
            // top of the header. Animated in with the
            // agent-paused-banner primitive when the pause
            // begins.
            VStack(spacing: 0) {
                Rectangle()
                    .fill(Color.orange)
                    .frame(height: 3)
                Color.clear
            }
        } else {
            Color.clear
        }
    }
}
