import SwiftUI
import TesseraCore

// MARK: - ChatQueueRowView

/// One row in the chat panel's queue list. The row is
/// pure presentational: it reads the `ChatQueueItemDisplay`
/// and renders the per-state visual treatment (per spec
/// §6.3). The row's interactivity is provided by the
/// parent list (drag-to-reorder, tap to act, etc.).
///
/// **State treatments:**
/// - `pending`     — italic, 60% opacity, clock icon.
/// - `inProgress`  — normal, highlight background, pulse
///                   on the icon, "Hold your horses" button
///                   visible (rendered by the parent).
/// - `applied`     — normal, checkmark, receipt chip.
/// - `failed`      — red flash, error icon, retry button.
/// - `superseded`  — 50% opacity, "replaces #N" badge.
///
/// The row respects the system Reduce Motion setting via
/// Phase 2's `thinkingPulse(isActive:)` modifier.
public struct ChatQueueRowView: View {

    public let display: ChatQueueItemDisplay
    public let onTap: (() -> Void)?
    public let onReceiptChipTap: (() -> Void)?

    public init(
        display: ChatQueueItemDisplay,
        onTap: (() -> Void)? = nil,
        onReceiptChipTap: (() -> Void)? = nil
    ) {
        self.display = display
        self.onTap = onTap
        self.onReceiptChipTap = onReceiptChipTap
    }

    public var body: some View {
        HStack(alignment: .top, spacing: 8) {
            iconView
                .frame(width: 18, height: 18)
            VStack(alignment: .leading, spacing: 4) {
                messageText
                if let chip = receiptChipText {
                    receiptChipView(chip)
                }
                if let badge = display.style.replaceBadge {
                    replacesBadge(position: badge)
                }
                if display.style.showsProgress, let progress = display.meta.failureNote {
                    // The state machine stores agent progress
                    // (e.g., "Streaming block 2 of 4") in the
                    // failure-note side channel for in-progress
                    // items. Render it as the progress caption.
                    progressCaption(progress)
                }
                if let note = display.meta.failureNote, display.style.state == .failed {
                    failureNote(note)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 4)
        .padding(.horizontal, 8)
        .background(backgroundForStyle)
        .opacity(display.style.opacity)
        .contentShape(Rectangle())
        .onTapGesture {
            onTap?()
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(accessibilityLabel)
    }

    // MARK: - Pieces

    private var iconView: some View {
        Image(systemName: display.style.iconSystemName)
            .font(.system(size: 12, weight: .semibold))
            .foregroundStyle(display.style.iconTint)
            .thinkingPulse(isActive: display.style.pulseAnimation)
    }

    private var messageText: some View {
        Text(display.message)
            .font(.system(size: 13))
            .italic(display.style.isItalic)
            .lineLimit(3)
            .multilineTextAlignment(.leading)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func receiptChipView(_ chip: String) -> some View {
        Button {
            onReceiptChipTap?()
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "doc.text")
                    .font(.system(size: 10))
                Text(chip)
                    .font(.system(size: 11))
                    .lineLimit(1)
            }
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.green.opacity(0.15))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 4)
                    .stroke(Color.green.opacity(0.4), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
        .help("View receipt in drawer")
    }

    private func replacesBadge(position: Int) -> some View {
        HStack(spacing: 4) {
            Image(systemName: "arrow.uturn.backward")
                .font(.system(size: 9))
            Text("replaces #\(position)")
                .font(.system(size: 10, weight: .medium))
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 1)
        .foregroundStyle(.secondary)
        .background(
            Capsule().fill(Color.secondary.opacity(0.12))
        )
    }

    private func progressCaption(_ text: String) -> some View {
        HStack(spacing: 4) {
            ProgressView()
                .controlSize(.small)
                .scaleEffect(0.6)
            Text(text)
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
        }
    }

    private func failureNote(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 11))
            .foregroundStyle(.red)
            .lineLimit(2)
    }

    @ViewBuilder
    private var backgroundForStyle: some View {
        switch display.style.backgroundStyle {
        case .clear:
            Color.clear
        case .subtleHighlight:
            RoundedRectangle(cornerRadius: 6)
                .fill(Color.yellow.opacity(0.08))
        case .redFlash:
            RoundedRectangle(cornerRadius: 6)
                .fill(Color.red.opacity(0.10))
        }
    }

    /// The text used in the receipt chip. For `applied`
    /// rows, the state machine's `markApplied` records the
    /// receipt id; the chat panel looks up the receipt's
    /// summary from the chain. The row falls back to a
    /// short generic text when the summary is not in the
    /// display's meta.
    private var receiptChipText: String? {
        guard display.style.state == .applied else { return nil }
        return display.style.receiptChip ?? "Receipt logged"
    }

    // MARK: - Accessibility

    private var accessibilityLabel: String {
        var parts: [String] = []
        parts.append("Position \(display.position)")
        switch display.style.state {
        case .pending: parts.append("pending")
        case .inProgress: parts.append("in progress")
        case .applied: parts.append("applied")
        case .failed: parts.append("failed")
        case .superseded: parts.append("superseded")
        }
        parts.append(display.message)
        if let badge = display.style.replaceBadge {
            parts.append("replaces #\(badge)")
        }
        return parts.joined(separator: ", ")
    }
}
