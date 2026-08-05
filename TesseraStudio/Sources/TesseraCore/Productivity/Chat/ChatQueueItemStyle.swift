import Foundation
import SwiftUI

// MARK: - ChatQueueItemStyle

/// The visual treatment of a `ChatQueueItem` row in the
/// chat panel (per spec §6.3). The treatment is encoded
/// as a value type so it can be unit-tested without a
/// SwiftUI view tree; the SwiftUI view consumes the style
/// via a single `func style(for:)` lookup.
///
/// **Per-state treatment:**
/// - `pending`     — italic, 60% opacity, clock icon.
/// - `inProgress`  — regular, subtle highlight, pulse
///                   animation on the status dot, "Hold
///                   your horses" button visible.
/// - `applied`     — regular, checkmark, inline receipt
///                   chip.
/// - `failed`      — regular, red flash, error message,
///                   retry button.
/// - `superseded`  — regular, 50% opacity, "replaces #N"
///                   badge.
///
/// The `replaces` field on the `applied` and `superseded`
/// cases is the display position of the original item
/// (e.g., "replaces #3" for the third item in the queue).
/// `nil` when the original position is unknown.
public struct ChatQueueItemStyle: Sendable, Hashable {
    public enum State: String, Sendable, Hashable, CaseIterable {
        case pending
        case inProgress
        case applied
        case failed
        case superseded
    }

    public enum Icon: String, Sendable, Hashable {
        case clock
        case progress
        case checkmark
        case warning
        case superseded
    }

    public let state: State
    public let icon: Icon
    public let isItalic: Bool
    public let opacity: Double
    public let backgroundStyle: BackgroundStyle
    public let replaceBadge: Int?
    public let receiptChip: String?
    public let showsProgress: Bool
    public let showsRetry: Bool
    public let pulseAnimation: Bool

    public enum BackgroundStyle: Sendable, Hashable {
        case clear
        case subtleHighlight
        case redFlash
    }

    public init(
        state: State,
        icon: Icon,
        isItalic: Bool,
        opacity: Double,
        backgroundStyle: BackgroundStyle,
        replaceBadge: Int? = nil,
        receiptChip: String? = nil,
        showsProgress: Bool = false,
        showsRetry: Bool = false,
        pulseAnimation: Bool = false
    ) {
        self.state = state
        self.icon = icon
        self.isItalic = isItalic
        self.opacity = opacity
        self.backgroundStyle = backgroundStyle
        self.replaceBadge = replaceBadge
        self.receiptChip = receiptChip
        self.showsProgress = showsProgress
        self.showsRetry = showsRetry
        self.pulseAnimation = pulseAnimation
    }

    /// The style for a given `ChatQueueItem`. The `meta`
    /// argument is the state machine's per-item metadata
    /// (failure notes, supersede reasoning); `receiptCount`
    /// is the document's total receipt count.
    public static func style(
        for item: ChatQueueItem,
        in allItems: [ChatQueueItem],
        meta: ChatQueueItemMeta = .empty
    ) -> ChatQueueItemStyle {
        // Superseded items are styled as superseded regardless
        // of their underlying state (a superseded applied item
        // is still superseded — the user wants to see the
        // history of intent).
        if item.isSuperseded {
            let originalPos = item.displayPosition(among: allItems)
            return ChatQueueItemStyle(
                state: .superseded,
                icon: .superseded,
                isItalic: false,
                opacity: 0.5,
                backgroundStyle: .clear,
                replaceBadge: originalPos
            )
        }
        switch item.state {
        case .pending:
            return ChatQueueItemStyle(
                state: .pending,
                icon: .clock,
                isItalic: true,
                opacity: 0.6,
                backgroundStyle: .clear
            )
        case .inProgress:
            return ChatQueueItemStyle(
                state: .inProgress,
                icon: .progress,
                isItalic: false,
                opacity: 1.0,
                backgroundStyle: .subtleHighlight,
                showsProgress: true,
                pulseAnimation: true
            )
        case .applied:
            // The receipt chip is the receipt's summary
            // (e.g., "3 paragraphs updated, 1 list added").
            // The state machine's `markApplied` records the
            // receipt id; the chat panel looks up the
            // receipt's summary from the chain.
            return ChatQueueItemStyle(
                state: .applied,
                icon: .checkmark,
                isItalic: false,
                opacity: 1.0,
                backgroundStyle: .clear,
                receiptChip: meta.failureNote  // re-used as a side channel; chat panel overrides
            )
        case .failed:
            return ChatQueueItemStyle(
                state: .failed,
                icon: .warning,
                isItalic: false,
                opacity: 1.0,
                backgroundStyle: .redFlash,
                showsRetry: true
            )
        }
    }

    /// SF Symbol name for the row's leading icon.
    public var iconSystemName: String {
        switch icon {
        case .clock: return "clock"
        case .progress: return "circle.dotted"
        case .checkmark: return "checkmark.circle.fill"
        case .warning: return "exclamationmark.triangle.fill"
        case .superseded: return "arrow.uturn.backward"
        }
    }

    /// The row's foreground tint (where the system color is
    /// used; macOS and iOS both support this).
    public var iconTint: Color {
        switch state {
        case .pending: return .secondary
        case .inProgress: return .accentColor
        case .applied: return .green
        case .failed: return .red
        case .superseded: return .secondary
        }
    }
}

// MARK: - ChatQueueItemDisplay

/// The full row data the chat panel renders. This is the
/// output of `ChatQueueItemStyle.style(for:in:meta:)` plus
/// the message text and a `position` field (the 1-based
/// position in the queue, for the "replaces #N" badge and
/// for the row's accessibility label).
public struct ChatQueueItemDisplay: Sendable, Hashable, Identifiable {
    public let item: ChatQueueItem
    public let style: ChatQueueItemStyle
    public let meta: ChatQueueItemMeta
    public let message: String
    public let position: Int

    public var id: UUID { item.id }

    public init(
        item: ChatQueueItem,
        style: ChatQueueItemStyle,
        meta: ChatQueueItemMeta,
        message: String,
        position: Int
    ) {
        self.item = item
        self.style = style
        self.meta = meta
        self.message = message
        self.position = position
    }

    /// Build a display row from a queue + an item.
    public static func display(
        for item: ChatQueueItem,
        in queue: [ChatQueueItem],
        meta: ChatQueueItemMeta
    ) -> ChatQueueItemDisplay {
        let style = ChatQueueItemStyle.style(for: item, in: queue, meta: meta)
        let position = item.displayPosition(among: queue) ?? 0
        return ChatQueueItemDisplay(
            item: item,
            style: style,
            meta: meta,
            message: item.message,
            position: position
        )
    }
}
