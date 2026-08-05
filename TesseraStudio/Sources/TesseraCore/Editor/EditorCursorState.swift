import Foundation

// MARK: - EditorCursorState

/// The two-cursor model (per spec §6.5). The user and the
/// agent have separate cursors in the same document; both can
/// be active at the same time and they don't conflict with
/// each other. The user can move their cursor freely without
/// affecting the agent's cursor (and vice versa).
///
/// The state is held in the platform view layer; the data
/// model is small and value-typed so it can be observed and
/// rebound by SwiftUI without an extra layer of state.
///
/// The data model carries the `blockID` + `offset` pair (the
/// `TextCursor` from Phase 1) plus a coarse `selectionRange`
/// for when the user (or agent) has a non-empty selection.
/// The two fields are independent: a cursor without a
/// selection has `selectionRange == nil`.
///
/// Phase 1 already provides `TextCursor` and `CursorPair`; this
/// is the editor-side view that extends them with the
/// selection range and the per-cursor data the platform
/// text view needs (e.g., the agent cursor's "active" flag
/// controls the blink animation).
public struct EditorCursorState: Codable, Sendable, Hashable {
    public var userCursor: TextCursor?
    public var userSelection: CursorSelection?
    public var agentCursor: TextCursor?
    public var agentSelection: CursorSelection?
    /// When the agent cursor is "active" (the agent is currently
    /// editing in the document), the cursor blinks at the
    /// standard 530ms rate. When inactive, the cursor is static
    /// (Reduce Motion fallback). The platform view layer reads
    /// this flag to drive the animation.
    public var agentCursorActive: Bool

    public init(
        userCursor: TextCursor? = nil,
        userSelection: CursorSelection? = nil,
        agentCursor: TextCursor? = nil,
        agentSelection: CursorSelection? = nil,
        agentCursorActive: Bool = false
    ) {
        self.userCursor = userCursor
        self.userSelection = userSelection
        self.agentCursor = agentCursor
        self.agentSelection = agentSelection
        self.agentCursorActive = agentCursorActive
    }

    public static let empty = EditorCursorState()

    /// True iff the state has no active cursors.
    public var isEmpty: Bool {
        userCursor == nil && agentCursor == nil
    }

    /// True iff the user cursor is currently active (i.e. the
    /// user has focus in the editor).
    public var hasUserFocus: Bool {
        userCursor != nil
    }

    /// True iff the agent cursor is currently active AND
    /// blinking. Inactive when the agent is idle.
    public var hasAgentActive: Bool {
        agentCursor != nil && agentCursorActive
    }

    /// The `CursorPair` data model from Phase 1, for storage /
    /// receipt attachments.
    public var asCursorPair: CursorPair {
        CursorPair(user: userCursor, agent: agentCursor)
    }
}

// MARK: - CursorSelection

/// A selection in a single block. Selections that cross
/// block boundaries are represented as two `CursorSelection`s
/// (one per block) with the same `anchor`/`head` on each side
/// of the boundary; the platform text view handles the
/// multi-block case internally and emits a per-block
/// selection for the editor to consume.
public struct CursorSelection: Codable, Sendable, Hashable {
    public var blockID: UUID
    public var anchorOffset: Int
    public var headOffset: Int

    public init(blockID: UUID, anchorOffset: Int, headOffset: Int) {
        self.blockID = blockID
        self.anchorOffset = anchorOffset
        self.headOffset = headOffset
    }

    public var isEmpty: Bool { anchorOffset == headOffset }
    public var length: Int { abs(headOffset - anchorOffset) }
    public var lowerOffset: Int { min(anchorOffset, headOffset) }
    public var upperOffset: Int { max(anchorOffset, headOffset) }
}

// MARK: - CursorInBlock

/// A position within a single block, addressing characters
/// in the block's flattened text (i.e., the concatenation of
/// all `InlineRun.text` values). This is the unit the
/// `TesseraTextContentManager` uses to translate between
/// platform `NSTextLocation` values and the AST's
/// `(blockID, runIndex, runOffset)` triple.
public struct CursorInBlock: Codable, Sendable, Hashable {
    public var blockID: UUID
    public var runIndex: Int
    public var runOffset: Int

    public init(blockID: UUID, runIndex: Int, runOffset: Int) {
        self.blockID = blockID
        self.runIndex = runIndex
        self.runOffset = runOffset
    }
}

// MARK: - TextCursorInBlock / CursorSelection extension

extension TextCursor {
    /// Convert a `TextCursor` (blockID + offset into the
    /// flattened text) to a `CursorInBlock` (blockID +
    /// runIndex + runOffset) by walking the block's inline
    /// runs. Returns nil if the cursor is in a block that
    /// doesn't have inline content (a divider, an image, etc.)
    /// or the offset is out of range.
    public func resolved(in block: Block) -> CursorInBlock? {
        guard block.type != .divider,
              block.type != .image,
              block.type != .equation else { return nil }
        var remaining = offset
        for (idx, run) in block.content.enumerated() {
            if remaining <= run.text.count {
                return CursorInBlock(blockID: blockID, runIndex: idx, runOffset: remaining)
            }
            remaining -= run.text.count
        }
        // The offset is at the end of the block's content.
        let lastIndex = max(0, block.content.count - 1)
        let lastCount = block.content.last?.text.count ?? 0
        return CursorInBlock(blockID: blockID, runIndex: lastIndex, runOffset: lastCount)
    }

    /// Construct a `TextCursor` from a `CursorInBlock`,
    /// translating the run-local offset back to the
    /// block's flattened offset.
    public init(_ cursor: CursorInBlock, in block: Block) {
        let flatOffset = block.content.prefix(cursor.runIndex).reduce(0) { $0 + $1.text.count }
            + cursor.runOffset
        self.init(
            blockID: cursor.blockID,
            offset: flatOffset,
            affinity: .downstream
        )
    }
}
