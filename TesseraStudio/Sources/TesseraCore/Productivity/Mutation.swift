import Foundation

// MARK: - Mutation

/// A typed operation against a ``DocumentAST``. The single interface
/// for both the user (Phase 2 editor) and the agent (Phase 3 chat
/// panel). Every operation is undoable as a unit and produces one
/// ``Receipt``.
///
/// The variants fall into three groups:
///
/// - **Block-level** (`insertBlockAfter`, `deleteBlock`, ...): the
///   structural mutations. The editor invokes these for splits,
///   merges, and paste; the agent uses them for the same things
///   plus its own structural reasoning (add a list, split a
///   paragraph, ...).
/// - **Attribute / content** (`setBlockAttribute`,
///   `setBlockContent`, ...): fine-grained mutations that are
///   cheaper to undo than a full block replacement. The editor
///   emits a `setInlineAnnotation` per format-key-press, the agent
///   emits a `setBlockContent` when it rewrites a paragraph.
/// - **Document-level** (`setDocumentTitle`, `setDocumentMeta`):
///   metadata mutations. These do not touch the block tree.
public enum Mutation: Codable, Sendable, Hashable {
    // Block-level operations
    case insertBlockAfter(parentID: UUID?, anchorID: UUID?, block: Block)
    case insertBlocksAfter(parentID: UUID?, anchorID: UUID?, blocks: [Block])
    case replaceBlock(blockID: UUID, block: Block)
    case deleteBlock(blockID: UUID)
    case moveBlock(blockID: UUID, newParent: UUID?, newIndex: Int)

    // Attribute / content operations
    case setBlockAttribute(blockID: UUID, key: String, value: AnyCodable)
    case setBlockContent(blockID: UUID, content: [InlineRun])
    case appendInlineRun(blockID: UUID, run: InlineRun)
    case replaceInlineRun(blockID: UUID, index: Int, run: InlineRun)
    case deleteInlineRun(blockID: UUID, index: Int)
    case setInlineAnnotation(
        blockID: UUID,
        index: Int,
        annotation: InlineRun.Annotation,
        enabled: Bool
    )

    // Document-level operations
    case setDocumentTitle(title: String)
    case setDocumentMeta(key: String, value: AnyCodable)

    // MARK: - Inverse

    /// Compute the inverse of this mutation against the pre-mutation
    /// snapshot of affected blocks. The snapshot is what the
    /// `MutationEngine` captured at apply-time (a `[UUID: Block]`
    /// map of every block this mutation touched, BEFORE the
    /// mutation ran). Used by ``ReceiptUndoManager`` to build the
    /// undo receipt.
    ///
    /// The inverse is itself a `Mutation` (or a list of `Mutation`s).
    /// The mutation engine applies them in order.
    ///
    /// - For an `insertBlockAfter`, the inverse is a `deleteBlock`.
    /// - For a `deleteBlock`, the inverse is an `insertBlockAfter`
    ///   that re-inserts the deleted block at its original anchor
    ///   with its original parent.
    /// - For a `replaceBlock`, the inverse is a `replaceBlock` with
    ///   the old block (from the pre-snapshot).
    /// - For attribute mutations, the inverse carries the old value
    ///   (from the pre-snapshot).
    public func inverse(preMutation pre: [UUID: Block]) -> [Mutation] {
        switch self {
        case .insertBlockAfter(_, _, let block):
            return [.deleteBlock(blockID: block.id)]

        case .insertBlocksAfter(_, _, let blocks):
            return blocks.map { .deleteBlock(blockID: $0.id) }

        case .replaceBlock(let blockID, _):
            if let oldBlock = pre[blockID] {
                return [.replaceBlock(blockID: blockID, block: oldBlock)]
            }
            return []

        case .deleteBlock(let blockID):
            guard let oldBlock = pre[blockID] else { return [] }
            // Compute the original anchor: the block immediately
            // before oldBlock in its parent's children list, in the
            // pre-snapshot. If the block was the first child, anchor
            // is nil. We also need the original parent -- which the
            // pre-snapshot block carries via its `parentID` field.
            let parentID = oldBlock.parentID
            let siblings = oldBlock.parentID == nil
                ? Self.rootOrderFromPreSnapshot(oldBlock.id, pre: pre)
                : (pre[oldBlock.parentID!]?.children ?? [])
            let oldIndex = siblings.firstIndex(of: oldBlock.id) ?? 0
            let anchorID: UUID? = oldIndex > 0 ? siblings[oldIndex - 1] : nil
            return [.insertBlockAfter(
                parentID: parentID,
                anchorID: anchorID,
                block: oldBlock
            )]

        case .moveBlock(let blockID, let newParent, let newIndex):
            // The inverse is whatever move puts the block back where
            // it was. The pre-snapshot tells us the old parent and
            // old index; the `newParent` / `newIndex` arguments are
            // the post-move state (carried in the mutation itself).
            guard let oldBlock = pre[blockID] else { return [] }
            let oldParent = oldBlock.parentID
            let oldSiblings = oldParent == nil
                ? Self.rootOrderFromPreSnapshot(blockID, pre: pre)
                : (pre[oldParent!]?.children ?? [])
            let oldIndex = oldSiblings.firstIndex(of: blockID) ?? 0
            if oldParent == newParent && oldIndex == newIndex { return [] }
            return [.moveBlock(
                blockID: blockID,
                newParent: oldParent,
                newIndex: oldIndex
            )]

        case .setBlockAttribute(let blockID, let key, _):
            guard let oldBlock = pre[blockID],
                  let oldValue = oldBlock.attributes[key] else {
                return []
            }
            return [.setBlockAttribute(blockID: blockID, key: key, value: oldValue)]

        case .setBlockContent(let blockID, _):
            guard let oldBlock = pre[blockID] else { return [] }
            return [.setBlockContent(blockID: blockID, content: oldBlock.content)]

        case .appendInlineRun(let blockID, _):
            // The inverse of an append is to delete the last run.
            // The pre-snapshot tells us the OLD content length; the
            // new content length is OLD + 1. So we delete at index
            // (oldCount) -- the position the newly-appended run now
            // occupies.
            guard let oldBlock = pre[blockID] else { return [] }
            let newIndex = oldBlock.content.count
            return [.deleteInlineRun(blockID: blockID, index: newIndex)]

        case .replaceInlineRun(let blockID, let index, _):
            guard let oldBlock = pre[blockID],
                  index < oldBlock.content.count else {
                return []
            }
            let oldRun = oldBlock.content[index]
            return [.replaceInlineRun(blockID: blockID, index: index, run: oldRun)]

        case .deleteInlineRun(let blockID, let index):
            guard let oldBlock = pre[blockID],
                  index < oldBlock.content.count else {
                return []
            }
            let oldRun = oldBlock.content[index]
            // The inverse of delete-at-index is append. The original
            // position is lost (coarse-grained undo); this matches
            // the spec's "one receipt = one undo unit" semantics.
            return [.appendInlineRun(blockID: blockID, run: oldRun)]

        case .setInlineAnnotation(let blockID, let index, let annotation, let enabled):
            // Flip the presence flag. The pre-snapshot tells us the
            // pre-mutation state of the run's annotations.
            guard let oldBlock = pre[blockID],
                  index < oldBlock.content.count else {
                return []
            }
            let oldRun = oldBlock.content[index]
            let hadIt = oldRun.annotations.contains(annotation)
            // The mutation's `enabled` is the post-mutation state;
            // the inverse sets it to `hadIt` (the pre-mutation state).
            // We return the mutation in the "post = pre" form; the
            // engine applies it, which is the inverse of the original.
            return [.setInlineAnnotation(
                blockID: blockID,
                index: index,
                annotation: annotation,
                enabled: hadIt
            )]

        case .setDocumentTitle, .setDocumentMeta:
            // Document-level metadata is restored by the data layer
            // wrapper, not the in-memory engine. The receipt's
            // `preMutationSnapshot` includes the prior title when
            // relevant (the data layer extracts it from the entity
            // row at apply-time).
            return []
        }
    }

    /// Recover the root order (i.e. `rootChildren`) from a
    /// pre-mutation snapshot. The snapshot is a `[UUID: Block]`
    /// map; root children are the blocks whose `parentID` is nil
    /// and whose id does NOT appear as a child of any other block.
    /// This recovers the document's flat shape when the pre-state
    /// doesn't carry `rootChildren` directly.
    private static func rootOrderFromPreSnapshot(
        _ blockID: UUID,
        pre: [UUID: Block]
    ) -> [UUID] {
        // Walk all blocks in the snapshot; collect the ones that
        // are NOT children of another block. Their order is the
        // pre-mutation root order.
        let allChildIDs: Set<UUID> = pre.values.reduce(into: []) { acc, block in
            acc.formUnion(block.children)
        }
        return pre.values
            .filter { !allChildIDs.contains($0.id) }
            .map { $0.id }
    }

    /// A short human-readable description of this mutation, used in
    /// the receipt's `summary` field. The mutation engine composes
    /// multiple summaries into a one-line summary like
    /// "3 paragraphs updated, 1 list added".
    public var shortDescription: String {
        switch self {
        case .insertBlockAfter(_, _, let block):
            return "insert \(block.type.rawValue) block"
        case .insertBlocksAfter(_, _, let blocks):
            return "insert \(blocks.count) blocks"
        case .replaceBlock(_, let block):
            return "replace \(block.type.rawValue) block"
        case .deleteBlock:
            return "delete block"
        case .moveBlock:
            return "move block"
        case .setBlockAttribute(_, let key, _):
            return "set attribute '\(key)'"
        case .setBlockContent:
            return "set block content"
        case .appendInlineRun:
            return "append text"
        case .replaceInlineRun:
            return "replace text"
        case .deleteInlineRun:
            return "delete text"
        case .setInlineAnnotation(_, _, let annotation, _):
            let label: String
            switch annotation {
            case .bold: label = "bold"
            case .italic: label = "italic"
            case .underline: label = "underline"
            case .strikethrough: label = "strikethrough"
            case .code: label = "code"
            case .subscript: label = "subscript"
            case .superscript: label = "superscript"
            case .link: label = "link"
            case .color: label = "color"
            }
            return "set \(label) annotation"
        case .setDocumentTitle(let title):
            return "set title '\(title.prefix(40))'"
        case .setDocumentMeta(let key, _):
            return "set meta '\(key)'"
        }
    }

    // MARK: - Inverse helpers

    /// Compute the `insertBlockAfter` mutations that re-establish the
    /// sibling ordering when a multi-block insert is undone. For a
    /// single-block insert the empty-list case is a no-op. The
    /// `removedChildIDs` parameter is the list of block IDs that were
    /// just inserted; their relative ordering inside the parent is
    /// what we're preserving (not their position, which the next
    /// `insertBlockAfter` after the deletions puts right).
    private static func restoreContainerOrdering(
        parentID: UUID?,
        anchorID: UUID?,
        removedChildIDs: [UUID]
    ) -> [Mutation] {
        // We do not need to re-insert: the per-block `deleteBlock`
        // inverses already cover re-insertion, and the relative
        // ordering is preserved by the order of those inverses.
        return []
    }

    /// Compute the inverse for a `deleteBlock`. The deleted block
    /// must still be reachable from `document.blocks` (we look it up
    /// BEFORE the mutation is applied -- but the caller is
    /// responsible for that). The inverse is one `insertBlockAfter`
    /// that puts the block back at its original location, with
    /// `parentID` and `anchorID` derived from the block's stored
    /// `parentID` and the prior sibling order.
    private static func restoreDeletedBlock(
        _ oldBlock: Block,
        in document: DocumentAST
    ) -> [Mutation] {
        let parentID = oldBlock.parentID
        let siblings = document.children(of: parentID)
        // Find the position the block occupied (or its old index).
        // If we can't find the block in the parent (it was deleted),
        // the inverse appends to the end of the parent's children.
        let oldIndex: Int
        if let idx = siblings.firstIndex(of: oldBlock.id) {
            oldIndex = idx
        } else {
            oldIndex = siblings.count
        }
        // Anchor = the sibling immediately BEFORE oldIndex, or nil
        // for the start of the list.
        let anchorID: UUID? = oldIndex > 0 ? siblings[oldIndex - 1] : nil
        return [.insertBlockAfter(
            parentID: parentID,
            anchorID: anchorID,
            block: oldBlock
        )]
    }
}

// MARK: - MutationError

/// Errors raised by ``MutationEngine`` when a mutation cannot be
/// applied or is invalid against the current document state. The
/// undo manager surfaces these to the caller so it can decide
/// whether to retry, fail silently, or surface a user-visible
/// message.
public enum MutationError: Error, Sendable, Equatable {
    /// The mutation references a block ID that is not in the document.
    case blockNotFound(blockID: UUID)
    /// The mutation's anchor ID is not in the target parent.
    case anchorNotFound(parentID: UUID?, anchorID: UUID)
    /// The mutation's newIndex is out of range for the target parent.
    case indexOutOfRange(parentID: UUID?, index: Int, count: Int)
    /// The inline run index is out of range.
    case inlineIndexOutOfRange(blockID: UUID, index: Int, count: Int)
    /// The move would create a cycle (block moved into one of its descendants).
    case wouldCreateCycle(blockID: UUID, newParent: UUID)
    /// The block type doesn't support the requested mutation (e.g. a
    /// divider has no `content`, so `setBlockContent` is rejected).
    case invalidOperation(reason: String)
    /// Catch-all for unexpected structural problems.
    case invariantViolation(reason: String)
}

// MARK: - Array safe subscript

extension Array {
    /// `arr[safe: i]` returns nil for out-of-range indices. Used
    /// throughout the mutation engine to express "the run at this
    /// index" without a precondition-failing trap.
    fileprivate subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
