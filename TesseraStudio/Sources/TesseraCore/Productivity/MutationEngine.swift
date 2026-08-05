import Foundation

// MARK: - MutationEngine

/// Stateless applicator + validator for ``Mutation`` operations
/// against an in-memory ``DocumentAST``.
///
/// The engine is a value type with two responsibilities:
/// 1. **Validate** a mutation against the current document (so the
///    caller can pre-flight an instruction without committing).
/// 2. **Apply** a mutation to the document, throwing
///    ``MutationError`` on any invariant violation.
///
/// The engine is deliberately not an actor: it is in-memory only,
/// has no I/O, and the caller serializes access to the document
/// (the chat panel + editor + agent all share one document at a
/// time, protected by the editor's `TesseraTextContentManager`).
/// Making the engine a value type means it composes with
/// `inout DocumentAST` arguments and has no implicit shared state.
public struct MutationEngine: Sendable {

    public init() {}

    // MARK: - Validate (pre-flight)

    /// Check a mutation against the current document state. Throws
    /// ``MutationError`` if the mutation is invalid; returns
    /// silently if it would succeed.
    ///
    /// The engine validates against the document AS IT IS NOW, not
    /// as it was at apply-time. The caller is responsible for
    /// serializing access; if a mutation passes validation but fails
    /// at apply-time (because another mutation landed in between),
    /// the apply throws the same error.
    public func validate(_ mutation: Mutation, against document: DocumentAST) throws {
        switch mutation {
        case .insertBlockAfter(let parentID, let anchorID, _):
            try validateInsertion(parentID: parentID, anchorID: anchorID, in: document)

        case .insertBlocksAfter(let parentID, let anchorID, let blocks):
            // Validate insertion location ONCE (not per block; the
            // anchor and parent are shared).
            try validateInsertion(parentID: parentID, anchorID: anchorID, in: document)
            // All new blocks must be unique IDs.
            var seen = Set<UUID>()
            for block in blocks {
                if document.blocks[block.id] != nil {
                    throw MutationError.invalidOperation(
                        reason: "block \(block.id) already exists in document"
                    )
                }
                if !seen.insert(block.id).inserted {
                    throw MutationError.invalidOperation(
                        reason: "duplicate block id \(block.id) in insertBlocksAfter"
                    )
                }
            }

        case .replaceBlock(let blockID, let newBlock):
            guard document.blocks[blockID] != nil else {
                throw MutationError.blockNotFound(blockID: blockID)
            }
            // The replacement must keep the same id (the operation
            // is "change the block at this id", not "substitute a
            // different block with a different id").
            if newBlock.id != blockID {
                throw MutationError.invalidOperation(
                    reason: "replaceBlock id mismatch: \(blockID) vs \(newBlock.id)"
                )
            }

        case .deleteBlock(let blockID):
            guard document.blocks[blockID] != nil else {
                throw MutationError.blockNotFound(blockID: blockID)
            }

        case .moveBlock(let blockID, let newParent, let newIndex):
            guard let block = document.blocks[blockID] else {
                throw MutationError.blockNotFound(blockID: blockID)
            }
            // Cycle check: cannot move a block into one of its descendants.
            if let newParent, isAncestor(of: newParent, in: document, root: blockID) {
                throw MutationError.wouldCreateCycle(blockID: blockID, newParent: newParent)
            }
            try validateContainerIndex(parentID: newParent, index: newIndex, in: document)

        case .setBlockAttribute(let blockID, _, _):
            guard document.blocks[blockID] != nil else {
                throw MutationError.blockNotFound(blockID: blockID)
            }

        case .setBlockContent(let blockID, _):
            guard document.blocks[blockID] != nil else {
                throw MutationError.blockNotFound(blockID: blockID)
            }
            // Divider blocks cannot have content.
            if document.blocks[blockID]?.type == .divider {
                throw MutationError.invalidOperation(
                    reason: "divider blocks cannot have content"
                )
            }

        case .appendInlineRun(let blockID, _):
            guard document.blocks[blockID] != nil else {
                throw MutationError.blockNotFound(blockID: blockID)
            }

        case .replaceInlineRun(let blockID, let index, _):
            let count = document.blocks[blockID]?.content.count ?? 0
            guard document.blocks[blockID] != nil else {
                throw MutationError.blockNotFound(blockID: blockID)
            }
            guard index >= 0 && index < count else {
                throw MutationError.inlineIndexOutOfRange(blockID: blockID, index: index, count: count)
            }

        case .deleteInlineRun(let blockID, let index):
            let count = document.blocks[blockID]?.content.count ?? 0
            guard document.blocks[blockID] != nil else {
                throw MutationError.blockNotFound(blockID: blockID)
            }
            guard index >= 0 && index < count else {
                throw MutationError.inlineIndexOutOfRange(blockID: blockID, index: index, count: count)
            }

        case .setInlineAnnotation(let blockID, let index, _, _):
            let count = document.blocks[blockID]?.content.count ?? 0
            guard document.blocks[blockID] != nil else {
                throw MutationError.blockNotFound(blockID: blockID)
            }
            guard index >= 0 && index < count else {
                throw MutationError.inlineIndexOutOfRange(blockID: blockID, index: index, count: count)
            }

        case .setDocumentTitle, .setDocumentMeta:
            // Document-level metadata does not touch the AST.
            return
        }
    }

    // MARK: - Apply (mutating)

    /// Apply a mutation to the document, in place. Throws
    /// ``MutationError`` if the mutation is invalid. The
    /// mutation is applied atomically (the document is either
    /// fully updated or unchanged).
    ///
    /// Returns a `[UUID: Block]` snapshot of the blocks the
    /// mutation touched, in their PRE-mutation state. The
    /// snapshot is what the receipt needs to undo itself; it
    /// must be captured at apply-time, not at undo-time, because
    /// the undo may run after the document has been further
    /// mutated.
    @discardableResult
    public mutating func apply(_ mutation: Mutation, to document: inout DocumentAST) throws -> [UUID: Block] {
        // Validate first; if it fails, the document is untouched.
        try validate(mutation, against: document)
        // Capture the pre-mutation snapshot of every block the
        // mutation is about to touch. The set of "touched" blocks
        // depends on the mutation; we capture conservatively (the
        // mutation may reference more than it directly mutates,
        // e.g. a `moveBlock` touches the block AND its old parent
        // AND its new parent).
        let pre = Self.capturePreSnapshot(of: mutation, in: document)
        switch mutation {
        case .insertBlockAfter(let parentID, let anchorID, var block):
            block.parentID = parentID
            document.blocks[block.id] = block
            insertChild(block.id, into: parentID, after: anchorID, in: &document)

        case .insertBlocksAfter(let parentID, let anchorID, let blocks):
            // Insert in the given order, each one anchoring after the
            // previously-inserted block.
            var currentAnchor = anchorID
            for var block in blocks {
                block.parentID = parentID
                document.blocks[block.id] = block
                insertChild(block.id, into: parentID, after: currentAnchor, in: &document)
                currentAnchor = block.id
            }

        case .replaceBlock(let blockID, let newBlock):
            document.blocks[blockID] = newBlock

        case .deleteBlock(let blockID):
            // Remove the block and detach it from its parent's
            // children list.
            if let removed = document.blocks.removeValue(forKey: blockID) {
                removeChild(blockID, from: removed.parentID, in: &document)
            }

        case .moveBlock(let blockID, let newParent, let newIndex):
            guard var block = document.blocks[blockID] else { return pre }
            // Detach from old parent.
            let oldParent = block.parentID
            removeChild(blockID, from: oldParent, in: &document)
            // Update the parent pointer and insert at the new index.
            block.parentID = newParent
            document.blocks[blockID] = block
            insertChildAt(blockID, into: newParent, at: newIndex, in: &document)

        case .setBlockAttribute(let blockID, let key, let value):
            document.blocks[blockID]?.attributes[key] = value

        case .setBlockContent(let blockID, let content):
            document.blocks[blockID]?.content = content

        case .appendInlineRun(let blockID, let run):
            document.blocks[blockID]?.content.append(run)

        case .replaceInlineRun(let blockID, let index, let run):
            guard index >= 0, index < (document.blocks[blockID]?.content.count ?? 0) else {
                throw MutationError.inlineIndexOutOfRange(
                    blockID: blockID,
                    index: index,
                    count: document.blocks[blockID]?.content.count ?? 0
                )
            }
            document.blocks[blockID]?.content[index] = run

        case .deleteInlineRun(let blockID, let index):
            guard index >= 0, index < (document.blocks[blockID]?.content.count ?? 0) else {
                throw MutationError.inlineIndexOutOfRange(
                    blockID: blockID,
                    index: index,
                    count: document.blocks[blockID]?.content.count ?? 0
                )
            }
            document.blocks[blockID]?.content.remove(at: index)

        case .setInlineAnnotation(let blockID, let index, let annotation, let enabled):
            guard index >= 0, index < (document.blocks[blockID]?.content.count ?? 0) else {
                throw MutationError.inlineIndexOutOfRange(
                    blockID: blockID,
                    index: index,
                    count: document.blocks[blockID]?.content.count ?? 0
                )
            }
            applyAnnotation(annotation, enabled: enabled, to: &document.blocks[blockID]!.content[index])

        case .setDocumentTitle, .setDocumentMeta:
            // Document-level metadata; the AST has no field for it.
            // The data layer wrapper is responsible for persisting.
            return pre
        }
        return pre
    }

    /// Capture the pre-mutation snapshot of every block the
    /// mutation is about to touch. The "touched" set is the block
    /// being mutated, plus its parent(s) (whose `children` array
    /// changes).
    private static func capturePreSnapshot(
        of mutation: Mutation,
        in document: DocumentAST
    ) -> [UUID: Block] {
        var out: [UUID: Block] = [:]
        let capture: (UUID) -> Void = { id in
            if let block = document.blocks[id] {
                out[id] = block
            }
        }
        switch mutation {
        case .insertBlockAfter(let parentID, _, _):
            if let pid = parentID { capture(pid) }
        case .insertBlocksAfter(let parentID, _, _):
            if let pid = parentID { capture(pid) }
        case .replaceBlock(let blockID, _):
            capture(blockID)
        case .deleteBlock(let blockID):
            capture(blockID)
        case .moveBlock(let blockID, let newParent, _):
            capture(blockID)
            if let oldParent = document.blocks[blockID]?.parentID {
                capture(oldParent)
            }
            if let newParent { capture(newParent) }
        case .setBlockAttribute(let blockID, _, _):
            capture(blockID)
        case .setBlockContent(let blockID, _):
            capture(blockID)
        case .appendInlineRun(let blockID, _):
            capture(blockID)
        case .replaceInlineRun(let blockID, _, _):
            capture(blockID)
        case .deleteInlineRun(let blockID, _):
            capture(blockID)
        case .setInlineAnnotation(let blockID, _, _, _):
            capture(blockID)
        case .setDocumentTitle, .setDocumentMeta:
            break
        }
        return out
    }

    // MARK: - Internals: structural helpers

    private func validateInsertion(
        parentID: UUID?,
        anchorID: UUID?,
        in document: DocumentAST
    ) throws {
        if let parentID, document.blocks[parentID] == nil {
            // A nil parent is the document root (legal). A non-nil
            // parent must exist.
            throw MutationError.blockNotFound(blockID: parentID)
        }
        if let anchorID {
            // The anchor must be a child of the same parent (or, if
            // parent is nil, a root child). It must exist.
            guard let anchorBlock = document.blocks[anchorID] else {
                throw MutationError.blockNotFound(blockID: anchorID)
            }
            let siblings = document.children(of: parentID)
            if !siblings.contains(anchorID) {
                throw MutationError.anchorNotFound(parentID: parentID, anchorID: anchorID)
            }
            // The anchor's parent must match the insertion parent.
            // (If the caller passed a mismatched anchor, the new
            // block would be placed in a parent the anchor isn't in,
            // which is a programmer error.)
            if anchorBlock.parentID != parentID {
                throw MutationError.anchorNotFound(parentID: parentID, anchorID: anchorID)
            }
        }
    }

    private func validateContainerIndex(
        parentID: UUID?,
        index: Int,
        in document: DocumentAST
    ) throws {
        if let parentID, document.blocks[parentID] == nil {
            throw MutationError.blockNotFound(blockID: parentID)
        }
        let siblings = document.children(of: parentID)
        // Index must be in [0, count]. count is the "insert at end"
        // position; count+1 is out of range.
        guard index >= 0 && index <= siblings.count else {
            throw MutationError.indexOutOfRange(
                parentID: parentID,
                index: index,
                count: siblings.count
            )
        }
    }

    /// `true` if `candidate` is a descendant of `root` in the tree
    /// (including `root` itself). Used by the cycle check.
    private func isAncestor(
        of candidate: UUID,
        in document: DocumentAST,
        root: UUID
    ) -> Bool {
        // Walk downward from `root` looking for `candidate`.
        var stack: [UUID] = [root]
        while let id = stack.popLast() {
            if id == candidate { return true }
            if let block = document.blocks[id] {
                stack.append(contentsOf: block.children)
            }
        }
        return false
    }

    /// Insert `childID` into the children list of `parentID` (or
    /// `rootChildren` if `parentID` is nil), immediately after
    /// `anchorID`. If `anchorID` is nil, prepend to the front of the
    /// list.
    private func insertChild(
        _ childID: UUID,
        into parentID: UUID?,
        after anchorID: UUID?,
        in document: inout DocumentAST
    ) {
        if let parentID {
            var children = document.blocks[parentID]?.children ?? []
            insertIntoArray(&children, item: childID, after: anchorID)
            document.blocks[parentID]?.children = children
        } else {
            insertIntoArray(&document.rootChildren, item: childID, after: anchorID)
        }
    }

    /// Insert at an absolute index (used by `moveBlock`).
    private func insertChildAt(
        _ childID: UUID,
        into parentID: UUID?,
        at index: Int,
        in document: inout DocumentAST
    ) {
        if let parentID {
            var children = document.blocks[parentID]?.children ?? []
            let safe = max(0, min(index, children.count))
            children.insert(childID, at: safe)
            document.blocks[parentID]?.children = children
        } else {
            let safe = max(0, min(index, document.rootChildren.count))
            document.rootChildren.insert(childID, at: safe)
        }
    }

    private func removeChild(
        _ childID: UUID,
        from parentID: UUID?,
        in document: inout DocumentAST
    ) {
        if let parentID {
            document.blocks[parentID]?.children.removeAll { $0 == childID }
        } else {
            document.rootChildren.removeAll { $0 == childID }
        }
    }

    private func insertIntoArray(
        _ array: inout [UUID],
        item: UUID,
        after anchorID: UUID?
    ) {
        if let anchorID, let idx = array.firstIndex(of: anchorID) {
            array.insert(item, at: idx + 1)
        } else {
            array.insert(item, at: 0)
        }
    }

    private func applyAnnotation(
        _ annotation: InlineRun.Annotation,
        enabled: Bool,
        to run: inout InlineRun
    ) {
        if enabled {
            if !run.annotations.contains(annotation) {
                run.annotations.append(annotation)
            }
        } else {
            run.annotations.removeAll { $0 == annotation }
        }
    }
}
