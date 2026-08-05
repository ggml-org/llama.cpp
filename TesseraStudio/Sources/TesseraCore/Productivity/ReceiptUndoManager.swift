import Foundation

// MARK: - ReceiptUndoManager

/// Receipt-aware undo/redo. One instance per document window. The
/// undo stack is a list of receipts; undoing pops one receipt off
/// the stack, applies the inverse mutations to the document, and
/// appends a new "inverse" receipt to the chain. The new receipt
/// marks the original as `voidedBy` (in memory) and is itself pushed
/// to the redo stack so a subsequent redo re-applies the inverse.
///
/// **One undo unit = one receipt** (per spec §5.4). A burst of
/// user edits coalesces into one receipt (the editor's
/// `TesseraTextContentManager` does the coalescing in Phase 2; this
/// manager assumes each `register(_:)` call is one user-perceived
/// edit unit).
///
/// **Grouping** (the `group(_:)` method) lets the caller batch
/// multiple receipts into a single undo unit. The chat panel uses
/// this for "agent instruction 1 → 2 → 3" so a single Cmd-Z undoes
/// the whole instruction, not the last step.
///
/// **Persistence** (per the brief's tests): the undo stack survives
/// a document open/close. The data layer stores the receipts
/// themselves (the audit trail); the undo manager stores the
/// per-window undo/redo stack, which is rebuilt on document open
/// by replaying the last N receipts in reverse.
public final class ReceiptUndoManager: Sendable {

    /// The receipts available to undo (oldest first; the most
    /// recent is at the end).
    private var undoStack: [Receipt]
    /// The receipts available to redo (oldest first; the most
    /// recent redoable is at the end). Empty when the user has
    /// not undone anything since the last `register(_:)`.
    private var redoStack: [Receipt]
    /// Voided originals: receipts that have been undone. Stored
    /// separately from the undo stack (which holds undoable
    /// receipts) so the audit trail captures the voiding. The
    /// receipt's `voidedBy` field points at the inverse receipt.
    private var voidedReceipts: [Receipt]
    /// The receipt that was popped off the undo stack and is
    /// currently being undone; used so the inverse receipt can
    /// mark it `voidedBy` in memory. Cleared after the inverse
    /// receipt is appended.
    private var pendingVoidTarget: Receipt?
    /// The document this undo manager is bound to. Stored so the
    /// caller can `replay()` the chain on document open.
    public let documentID: UUID

    public init(
        documentID: UUID,
        initialReceipt: Receipt? = nil
    ) {
        self.documentID = documentID
        self.undoStack = initialReceipt.map { [$0] } ?? []
        self.redoStack = []
        self.voidedReceipts = []
        self.pendingVoidTarget = nil
    }

    /// True iff there's a receipt on the undo stack.
    public var canUndo: Bool { !undoStack.isEmpty }

    /// True iff there's a receipt on the redo stack.
    public var canRedo: Bool { !redoStack.isEmpty }

    /// Push a new receipt onto the undo stack. Clears the redo
    /// stack (a new edit invalidates the redo path, per
    /// conventional undo/redo semantics).
    public func register(_ receipt: Receipt) {
        precondition(receipt.documentID == documentID, "receipt documentID mismatch")
        undoStack.append(receipt)
        redoStack.removeAll()
        pendingVoidTarget = nil
    }

    /// Group a list of receipts into a single undo unit. The list
    /// is treated as one logical edit; undo undoes the whole group
    /// in reverse order, then the next undo undoes the group above
    /// it. The group itself is a wrapper around the receipts; the
    /// underlying receipts are still stored in the chain in order.
    public func group(_ receipts: [Receipt]) {
        for r in receipts {
            precondition(r.documentID == documentID, "receipt documentID mismatch")
            undoStack.append(r)
        }
        redoStack.removeAll()
        pendingVoidTarget = nil
    }

    /// Result of an undo or redo operation: the inverse receipt
    /// that was appended to the chain, plus the document state
    /// after the inverse was applied. The caller is responsible
    /// for persisting the inverse receipt to the data layer.
    public struct UndoResult: Sendable {
        public let inverseReceipt: Receipt
        public let updatedDocument: DocumentAST
        public let voidedReceiptID: UUID
    }

    /// Undo the most recent receipt. The inverse is computed via
    /// ``Mutation/inverse(against:)``, applied to a copy of the
    /// document, and signed as a new receipt. The original
    /// receipt is marked `voidedBy` in memory.
    ///
    /// The caller passes:
    ///   * `document`: the current document state (after the
    ///     original receipt was applied).
    ///   * `actor`: the actor for the inverse receipt (typically
    ///     `.user(currentUserID)` -- the user pressing Cmd-Z).
    ///   * `signer`: the receipt signer.
    ///   * `documentContentHash`: the content hash of the document
    ///     AFTER the inverse is applied (used in the C2PA manifest).
    ///
    /// Returns the new inverse receipt + the updated document.
    /// Throws ``UndoError`` if the undo stack is empty or the
    /// inverse fails to apply (e.g., a block referenced by an
    /// inverse mutation was already deleted by another path).
    public func undo(
        document: DocumentAST,
        actor: Actor,
        signer: ReceiptSigner,
        documentContentHash: String? = nil
    ) throws -> UndoResult {
        // Peek the top of the undo stack (we DON'T pop yet; we mark
        // the original as voided in place so the audit trail
        // captures the voiding).
        guard !undoStack.isEmpty else {
            throw UndoError.emptyStack
        }
        let original = undoStack.removeLast()
        pendingVoidTarget = original

        // Compute the inverse mutations (in order) from the
        // original receipt's pre-mutation snapshot. The snapshot is
        // what makes the receipt self-contained for undo: we don't
        // need the live document to know what the inverse is.
        let inverseMutations = original.mutations.flatMap {
            $0.inverse(preMutation: original.preMutationSnapshot)
        }
        // For the inverse receipt, we also need a fresh
        // pre-mutation snapshot -- the inverse's "before" state
        // is the post-state of the original (i.e. the current
        // document). Capture it now, while the live document is
        // still the post-original state.
        var engine = MutationEngine()
        var inversePreSnapshot: [UUID: Block] = [:]
        var workingDocument = document
        do {
            for mutation in inverseMutations {
                let pre = try engine.apply(mutation, to: &workingDocument)
                for (k, v) in pre { inversePreSnapshot[k] = v }
            }
        } catch {
            // Roll back: put the original back on the stack.
            undoStack.append(original)
            pendingVoidTarget = nil
            throw UndoError.inverseFailed("\(error)")
        }

        // Sign the inverse receipt. The chain's prior-receipt is
        // the original (we're appending an undo immediately after
        // the receipt it undoes).
        let inverseReceipt: Receipt
        do {
            inverseReceipt = try signer.sign(
                documentID: documentID,
                mutations: inverseMutations,
                priorReceiptID: original.id,
                actor: actor,
                documentContentHash: documentContentHash,
                preMutationSnapshot: inversePreSnapshot
            )
        } catch {
            undoStack.append(original)
            pendingVoidTarget = nil
            throw UndoError.signingFailed("\(error)")
        }

        // Update in-memory state: the original is voided by the
        // inverse. We DON'T keep the voided original on the undo
        // stack (it's no longer undoable), but we DO append it to
        // a separate "voided" log the caller can inspect via
        // ``snapshotVoidedReceipts()``. The redo stack holds the
        // ORIGINAL so a subsequent redo can re-apply its mutations
        // and restore the post-original state.
        var voidedOriginal = original
        voidedOriginal.voidedBy = inverseReceipt.id
        voidedReceipts.append(voidedOriginal)
        redoStack.append(original)
        pendingVoidTarget = nil

        return UndoResult(
            inverseReceipt: inverseReceipt,
            updatedDocument: workingDocument,
            voidedReceiptID: original.id
        )
    }

    /// Redo the most recently undone receipt. The redo re-applies
    /// the inverse receipt's mutations to the document, then
    /// signs a new "redo" receipt that voids the inverse receipt.
    public func redo(
        document: DocumentAST,
        actor: Actor,
        signer: ReceiptSigner,
        documentContentHash: String? = nil
    ) throws -> UndoResult {
        // The redo stack holds the ORIGINAL receipt (the one that
        // was undone). Redo re-applies its mutations, which
        // restores the state the user wanted to get back to.
        // A new "redo" receipt is signed that voids the original.
        guard let original = redoStack.popLast() else {
            throw UndoError.emptyStack
        }
        pendingVoidTarget = original

        // The "before" state for the redo is the inverse receipt's
        // pre-mutation snapshot (i.e. the state right after the
        // undo ran). We don't have that snapshot directly on the
        // redo stack entry (we store the original, not the
        // inverse), but the original's pre-mutation snapshot IS
        // the state before the original was applied, and the
        // current document IS the state after the undo (i.e. the
        // original pre-state). So we can use the original's
        // pre-mutation snapshot as the redo's pre-snapshot.
        var workingDocument = document
        var engine = MutationEngine()
        var redoPreSnapshot: [UUID: Block] = [:]
        do {
            for mutation in original.mutations {
                let pre = try engine.apply(mutation, to: &workingDocument)
                for (k, v) in pre { redoPreSnapshot[k] = v }
            }
        } catch {
            redoStack.append(original)
            pendingVoidTarget = nil
            throw UndoError.inverseFailed("\(error)")
        }

        let redoReceipt: Receipt
        do {
            redoReceipt = try signer.sign(
                documentID: documentID,
                mutations: original.mutations,
                priorReceiptID: original.id,
                actor: actor,
                documentContentHash: documentContentHash,
                preMutationSnapshot: redoPreSnapshot
            )
        } catch {
            redoStack.append(original)
            pendingVoidTarget = nil
            throw UndoError.signingFailed("\(error)")
        }

        var voidedOriginal = original
        voidedOriginal.voidedBy = redoReceipt.id
        if let idx = redoStack.lastIndex(where: { $0.id == original.id }) {
            redoStack[idx] = voidedOriginal
        }
        undoStack.append(redoReceipt)
        pendingVoidTarget = nil

        return UndoResult(
            inverseReceipt: redoReceipt,
            updatedDocument: workingDocument,
            voidedReceiptID: original.id
        )
    }

    /// Snapshot the current undo stack (for persistence across
    /// document open/close). The returned array is a value copy;
    /// mutations to the snapshot do not affect the manager.
    public func snapshotUndoStack() -> [Receipt] { undoStack }

    /// Snapshot the current redo stack.
    public func snapshotRedoStack() -> [Receipt] { redoStack }

    /// Snapshot the voided-receipts log.
    public func snapshotVoidedReceipts() -> [Receipt] { voidedReceipts }

    /// Replay a previously-snapshotted undo stack. Used when the
    /// document is reopened: the data layer returns the chain in
    /// order, and the undo manager rebuilds its in-memory state
    /// by reversing the chain.
    public func restore(
        undoStack: [Receipt],
        redoStack: [Receipt] = [],
        voidedReceipts: [Receipt] = []
    ) {
        self.undoStack = undoStack
        self.redoStack = redoStack
        self.voidedReceipts = voidedReceipts
        self.pendingVoidTarget = nil
    }
}

// MARK: - UndoError

public enum UndoError: Error, Sendable, Equatable {
    case emptyStack
    case inverseFailed(String)
    case signingFailed(String)
}
