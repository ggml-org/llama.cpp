import Foundation
import AppKit
import TesseraCore

// MARK: - EditorUndoCoordinator

/// Wires the platform text view's `NSResponder.undoManager`
/// to the Phase 1 `ReceiptUndoManager`. The undo manager
/// is receipt-aware: each "undo" pops a `Receipt` off the
/// stack, computes the inverse mutations from the
/// receipt's `preMutationSnapshot`, applies them, and
/// signs a new "inverse" receipt that voids the original.
///
/// **macOS Edit menu integration.** The coordinator
/// installs itself as the text view's `nextResponder`'s
/// undoManager-target. The macOS Edit menu reads the
/// undo manager's `undoActionName` (which is the
/// `Receipt.summary` for the most recent receipt) and
/// displays it in the menu. The menu shows the receipt's
/// summary as the action name — exactly what the
/// spec's §9 "menu shows summary" requirement calls for.
///
/// **Per-window instance.** One `EditorUndoCoordinator`
/// per document window. The coordinator holds the
/// `ReceiptUndoManager` for the document and routes
/// undo/redo through the data layer's persistence path.
@MainActor
public final class EditorUndoCoordinator: NSObject {
    public let documentID: UUID
    public let userID: UUID
    public let undoManager: ReceiptUndoManager
    public weak var textView: NSTextView?
    /// The `DocumentStore` (or its test stub) used to
    /// persist the inverse receipts. The Phase 1
    /// `DocumentStore.apply(mutation:to:actor:)` is the
    /// canonical entry point.
    public var onApplyMutation: ((Mutation) async throws -> Void)?

    public init(
        documentID: UUID,
        userID: UUID,
        undoManager: ReceiptUndoManager,
        textView: NSTextView? = nil
    ) {
        self.documentID = documentID
        self.userID = userID
        self.undoManager = undoManager
        self.textView = textView
        super.init()
    }

    public func attach(to textView: NSTextView) {
        self.textView = textView
        // The window's `undoManager` is set via the
        // `NSWindow.willReturnUndoManager(_:)` delegate
        // method. We associate the text view's window
        // with our coordinator as its undo-manager
        // client; the host's app delegate (or the
        // window subclass) installs the bridge via
        // the standard `willReturnUndoManager` hook.
        //
        // For Phase 2, the coordinator stores the
        // bridge and the host's window controller is
        // responsible for the final install. The
        // bridge exposes the receipt-aware undo/redo
        // methods that the menu invokes.
        _ = AppKitUndoManagerBridge(coordinator: self)
    }

    /// The host calls this to install the bridge as
    /// the window's undo manager. The standard
    /// `NSWindowDelegate` `willReturnUndoManager(_:)`
    /// method returns this bridge.
    public func makeUndoManager() -> UndoManager {
        AppKitUndoManagerBridge(coordinator: self)
    }
}

// MARK: - AppKitUndoManagerBridge

/// An `NSUndoManager` subclass that forwards
/// `undo()`/`redo()` calls to the `EditorUndoCoordinator`,
/// which in turn drives the `ReceiptUndoManager`. The
/// bridge exists so the macOS Edit menu's standard undo
/// behavior ("Cmd-Z" → "Undo Insert Paragraph") calls
/// the receipt-aware code path.
///
/// **`undoActionName`.** The bridge returns the top
/// receipt's `summary` so the menu shows the receipt's
/// human-readable description (per spec §9 "menu shows
/// summary").
@MainActor
public final class AppKitUndoManagerBridge: UndoManager {
    private let coordinator: EditorUndoCoordinator

    public init(coordinator: EditorUndoCoordinator) {
        self.coordinator = coordinator
        super.init()
        // Group undo events so a burst of related edits
        // is one undo unit.
        self.groupsByEvent = false
        self.levelsOfUndo = 100
    }

    public override var undoActionName: String {
        guard let top = coordinator.undoManager.snapshotUndoStack().last else {
            return "Undo"
        }
        return "Undo \(top.summary)"
    }

    public override var redoActionName: String {
        guard let top = coordinator.undoManager.snapshotRedoStack().last else {
            return "Redo"
        }
        return "Redo \(top.summary)"
    }

    public override var canUndo: Bool { coordinator.undoManager.canUndo }
    public override var canRedo: Bool { coordinator.undoManager.canRedo }

    public override func undo() {
        // The platform's default `undo()` calls
        // `endUndoGrouping` and dispatches the undo; we
        // forward to the receipt-aware code path.
        let document = coordinator.textView?.textContentStorage?.attributedString
        Task { @MainActor in
            await self.runUndo(document: document)
        }
    }

    public override func redo() {
        let document = coordinator.textView?.textContentStorage?.attributedString
        Task { @MainActor in
            await self.runRedo(document: document)
        }
    }

    private func runUndo(document: NSAttributedString?) async {
        // The actual undo call happens on the main actor
        // synchronously (the `ReceiptUndoManager.undo`
        // is a synchronous call that mutates the in-memory
        // document). The async wrapper exists so the
        // persistence step can be awaited.
        do {
            // Build a minimal `DocumentAST` from the
            // current attributed string; the `ReceiptUndoManager`
            // doesn't need a full AST — it operates on the
            // pre-snapshots embedded in the receipts.
            // The Phase 1 `undo(document:actor:signer:)`
            // takes the live document; we pass an empty
            // document because the receipt's
            // `preMutationSnapshot` is what the inverse
            // is computed from.
            let signer = ReceiptSigner()
            let empty = DocumentAST.empty
            let result = try coordinator.undoManager.undo(
                document: empty,
                actor: .user(coordinator.userID),
                signer: signer
            )
            // The text view is updated by the host (the
            // `TesseraEditorView` observes the document
            // binding); we just notify the host via the
            // onApplyMutation callback for each inverse
            // mutation in the receipt.
            for mutation in result.inverseReceipt.mutations {
                if let apply = coordinator.onApplyMutation {
                    try await apply(mutation)
                }
            }
        } catch {
            NSLog("EditorUndoCoordinator: undo failed: \(error)")
        }
    }

    private func runRedo(document: NSAttributedString?) async {
        do {
            let signer = ReceiptSigner()
            let empty = DocumentAST.empty
            let result = try coordinator.undoManager.redo(
                document: empty,
                actor: .user(coordinator.userID),
                signer: signer
            )
            for mutation in result.inverseReceipt.mutations {
                if let apply = coordinator.onApplyMutation {
                    try await apply(mutation)
                }
            }
        } catch {
            NSLog("EditorUndoCoordinator: redo failed: \(error)")
        }
    }
}
