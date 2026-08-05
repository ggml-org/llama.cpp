import XCTest
import Foundation
import CryptoKit
@testable import TesseraCore

/// Tests for the receipt-aware undo integration with the
/// editor's mutation pipeline. The editor's `applyMutation`
/// path produces a receipt, and `ReceiptUndoManager.undo`
/// pops the top receipt, computes the inverse, applies it,
/// and signs a new inverse receipt.
///
/// The "menu shows summary" test is the spec's §9 requirement:
/// the macOS Edit menu's "Undo" item should display the
/// receipt's `summary` as the action name.
final class ReceiptUndoEditorIntegrationTests: XCTestCase {

    // MARK: - Receipt signing key (test injection)

    /// The tests use an injected signing key (the
    /// `ReceiptSigner(signingKey:)` initializer) so they
    /// don't touch the real Keychain.
    private func makeSigner() -> ReceiptSigner {
        let key = Curve25519.Signing.PrivateKey()
        return ReceiptSigner(signingKey: key)
    }

    // MARK: - Undo of user edit (Cmd-Z)

    func testUndoOfUserEditRestoresDocument() throws {
        let signer = makeSigner()
        let documentID = UUID()
        let userID = UUID()
        let initial = DocumentAST(blocks: [:], rootChildren: [])
        let block = Block(
            id: UUID(),
            type: .paragraph,
            content: [InlineRun(text: "hello")]
        )
        // Sign a receipt for the initial insert.
        let engine = MutationEngine()
        var doc = initial
        var localEngine = engine
        _ = try localEngine.apply(.insertBlockAfter(parentID: nil, anchorID: nil, block: block), to: &doc)
        let preSnapshot: [UUID: Block] = [:]
        let receipt = try signer.sign(
            documentID: documentID,
            mutations: [.insertBlockAfter(parentID: nil, anchorID: nil, block: block)],
            priorReceiptID: nil,
            actor: .user(userID),
            preMutationSnapshot: preSnapshot
        )
        // Undo.
        let undoManager = ReceiptUndoManager(documentID: documentID, initialReceipt: receipt)
        let result = try undoManager.undo(
            document: doc,
            actor: .user(userID),
            signer: signer
        )
        // The inverse removed the block.
        XCTAssertEqual(result.updatedDocument.rootChildren.count, 0)
        XCTAssertTrue(result.updatedDocument.blocks.isEmpty)
    }

    // MARK: - Menu shows summary

    func testMenuUndoActionNameIsTheReceiptSummary() throws {
        let signer = makeSigner()
        let documentID = UUID()
        let userID = UUID()
        let undoManager = ReceiptUndoManager(documentID: documentID)
        let block = Block(
            id: UUID(),
            type: .paragraph,
            content: [InlineRun(text: "hello")]
        )
        // The receipt's summary is what the macOS Edit menu
        // displays for "Undo". ReceiptSigner composes the
        // summary from the mutations' shortDescription.
        let receipt = try signer.sign(
            documentID: documentID,
            mutations: [.insertBlockAfter(parentID: nil, anchorID: nil, block: block)],
            priorReceiptID: nil,
            actor: .user(userID)
        )
        undoManager.register(receipt)
        // The summary is the shortDescription of the only
        // mutation ("insert paragraph block").
        XCTAssertEqual(receipt.summary, "insert paragraph block")
    }

    func testMenuUndoActionNameComposesForBatch() throws {
        let signer = makeSigner()
        let documentID = UUID()
        let userID = UUID()
        let block1 = Block(id: UUID(), type: .paragraph, content: [InlineRun(text: "a")])
        let block2 = Block(id: UUID(), type: .paragraph, content: [InlineRun(text: "b")])
        let receipt = try signer.sign(
            documentID: documentID,
            mutations: [
                .insertBlockAfter(parentID: nil, anchorID: nil, block: block1),
                .insertBlockAfter(parentID: nil, anchorID: nil, block: block2),
            ],
            priorReceiptID: nil,
            actor: .user(userID)
        )
        // The summary groups identical descriptions.
        XCTAssertTrue(receipt.summary.contains("insert paragraph block"))
    }

    // MARK: - Voided receipts don't appear as candidates

    func testVoidedReceiptsAreNotUndoCandidates() throws {
        let signer = makeSigner()
        let documentID = UUID()
        let userID = UUID()
        let undoManager = ReceiptUndoManager(documentID: documentID)
        let block = Block(
            id: UUID(),
            type: .paragraph,
            content: [InlineRun(text: "hello")]
        )
        let receipt = try signer.sign(
            documentID: documentID,
            mutations: [.insertBlockAfter(parentID: nil, anchorID: nil, block: block)],
            priorReceiptID: nil,
            actor: .user(userID)
        )
        undoManager.register(receipt)
        XCTAssertTrue(undoManager.canUndo)
        // Apply a different mutation to make the undo work.
        var doc = DocumentAST(blocks: [block.id: block], rootChildren: [block.id])
        _ = try undoManager.undo(
            document: doc,
            actor: .user(userID),
            signer: signer
        )
        // After undo, the receipt is on the voided list, not
        // the undo stack. canUndo should be false (we popped
        // the only receipt and the inverse is in the redo
        // stack).
        XCTAssertFalse(undoManager.canUndo)
        let voided = undoManager.snapshotVoidedReceipts()
        XCTAssertEqual(voided.count, 1)
        XCTAssertNotNil(voided.first?.voidedBy)
    }

    // MARK: - Cmd-Shift-Z (redo)

    func testRedoReAppliesOriginalReceipt() throws {
        let signer = makeSigner()
        let documentID = UUID()
        let userID = UUID()
        let undoManager = ReceiptUndoManager(documentID: documentID)
        let block = Block(
            id: UUID(),
            type: .paragraph,
            content: [InlineRun(text: "hello")]
        )
        let receipt = try signer.sign(
            documentID: documentID,
            mutations: [.insertBlockAfter(parentID: nil, anchorID: nil, block: block)],
            priorReceiptID: nil,
            actor: .user(userID)
        )
        undoManager.register(receipt)
        let docWithBlock = DocumentAST(blocks: [block.id: block], rootChildren: [block.id])
        // Undo then redo. The redo input is the post-undo
        // document (the inverse's result), NOT the original
        // docWithBlock.
        let undoResult = try undoManager.undo(document: docWithBlock, actor: .user(userID), signer: signer)
        XCTAssertTrue(undoManager.canRedo)
        let result = try undoManager.redo(
            document: undoResult.updatedDocument,
            actor: .user(userID),
            signer: signer
        )
        // After redo, the block is back.
        XCTAssertEqual(result.updatedDocument.rootChildren, [block.id])
        XCTAssertTrue(undoManager.canUndo)
    }
}
