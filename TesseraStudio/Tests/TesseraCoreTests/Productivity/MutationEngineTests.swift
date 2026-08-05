import XCTest
@testable import TesseraCore

/// Tests for the Mutation API: each mutation variant applies
/// correctly, validation rejects invalid mutations, composition
/// of N mutations produces the expected final state, edge cases
/// (empty document, deep nesting, cyclic moves rejected).
final class MutationEngineTests: XCTestCase {

    // MARK: - insertBlockAfter

    func testInsertBlockAfterAtRoot() throws {
        var doc = DocumentAST.empty
        var engine = MutationEngine()
        let block = Block(type: .paragraph, content: [InlineRun(text: "hi")])
        try engine.apply(
            .insertBlockAfter(parentID: nil, anchorID: nil, block: block),
            to: &doc
        )
        XCTAssertEqual(doc.rootChildren, [block.id])
        XCTAssertEqual(doc.blocks[block.id]?.content.first?.text, "hi")
    }

    func testInsertBlockAfterExistingAnchor() throws {
        let first = Block(type: .paragraph)
        var doc = DocumentAST(blocks: [first.id: first], rootChildren: [first.id])
        var engine = MutationEngine()
        let second = Block(type: .paragraph)
        try engine.apply(
            .insertBlockAfter(parentID: nil, anchorID: first.id, block: second),
            to: &doc
        )
        XCTAssertEqual(doc.rootChildren, [first.id, second.id])
    }

    func testInsertBlockAfterWithParent() throws {
        let parent = Block(type: .list)
        let firstChild = Block(type: .listItem, parentID: parent.id)
        var doc = DocumentAST(
            blocks: [parent.id: parent, firstChild.id: firstChild],
            rootChildren: [parent.id]
        )
        doc.blocks[parent.id]?.children = [firstChild.id]
        var engine = MutationEngine()
        let secondChild = Block(type: .listItem, parentID: parent.id)
        try engine.apply(
            .insertBlockAfter(parentID: parent.id, anchorID: firstChild.id, block: secondChild),
            to: &doc
        )
        XCTAssertEqual(doc.blocks[parent.id]?.children, [firstChild.id, secondChild.id])
    }

    // MARK: - insertBlocksAfter

    func testInsertBlocksAfter() throws {
        var doc = DocumentAST.empty
        var engine = MutationEngine()
        let blocks = (0..<3).map { _ in Block(type: .paragraph) }
        try engine.apply(
            .insertBlocksAfter(parentID: nil, anchorID: nil, blocks: blocks),
            to: &doc
        )
        XCTAssertEqual(doc.rootChildren.map { $0 }, blocks.map { $0.id })
    }

    func testInsertBlocksAfterRejectsDuplicates() {
        var doc = DocumentAST.empty
        var engine = MutationEngine()
        let id = UUID()
        let a = Block(id: id, type: .paragraph)
        let b = Block(id: id, type: .paragraph)
        XCTAssertThrowsError(try engine.apply(
            .insertBlocksAfter(parentID: nil, anchorID: nil, blocks: [a, b]),
            to: &doc
        ))
    }

    // MARK: - replaceBlock

    func testReplaceBlock() throws {
        let original = Block(type: .paragraph, content: [InlineRun(text: "old")])
        let doc = DocumentAST(blocks: [original.id: original], rootChildren: [original.id])
        var working = doc
        var engine = MutationEngine()
        let replacement = Block(
            id: original.id,
            type: .paragraph,
            content: [InlineRun(text: "new")]
        )
        try engine.apply(.replaceBlock(blockID: original.id, block: replacement), to: &working)
        XCTAssertEqual(working.blocks[original.id]?.content.first?.text, "new")
    }

    func testReplaceBlockRejectsIDMismatch() {
        let original = Block(type: .paragraph)
        let doc = DocumentAST(blocks: [original.id: original], rootChildren: [original.id])
        var working = doc
        var engine = MutationEngine()
        let replacement = Block(type: .paragraph)
        XCTAssertThrowsError(try engine.apply(
            .replaceBlock(blockID: original.id, block: replacement),
            to: &working
        ))
    }

    // MARK: - deleteBlock

    func testDeleteBlock() throws {
        let block = Block(type: .paragraph)
        var doc = DocumentAST(blocks: [block.id: block], rootChildren: [block.id])
        var engine = MutationEngine()
        try engine.apply(.deleteBlock(blockID: block.id), to: &doc)
        XCTAssertFalse(doc.contains(block.id))
        XCTAssertTrue(doc.rootChildren.isEmpty)
    }

    func testDeleteBlockAlsoDetachesFromParent() throws {
        let parent = Block(type: .list)
        let child = Block(type: .listItem, parentID: parent.id)
        var doc = DocumentAST(
            blocks: [parent.id: parent, child.id: child],
            rootChildren: [parent.id]
        )
        doc.blocks[parent.id]?.children = [child.id]
        var engine = MutationEngine()
        try engine.apply(.deleteBlock(blockID: child.id), to: &doc)
        XCTAssertEqual(doc.blocks[parent.id]?.children, [])
    }

    // MARK: - moveBlock

    func testMoveBlockReorder() throws {
        let a = Block(type: .paragraph)
        let b = Block(type: .paragraph)
        let c = Block(type: .paragraph)
        var doc = DocumentAST(
            blocks: [a.id: a, b.id: b, c.id: c],
            rootChildren: [a.id, b.id, c.id]
        )
        var engine = MutationEngine()
        // Move `c` to the front (index 0).
        try engine.apply(
            .moveBlock(blockID: c.id, newParent: nil, newIndex: 0),
            to: &doc
        )
        XCTAssertEqual(doc.rootChildren, [c.id, a.id, b.id])
    }

    func testMoveBlockToContainer() throws {
        let list = Block(type: .list)
        let item = Block(type: .paragraph)
        var doc = DocumentAST(
            blocks: [list.id: list, item.id: item],
            rootChildren: [list.id, item.id]
        )
        doc.blocks[list.id]?.children = []
        var engine = MutationEngine()
        // Move `item` into the list at the end.
        try engine.apply(
            .moveBlock(blockID: item.id, newParent: list.id, newIndex: 0),
            to: &doc
        )
        XCTAssertEqual(doc.blocks[list.id]?.children, [item.id])
        XCTAssertEqual(doc.blocks[item.id]?.parentID, list.id)
        XCTAssertEqual(doc.rootChildren, [list.id])
    }

    func testMoveBlockRejectsCycle() throws {
        let parent = Block(type: .list)
        let child = Block(type: .listItem, parentID: parent.id)
        var doc = DocumentAST(
            blocks: [parent.id: parent, child.id: child],
            rootChildren: [parent.id]
        )
        doc.blocks[parent.id]?.children = [child.id]
        var engine = MutationEngine()
        // Try to move the parent into the child (would create a cycle).
        XCTAssertThrowsError(try engine.apply(
            .moveBlock(blockID: parent.id, newParent: child.id, newIndex: 0),
            to: &doc
        )) { error in
            guard case MutationError.wouldCreateCycle = error else {
                XCTFail("expected wouldCreateCycle, got \(error)")
                return
            }
        }
    }

    func testMoveBlockRejectsIndexOutOfRange() {
        let block = Block(type: .paragraph)
        let doc = DocumentAST(blocks: [block.id: block], rootChildren: [block.id])
        var working = doc
        var engine = MutationEngine()
        XCTAssertThrowsError(try engine.apply(
            .moveBlock(blockID: block.id, newParent: nil, newIndex: 99),
            to: &working
        )) { error in
            guard case MutationError.indexOutOfRange = error else {
                XCTFail("expected indexOutOfRange, got \(error)")
                return
            }
        }
    }

    // MARK: - setBlockAttribute

    func testSetBlockAttribute() throws {
        let block = Block(type: .heading, attributes: ["level": .number(1)])
        var doc = DocumentAST(blocks: [block.id: block], rootChildren: [block.id])
        var engine = MutationEngine()
        try engine.apply(
            .setBlockAttribute(blockID: block.id, key: "level", value: .number(3)),
            to: &doc
        )
        XCTAssertEqual(doc.blocks[block.id]?.attributes["level"]?.numberValue, 3)
    }

    // MARK: - setBlockContent

    func testSetBlockContent() throws {
        let block = Block(type: .paragraph, content: [InlineRun(text: "old")])
        var doc = DocumentAST(blocks: [block.id: block], rootChildren: [block.id])
        var engine = MutationEngine()
        try engine.apply(
            .setBlockContent(blockID: block.id, content: [InlineRun(text: "new")]),
            to: &doc
        )
        XCTAssertEqual(doc.blocks[block.id]?.content.first?.text, "new")
    }

    func testSetBlockContentRejectsDivider() {
        let block = Block(type: .divider)
        var doc = DocumentAST(blocks: [block.id: block], rootChildren: [block.id])
        var engine = MutationEngine()
        XCTAssertThrowsError(try engine.apply(
            .setBlockContent(
                blockID: block.id,
                content: [InlineRun(text: "x")]
            ),
            to: &doc
        )) { error in
            guard case MutationError.invalidOperation = error else {
                XCTFail("expected invalidOperation, got \(error)")
                return
            }
        }
    }

    // MARK: - inline run operations

    func testAppendInlineRun() throws {
        let block = Block(type: .paragraph, content: [InlineRun(text: "a")])
        var doc = DocumentAST(blocks: [block.id: block], rootChildren: [block.id])
        var engine = MutationEngine()
        try engine.apply(
            .appendInlineRun(blockID: block.id, run: InlineRun(text: "b")),
            to: &doc
        )
        XCTAssertEqual(doc.blocks[block.id]?.content.count, 2)
        XCTAssertEqual(doc.blocks[block.id]?.content.last?.text, "b")
    }

    func testReplaceInlineRun() throws {
        let block = Block(type: .paragraph, content: [InlineRun(text: "a")])
        var doc = DocumentAST(blocks: [block.id: block], rootChildren: [block.id])
        var engine = MutationEngine()
        try engine.apply(
            .replaceInlineRun(blockID: block.id, index: 0, run: InlineRun(text: "z")),
            to: &doc
        )
        XCTAssertEqual(doc.blocks[block.id]?.content.first?.text, "z")
    }

    func testReplaceInlineRunRejectsOutOfRange() {
        let block = Block(type: .paragraph, content: [InlineRun(text: "a")])
        var doc = DocumentAST(blocks: [block.id: block], rootChildren: [block.id])
        var engine = MutationEngine()
        XCTAssertThrowsError(try engine.apply(
            .replaceInlineRun(blockID: block.id, index: 5, run: InlineRun(text: "z")),
            to: &doc
        )) { error in
            guard case MutationError.inlineIndexOutOfRange = error else {
                XCTFail("expected inlineIndexOutOfRange, got \(error)")
                return
            }
        }
    }

    func testDeleteInlineRun() throws {
        let block = Block(type: .paragraph, content: [InlineRun(text: "a"), InlineRun(text: "b")])
        var doc = DocumentAST(blocks: [block.id: block], rootChildren: [block.id])
        var engine = MutationEngine()
        try engine.apply(
            .deleteInlineRun(blockID: block.id, index: 0),
            to: &doc
        )
        XCTAssertEqual(doc.blocks[block.id]?.content.count, 1)
        XCTAssertEqual(doc.blocks[block.id]?.content.first?.text, "b")
    }

    func testSetInlineAnnotationAdd() throws {
        let block = Block(type: .paragraph, content: [InlineRun(text: "x")])
        var doc = DocumentAST(blocks: [block.id: block], rootChildren: [block.id])
        var engine = MutationEngine()
        try engine.apply(
            .setInlineAnnotation(blockID: block.id, index: 0, annotation: .bold, enabled: true),
            to: &doc
        )
        XCTAssertEqual(doc.blocks[block.id]?.content[0].annotations, [.bold])
    }

    func testSetInlineAnnotationRemove() throws {
        let block = Block(
            type: .paragraph,
            content: [InlineRun(text: "x", annotations: [.bold])]
        )
        var doc = DocumentAST(blocks: [block.id: block], rootChildren: [block.id])
        var engine = MutationEngine()
        try engine.apply(
            .setInlineAnnotation(blockID: block.id, index: 0, annotation: .bold, enabled: false),
            to: &doc
        )
        XCTAssertTrue(doc.blocks[block.id]?.content[0].annotations.isEmpty ?? true)
    }

    func testSetInlineAnnotationIsIdempotent() throws {
        let block = Block(type: .paragraph, content: [InlineRun(text: "x", annotations: [.bold])])
        var doc = DocumentAST(blocks: [block.id: block], rootChildren: [block.id])
        var engine = MutationEngine()
        try engine.apply(
            .setInlineAnnotation(blockID: block.id, index: 0, annotation: .bold, enabled: true),
            to: &doc
        )
        XCTAssertEqual(doc.blocks[block.id]?.content[0].annotations, [.bold])
    }

    // MARK: - Validation

    func testValidateRejectsNonexistentBlock() {
        let doc = DocumentAST.empty
        let engine = MutationEngine()
        XCTAssertThrowsError(try engine.validate(
            .deleteBlock(blockID: UUID()),
            against: doc
        )) { error in
            guard case MutationError.blockNotFound = error else {
                XCTFail("expected blockNotFound, got \(error)")
                return
            }
        }
    }

    func testValidateAcceptsValidMutation() throws {
        let block = Block(type: .paragraph)
        let doc = DocumentAST(blocks: [block.id: block], rootChildren: [block.id])
        let engine = MutationEngine()
        XCTAssertNoThrow(try engine.validate(
            .deleteBlock(blockID: block.id),
            against: doc
        ))
    }

    func testValidateAnchorNotInParent() {
        let parent = Block(type: .list)
        let foreign = Block(type: .listItem)  // NOT a child of parent
        let doc = DocumentAST(
            blocks: [parent.id: parent, foreign.id: foreign],
            rootChildren: [parent.id, foreign.id]
        )
        let engine = MutationEngine()
        let newBlock = Block(type: .listItem, parentID: parent.id)
        XCTAssertThrowsError(try engine.validate(
            .insertBlockAfter(parentID: parent.id, anchorID: foreign.id, block: newBlock),
            against: doc
        )) { error in
            guard case MutationError.anchorNotFound = error else {
                XCTFail("expected anchorNotFound, got \(error)")
                return
            }
        }
    }

    // MARK: - Composition

    func testComposedMutationsProduceFinalState() throws {
        var doc = DocumentAST.empty
        var engine = MutationEngine()

        // 1. Insert a heading.
        let heading = Block(type: .heading, attributes: ["level": .number(1)])
        try engine.apply(
            .insertBlockAfter(parentID: nil, anchorID: nil, block: heading),
            to: &doc
        )
        // 2. Append a paragraph after the heading.
        let para = Block(
            type: .paragraph,
            content: [InlineRun(text: "first paragraph")]
        )
        try engine.apply(
            .insertBlockAfter(parentID: nil, anchorID: heading.id, block: para),
            to: &doc
        )
        // 3. Append inline text to the paragraph.
        try engine.apply(
            .appendInlineRun(blockID: para.id, run: InlineRun(text: " — more text")),
            to: &doc
        )
        // 4. Add bold annotation to the first run.
        try engine.apply(
            .setInlineAnnotation(blockID: para.id, index: 0, annotation: .bold, enabled: true),
            to: &doc
        )
        // Final state.
        XCTAssertEqual(doc.rootChildren, [heading.id, para.id])
        XCTAssertEqual(doc.blocks[heading.id]?.attributes["level"]?.numberValue, 1)
        XCTAssertEqual(doc.blocks[para.id]?.content.count, 2)
        XCTAssertTrue(doc.blocks[para.id]?.content[0].annotations.contains(.bold) ?? false)
        XCTAssertEqual(doc.blocks[para.id]?.content[1].text, " — more text")
    }

    // MARK: - Empty document edge case

    func testEmptyDocument() throws {
        let doc = DocumentAST.empty
        XCTAssertTrue(doc.blocks.isEmpty)
        XCTAssertTrue(doc.rootChildren.isEmpty)
        let engine = MutationEngine()
        // Set-document mutations are no-ops against the AST.
        XCTAssertNoThrow(try engine.validate(.setDocumentTitle(title: "x"), against: doc))
    }

    // MARK: - Cycle detection: deeper cases

    func testDeepCycleRejected() throws {
        // Build a 3-level chain: a -> b -> c
        let a = Block(type: .list)
        let b = Block(type: .list, parentID: a.id)
        let c = Block(type: .list, parentID: b.id)
        var doc = DocumentAST(
            blocks: [a.id: a, b.id: b, c.id: c],
            rootChildren: [a.id]
        )
        doc.blocks[a.id]?.children = [b.id]
        doc.blocks[b.id]?.children = [c.id]
        doc.blocks[c.id]?.children = []
        let engine = MutationEngine()
        // Moving `a` into `c` is a cycle (a -> b -> c -> a).
        XCTAssertThrowsError(try engine.validate(
            .moveBlock(blockID: a.id, newParent: c.id, newIndex: 0),
            against: doc
        ))
    }

    // MARK: - Set document title / meta

    func testSetDocumentTitleIsNoOpAgainstAST() throws {
        let doc = DocumentAST.empty
        var working = doc
        var engine = MutationEngine()
        try engine.apply(
            .setDocumentTitle(title: "My Doc"),
            to: &working
        )
        // AST is unchanged.
        XCTAssertEqual(working, doc)
    }
}
