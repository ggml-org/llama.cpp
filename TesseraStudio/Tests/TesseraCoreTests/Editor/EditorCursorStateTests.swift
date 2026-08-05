import XCTest
import Foundation
@testable import TesseraCore

/// Tests for `EditorCursorState` (the two-cursor model) and
/// the cursor-in-block resolution helpers. The two-cursor
/// model is what makes the user and the agent coexist in the
/// same document without contention; the data model carries
/// both as named fields.
final class EditorCursorStateTests: XCTestCase {

    // MARK: - Two-cursor independence

    func testUserAndAgentCursorsCanCoexist() {
        let blockA = UUID()
        let blockB = UUID()
        let state = EditorCursorState(
            userCursor: TextCursor(blockID: blockA, offset: 0),
            agentCursor: TextCursor(blockID: blockB, offset: 5)
        )
        XCTAssertNotNil(state.userCursor)
        XCTAssertNotNil(state.agentCursor)
        XCTAssertNotEqual(state.userCursor?.blockID, state.agentCursor?.blockID)
    }

    func testUserCursorCanMoveIndependently() {
        let blockA = UUID()
        let blockB = UUID()
        var state = EditorCursorState(
            userCursor: TextCursor(blockID: blockA, offset: 0),
            agentCursor: TextCursor(blockID: blockA, offset: 10)
        )
        // Move the user cursor. The agent cursor stays put.
        state.userCursor = TextCursor(blockID: blockB, offset: 3)
        XCTAssertEqual(state.userCursor?.blockID, blockB)
        XCTAssertEqual(state.agentCursor?.blockID, blockA)
        XCTAssertEqual(state.agentCursor?.offset, 10)
    }

    func testAgentCursorDoesNotMoveWhenUserCursorMoves() {
        let block = UUID()
        var state = EditorCursorState(
            userCursor: TextCursor(blockID: block, offset: 0),
            agentCursor: TextCursor(blockID: block, offset: 5)
        )
        let agentBefore = state.agentCursor
        state.userCursor = TextCursor(blockID: block, offset: 20)
        XCTAssertEqual(state.agentCursor, agentBefore)
    }

    func testBothCursorsCanBeInTheSameParagraph() {
        let block = UUID()
        let state = EditorCursorState(
            userCursor: TextCursor(blockID: block, offset: 0),
            agentCursor: TextCursor(blockID: block, offset: 100)
        )
        XCTAssertEqual(state.userCursor?.blockID, state.agentCursor?.blockID)
        XCTAssertNotEqual(state.userCursor?.offset, state.agentCursor?.offset)
    }

    // MARK: - Agent cursor active flag

    func testAgentCursorActiveDefaultsToFalse() {
        let state = EditorCursorState(
            agentCursor: TextCursor(blockID: UUID(), offset: 0)
        )
        XCTAssertFalse(state.agentCursorActive)
    }

    func testAgentCursorActiveTrueWhenToggled() {
        var state = EditorCursorState(
            agentCursor: TextCursor(blockID: UUID(), offset: 0)
        )
        state.agentCursorActive = true
        XCTAssertTrue(state.hasAgentActive)
    }

    // MARK: - Cursor selection

    func testCursorSelectionTracksRange() {
        let block = UUID()
        let selection = CursorSelection(blockID: block, anchorOffset: 5, headOffset: 10)
        XCTAssertEqual(selection.length, 5)
        XCTAssertEqual(selection.lowerOffset, 5)
        XCTAssertEqual(selection.upperOffset, 10)
        XCTAssertFalse(selection.isEmpty)
    }

    func testEmptySelection() {
        let block = UUID()
        let selection = CursorSelection(blockID: block, anchorOffset: 5, headOffset: 5)
        XCTAssertTrue(selection.isEmpty)
        XCTAssertEqual(selection.length, 0)
    }

    // MARK: - CursorInBlock resolution

    func testTextCursorResolvesToRunIndex() {
        let block = Block(
            id: UUID(),
            type: .paragraph,
            content: [
                InlineRun(text: "hello"),       // length 5
                InlineRun(text: " "),           // length 1
                InlineRun(text: "world"),       // length 5
            ]
        )
        // Offset 0 is in run 0 at run-offset 0.
        let c0 = TextCursor(blockID: block.id, offset: 0)
        XCTAssertEqual(c0.resolved(in: block)?.runIndex, 0)
        XCTAssertEqual(c0.resolved(in: block)?.runOffset, 0)
        // Offset 5 is at the boundary of run 0 / run 1 (run 0's end).
        let c5 = TextCursor(blockID: block.id, offset: 5)
        XCTAssertEqual(c5.resolved(in: block)?.runIndex, 0)
        XCTAssertEqual(c5.resolved(in: block)?.runOffset, 5)
        // Offset 6 is in run 1 at run-offset 1.
        let c6 = TextCursor(blockID: block.id, offset: 6)
        XCTAssertEqual(c6.resolved(in: block)?.runIndex, 1)
        XCTAssertEqual(c6.resolved(in: block)?.runOffset, 1)
        // Offset 11 (= 6 + 5) is in run 2 at run-offset 5.
        let c11 = TextCursor(blockID: block.id, offset: 11)
        XCTAssertEqual(c11.resolved(in: block)?.runIndex, 2)
        XCTAssertEqual(c11.resolved(in: block)?.runOffset, 5)
    }

    func testTextCursorResolvesDividerBlockReturnsNil() {
        // Dividers don't have inline content.
        let block = Block(id: UUID(), type: .divider, content: [])
        let cursor = TextCursor(blockID: block.id, offset: 0)
        XCTAssertNil(cursor.resolved(in: block))
    }

    func testTextCursorResolvesImageBlockReturnsNil() {
        let block = Block(
            id: UUID(),
            type: .image,
            attributes: ["source": .string("https://example.com/x.png")],
            content: []
        )
        let cursor = TextCursor(blockID: block.id, offset: 0)
        XCTAssertNil(cursor.resolved(in: block))
    }

    func testCursorInBlockRoundTrip() {
        let block = Block(
            id: UUID(),
            type: .paragraph,
            content: [
                InlineRun(text: "hi"),
                InlineRun(text: " "),
                InlineRun(text: "there"),
            ]
        )
        let cursorInBlock = CursorInBlock(blockID: block.id, runIndex: 2, runOffset: 3)
        let textCursor = TextCursor(cursorInBlock, in: block)
        XCTAssertEqual(textCursor.blockID, block.id)
        XCTAssertEqual(textCursor.offset, 2 + 1 + 3)  // 2 (first run) + 1 (space) + 3 (offset into "there")
        let resolved = textCursor.resolved(in: block)
        XCTAssertEqual(resolved?.runIndex, 2)
        XCTAssertEqual(resolved?.runOffset, 3)
    }
}
