import XCTest
import Foundation
@testable import TesseraCore

/// Tests for `TextEditReducer` (the seam between
/// NSAttributedString edits and the Phase 1 Mutation API).
/// Covers: typing produces a setBlockContent mutation,
/// formatting produces a setInlineAnnotation mutation, paste
/// produces a setBlockContent mutation.
final class TextEditReducerTests: XCTestCase {

    // MARK: - diff() helper

    func testDiffEmpty() {
        let d = TextEditReducer.diff(before: "abc", after: "abc")
        XCTAssertTrue(d.isEmpty)
        XCTAssertEqual(d.commonPrefix, 3)
        XCTAssertEqual(d.commonSuffix, 0)
    }

    func testDiffInsertion() {
        let d = TextEditReducer.diff(before: "hello", after: "hello world")
        XCTAssertTrue(d.isInsertion)
        XCTAssertEqual(d.deletedRange.length, 0)
        XCTAssertEqual(d.insertedRange.length, 6)
    }

    func testDiffDeletion() {
        let d = TextEditReducer.diff(before: "hello world", after: "hello")
        XCTAssertTrue(d.isDeletion)
        XCTAssertEqual(d.deletedRange.length, 6)
        XCTAssertEqual(d.insertedRange.length, 0)
    }

    func testDiffReplacement() {
        let d = TextEditReducer.diff(before: "hello world", after: "hello there")
        XCTAssertTrue(d.isReplacement)
        XCTAssertEqual(d.deletedRange.length, 5)
        XCTAssertEqual(d.insertedRange.length, 5)
    }

    func testDiffCommonPrefixAndSuffix() {
        let d = TextEditReducer.diff(before: "abcXYZdef", after: "abc123def")
        XCTAssertEqual(d.commonPrefix, 3)
        XCTAssertEqual(d.commonSuffix, 3)
        XCTAssertTrue(d.isReplacement)
    }

    // MARK: - reduce()

    func testReduceInsertionReturnsSetBlockContent() {
        let r = TextEditReducer()
        let blockID = UUID()
        let before = [InlineRun(text: "hello")]
        let after = [InlineRun(text: "hello world")]
        let mutations = r.reduce(blockID: blockID, before: before, after: after)
        XCTAssertEqual(mutations.count, 1)
        if case .setBlockContent(let id, let content) = mutations[0] {
            XCTAssertEqual(id, blockID)
            XCTAssertEqual(content.first?.text, "hello world")
        } else {
            XCTFail("expected setBlockContent, got \(mutations[0])")
        }
    }

    func testReduceDeletionReturnsSetBlockContent() {
        let r = TextEditReducer()
        let blockID = UUID()
        let before = [InlineRun(text: "hello world")]
        let after = [InlineRun(text: "hello")]
        let mutations = r.reduce(blockID: blockID, before: before, after: after)
        XCTAssertEqual(mutations.count, 1)
        if case .setBlockContent(let id, _) = mutations[0] {
            XCTAssertEqual(id, blockID)
        } else {
            XCTFail("expected setBlockContent")
        }
    }

    func testReduceReplacementReturnsSetBlockContent() {
        let r = TextEditReducer()
        let blockID = UUID()
        let before = [InlineRun(text: "hello world")]
        let after = [InlineRun(text: "hello there")]
        let mutations = r.reduce(blockID: blockID, before: before, after: after)
        XCTAssertEqual(mutations.count, 1)
        if case .setBlockContent = mutations[0] {
            // OK
        } else {
            XCTFail("expected setBlockContent")
        }
    }

    func testReduceNoOpReturnsEmpty() {
        let r = TextEditReducer()
        let blockID = UUID()
        let before = [InlineRun(text: "hello")]
        let after = [InlineRun(text: "hello")]
        XCTAssertTrue(r.reduce(blockID: blockID, before: before, after: after).isEmpty)
    }

    // MARK: - reduceFormattingChange()

    func testReduceFormattingChangeAddsAnnotation() {
        let r = TextEditReducer()
        let blockID = UUID()
        let content = [InlineRun(text: "bold this", annotations: [])]
        let mutations = r.reduceFormattingChange(
            blockID: blockID,
            content: content,
            offset: 0,
            annotation: .bold
        )
        XCTAssertEqual(mutations.count, 1)
        if case .setInlineAnnotation(let id, let index, let annotation, let enabled) = mutations[0] {
            XCTAssertEqual(id, blockID)
            XCTAssertEqual(index, 0)
            XCTAssertEqual(annotation, .bold)
            XCTAssertTrue(enabled)
        } else {
            XCTFail("expected setInlineAnnotation")
        }
    }

    func testReduceFormattingChangeTogglesAnnotation() {
        let r = TextEditReducer()
        let blockID = UUID()
        let content = [InlineRun(text: "bold", annotations: [.bold])]
        let mutations = r.reduceFormattingChange(
            blockID: blockID,
            content: content,
            offset: 0,
            annotation: .bold
        )
        // Already has bold -> the toggle should turn it off.
        if case .setInlineAnnotation(_, _, _, let enabled) = mutations[0] {
            XCTAssertFalse(enabled)
        } else {
            XCTFail("expected setInlineAnnotation")
        }
    }

    func testReduceFormattingChangeAtEndOfBlock() {
        // The cursor is at the end of the block (offset 4,
        // at the end of "bold"). The formatting applies to
        // the last run.
        let r = TextEditReducer()
        let blockID = UUID()
        let content = [InlineRun(text: "bold", annotations: [])]
        let mutations = r.reduceFormattingChange(
            blockID: blockID,
            content: content,
            offset: 4,
            annotation: .italic
        )
        XCTAssertEqual(mutations.count, 1)
        if case .setInlineAnnotation(_, let index, let annotation, let enabled) = mutations[0] {
            XCTAssertEqual(index, 0)
            XCTAssertEqual(annotation, .italic)
            XCTAssertTrue(enabled)
        } else {
            XCTFail("expected setInlineAnnotation")
        }
    }

    // MARK: - reducePaste()

    func testReducePasteReturnsSetBlockContent() {
        let r = TextEditReducer()
        let blockID = UUID()
        let mutations = r.reducePaste(
            blockID: blockID,
            pastedText: "pasted text",
            existingAnnotations: []
        )
        XCTAssertEqual(mutations.count, 1)
        if case .setBlockContent(let id, let content) = mutations[0] {
            XCTAssertEqual(id, blockID)
            XCTAssertEqual(content.count, 1)
            XCTAssertEqual(content[0].text, "pasted text")
        } else {
            XCTFail("expected setBlockContent")
        }
    }

    // MARK: - NSRange.substring helper

    func testNSRangeSubstring() {
        let s = "hello world"
        XCTAssertEqual(NSRange(location: 0, length: 5).substring(in: s), "hello")
        XCTAssertEqual(NSRange(location: 6, length: 5).substring(in: s), "world")
        XCTAssertEqual(NSRange(location: 0, length: 0).substring(in: s), "")
        // Out-of-range returns empty.
        XCTAssertEqual(NSRange(location: 20, length: 5).substring(in: s), "")
    }
}
