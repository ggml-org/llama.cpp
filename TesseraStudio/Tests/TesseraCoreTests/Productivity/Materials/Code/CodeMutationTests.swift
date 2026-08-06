import XCTest
@testable import TesseraCore

/// Tests for the ``CodeMutation`` enum + the
/// ``CodeMutationEngine``. The tests cover the
/// engine's happy path, the validation errors
/// (ambiguous match, position out of range, ...), and
/// the inverse computation for undo.
final class CodeMutationTests: XCTestCase {

    private let engine = CodeMutationEngine()

    private func makeFile(body: String = "let x = 1\nlet y = 2\n") -> CodeFile {
        return CodeFile(
            path: "/tmp/test.swift",
            body: body
        )
    }

    // MARK: - replaceCodeBlock

    func testReplaceCodeBlockReplacesBody() throws {
        let file = makeFile()
        let result = try engine.apply(
            .replaceCodeBlock(fileID: file.id, newBody: "let z = 3\n"),
            to: file
        )
        XCTAssertEqual(result.updated.body, "let z = 3\n")
        XCTAssertEqual(result.preBody, file.body)
        XCTAssertEqual(result.updated.size, Int64("let z = 3\n".utf8.count))
        XCTAssertNotEqual(result.updated.checksum, file.checksum)
    }

    func testReplaceCodeBlockEmptyBody() throws {
        let file = makeFile()
        let result = try engine.apply(
            .replaceCodeBlock(fileID: file.id, newBody: ""),
            to: file
        )
        XCTAssertEqual(result.updated.body, "")
        XCTAssertEqual(result.updated.size, 0)
    }

    // MARK: - replaceCodeRange

    func testReplaceCodeRangeReplacesMatch() throws {
        let file = makeFile(body: "let x = 1\nlet y = 2\n")
        let result = try engine.apply(
            .replaceCodeRange(fileID: file.id, match: "1", replacement: "42"),
            to: file
        )
        XCTAssertEqual(result.updated.body, "let x = 42\nlet y = 2\n")
    }

    func testReplaceCodeRangeRejectsAmbiguousMatch() {
        let file = makeFile(body: "a a a")
        XCTAssertThrowsError(
            try engine.apply(
                .replaceCodeRange(fileID: file.id, match: "a", replacement: "b"),
                to: file
            )
        ) { error in
            guard case CodeMutationError.matchAmbiguous(let match, let count, _) = error else {
                XCTFail("expected matchAmbiguous, got \(error)")
                return
            }
            XCTAssertEqual(match, "a")
            XCTAssertEqual(count, 3)
        }
    }

    func testReplaceCodeRangeRejectsMissingMatch() {
        let file = makeFile(body: "abc")
        XCTAssertThrowsError(
            try engine.apply(
                .replaceCodeRange(fileID: file.id, match: "xyz", replacement: "1"),
                to: file
            )
        ) { error in
            guard case CodeMutationError.matchNotFound = error else {
                XCTFail("expected matchNotFound, got \(error)")
                return
            }
        }
    }

    func testReplaceCodeRangeRejectsEmptyMatch() {
        let file = makeFile(body: "abc")
        XCTAssertThrowsError(
            try engine.apply(
                .replaceCodeRange(fileID: file.id, match: "", replacement: "1"),
                to: file
            )
        )
    }

    // MARK: - insertCodeAt

    func testInsertCodeAtInsertsAtPosition() throws {
        let file = makeFile(body: "let x = 1;let y = 2")
        let result = try engine.apply(
            .insertCodeAt(fileID: file.id, position: 10, text: "\n"),
            to: file
        )
        XCTAssertEqual(result.updated.body, "let x = 1;\nlet y = 2")
    }

    func testInsertCodeAtBeginning() throws {
        let file = makeFile(body: "world")
        let result = try engine.apply(
            .insertCodeAt(fileID: file.id, position: 0, text: "hello "),
            to: file
        )
        XCTAssertEqual(result.updated.body, "hello world")
    }

    func testInsertCodeAtEnd() throws {
        let file = makeFile(body: "hello")
        let result = try engine.apply(
            .insertCodeAt(fileID: file.id, position: 5, text: " world"),
            to: file
        )
        XCTAssertEqual(result.updated.body, "hello world")
    }

    func testInsertCodeAtRejectsNegativePosition() {
        let file = makeFile(body: "abc")
        XCTAssertThrowsError(
            try engine.apply(
                .insertCodeAt(fileID: file.id, position: -1, text: "x"),
                to: file
            )
        ) { error in
            guard case CodeMutationError.positionOutOfRange = error else {
                XCTFail("expected positionOutOfRange, got \(error)")
                return
            }
        }
    }

    func testInsertCodeAtRejectsOutOfRange() {
        let file = makeFile(body: "abc")
        XCTAssertThrowsError(
            try engine.apply(
                .insertCodeAt(fileID: file.id, position: 100, text: "x"),
                to: file
            )
        ) { error in
            guard case CodeMutationError.positionOutOfRange(let pos, let len, _) = error else {
                XCTFail("expected positionOutOfRange, got \(error)")
                return
            }
            XCTAssertEqual(pos, 100)
            XCTAssertEqual(len, 3)
        }
    }

    // MARK: - addTag / removeTag

    func testAddTagAppends() throws {
        let file = makeFile()
        let result = try engine.apply(
            .addTag(fileID: file.id, tag: "core"),
            to: file
        )
        XCTAssertEqual(result.updated.tags, ["core"])
    }

    func testAddTagRejectsDuplicate() {
        var file = makeFile()
        file.tags = ["core"]
        XCTAssertThrowsError(
            try engine.apply(
                .addTag(fileID: file.id, tag: "core"),
                to: file
            )
        ) { error in
            guard case CodeMutationError.tagAlreadyPresent = error else {
                XCTFail("expected tagAlreadyPresent, got \(error)")
                return
            }
        }
    }

    func testRemoveTagRemoves() throws {
        var file = makeFile()
        file.tags = ["core", "stable"]
        let result = try engine.apply(
            .removeTag(fileID: file.id, tag: "core"),
            to: file
        )
        XCTAssertEqual(result.updated.tags, ["stable"])
    }

    func testRemoveTagRejectsMissing() {
        let file = makeFile()
        XCTAssertThrowsError(
            try engine.apply(
                .removeTag(fileID: file.id, tag: "nope"),
                to: file
            )
        ) { error in
            guard case CodeMutationError.tagNotPresent = error else {
                XCTFail("expected tagNotPresent, got \(error)")
                return
            }
        }
    }

    // MARK: - linkTo / unlinkFrom

    func testLinkToAppends() throws {
        let file = makeFile()
        let otherID = UUID()
        let result = try engine.apply(
            .linkTo(fileID: file.id, otherEntityID: otherID, linkType: "implements"),
            to: file
        )
        XCTAssertEqual(result.updated.linkedEntityIDs, [otherID])
    }

    func testLinkToRejectsDuplicate() throws {
        let file = makeFile()
        let otherID = UUID()
        // First link succeeds.
        let first = try engine.apply(
            .linkTo(fileID: file.id, otherEntityID: otherID, linkType: "implements"),
            to: file
        )
        // Second link is a no-op (the engine doesn't
        // append a duplicate). The result is still
        // valid; the linkedEntityIDs is unchanged.
        let second = try engine.apply(
            .linkTo(fileID: first.updated.id, otherEntityID: otherID, linkType: "implements"),
            to: first.updated
        )
        XCTAssertEqual(second.updated.linkedEntityIDs, [otherID])
    }

    func testUnlinkFromRemoves() throws {
        let a = UUID()
        let b = UUID()
        var file = makeFile()
        file.linkedEntityIDs = [a, b]
        let result = try engine.apply(
            .unlinkFrom(fileID: file.id, otherEntityID: a, linkType: "related"),
            to: file
        )
        XCTAssertEqual(result.updated.linkedEntityIDs, [b])
    }

    // MARK: - Inverse

    func testInverseReplaceCodeBlock() {
        let file = makeFile()
        let mutation = CodeMutation.replaceCodeBlock(fileID: file.id, newBody: "new")
        let inverses = mutation.inverse(preBody: "old", preTags: [], preLinks: [])
        XCTAssertEqual(inverses.count, 1)
        if case .replaceCodeBlock(_, let body) = inverses[0] {
            XCTAssertEqual(body, "old")
        } else {
            XCTFail("expected replaceCodeBlock")
        }
    }

    func testInverseAddTag() {
        let file = makeFile()
        let mutation = CodeMutation.addTag(fileID: file.id, tag: "core")
        let inverses = mutation.inverse(preBody: "", preTags: [], preLinks: [])
        if case .removeTag(_, let tag) = inverses[0] {
            XCTAssertEqual(tag, "core")
        } else {
            XCTFail("expected removeTag")
        }
    }

    func testInverseRemoveTag() {
        let file = makeFile()
        let mutation = CodeMutation.removeTag(fileID: file.id, tag: "core")
        let inverses = mutation.inverse(preBody: "", preTags: [], preLinks: [])
        if case .addTag(_, let tag) = inverses[0] {
            XCTAssertEqual(tag, "core")
        } else {
            XCTFail("expected addTag")
        }
    }

    func testInverseLinkTo() {
        let file = makeFile()
        let other = UUID()
        let mutation = CodeMutation.linkTo(
            fileID: file.id, otherEntityID: other, linkType: "implements"
        )
        let inverses = mutation.inverse(preBody: "", preTags: [], preLinks: [])
        if case .unlinkFrom(_, let o, let t) = inverses[0] {
            XCTAssertEqual(o, other)
            XCTAssertEqual(t, "implements")
        } else {
            XCTFail("expected unlinkFrom")
        }
    }

    // MARK: - Properties

    func testMutationFileIDAccessor() {
        let id = UUID()
        XCTAssertEqual(CodeMutation.replaceCodeBlock(fileID: id, newBody: "").fileID, id)
        XCTAssertEqual(CodeMutation.replaceCodeRange(fileID: id, match: "a", replacement: "b").fileID, id)
        XCTAssertEqual(CodeMutation.insertCodeAt(fileID: id, position: 0, text: "x").fileID, id)
        XCTAssertEqual(CodeMutation.addTag(fileID: id, tag: "t").fileID, id)
        XCTAssertEqual(CodeMutation.removeTag(fileID: id, tag: "t").fileID, id)
        XCTAssertEqual(CodeMutation.linkTo(fileID: id, otherEntityID: id, linkType: "x").fileID, id)
        XCTAssertEqual(CodeMutation.unlinkFrom(fileID: id, otherEntityID: id, linkType: "x").fileID, id)
    }

    func testMutationReceiptType() {
        let id = UUID()
        XCTAssertEqual(CodeMutation.replaceCodeBlock(fileID: id, newBody: "").receiptType, "code_file_body_replaced")
        XCTAssertEqual(CodeMutation.addTag(fileID: id, tag: "t").receiptType, "code_file_tagged")
        XCTAssertEqual(CodeMutation.removeTag(fileID: id, tag: "t").receiptType, "code_file_untagged")
        XCTAssertEqual(CodeMutation.linkTo(fileID: id, otherEntityID: id, linkType: "x").receiptType, "code_file_linked")
        XCTAssertEqual(CodeMutation.unlinkFrom(fileID: id, otherEntityID: id, linkType: "x").receiptType, "code_file_unlinked")
    }

    func testShortDescription() {
        let id = UUID()
        let desc = CodeMutation.replaceCodeRange(
            fileID: id, match: "foo", replacement: "bar"
        ).shortDescription
        XCTAssertTrue(desc.contains("foo"))
        XCTAssertTrue(desc.contains("bar"))
    }
}
