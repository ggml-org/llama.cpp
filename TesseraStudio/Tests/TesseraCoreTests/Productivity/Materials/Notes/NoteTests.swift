import XCTest
@testable import TesseraCore

/// Unit tests for the ``Note`` model: JSON round-trip, tag
/// normalization, snippet computation, plain-text / heading
/// extraction, word count + reading time, display title.
final class NoteTests: XCTestCase {

    // MARK: - JSON round-trip

    func testNoteRoundTripsJSON() throws {
        let date = Date(timeIntervalSince1970: 1_000_000)
        let original = Note(
            id: UUID(),
            title: "Q3 Review",
            body: DocumentAST(
                blocks: [
                    UUID(): Block(
                        type: .heading,
                        attributes: ["level": .number(1)],
                        content: [InlineRun(text: "Q3 Review")]
                    )
                ],
                rootChildren: [UUID()]
            ),
            tags: ["q3", "review", "2026"],
            pinnedAt: date,
            archivedAt: nil,
            linkedEntityIDs: [UUID()],
            createdAt: date,
            updatedAt: date
        )
        let data = try original.jsonData()
        let decoded = try Note.from(jsonData: data)
        XCTAssertEqual(decoded, original)
    }

    func testEmptyNoteRoundTrips() throws {
        let date = Date(timeIntervalSince1970: 1_000_000)
        let original = Note(
            id: UUID(),
            title: "",
            body: .empty,
            tags: [],
            createdAt: date,
            updatedAt: date
        )
        let data = try original.jsonData()
        let decoded = try Note.from(jsonData: data)
        XCTAssertEqual(decoded, original)
    }

    func testNoteJSONStringRoundTrip() throws {
        let date = Date(timeIntervalSince1970: 1_000_000)
        let original = Note(
            id: UUID(),
            title: "Untitled",
            body: .empty,
            tags: ["x"],
            createdAt: date,
            updatedAt: date
        )
        let body = try original.jsonDataString()
        let decoded = try Note.from(jsonDataString: body)
        XCTAssertEqual(decoded, original)
    }

    // MARK: - Tag normalization

    func testNormalizeTagsLowercasesAndTrims() {
        let input = ["  Q3  ", "Review", "REVIEW", "  q3", "2026"]
        let normalized = Note.normalizeTags(input)
        XCTAssertEqual(normalized, ["q3", "review", "2026"])
    }

    func testNormalizeTagsDropsEmpties() {
        let input = ["", "  ", "valid"]
        let normalized = Note.normalizeTags(input)
        XCTAssertEqual(normalized, ["valid"])
    }

    func testNormalizeTagsPreservesOrder() {
        let input = ["b", "a", "c", "a"]
        let normalized = Note.normalizeTags(input)
        XCTAssertEqual(normalized, ["b", "a", "c"])
    }

    func testNormalizeTagsHandlesDuplicatesCaseInsensitively() {
        let input = ["Foo", "FOO", "foo"]
        let normalized = Note.normalizeTags(input)
        XCTAssertEqual(normalized, ["foo"])
    }

    func testInitNormalizesTags() {
        let note = Note(tags: ["  Q3  ", "review", "Q3"])
        XCTAssertEqual(note.tags, ["q3", "review"])
    }

    // MARK: - Display title

    func testDisplayTitleFallsBackToUserTitle() {
        let note = Note(title: "My Note", body: .empty)
        XCTAssertEqual(note.displayTitle, "My Note")
    }

    func testDisplayTitleFallsBackToUntitled() {
        let note = Note(title: "", body: .empty)
        XCTAssertEqual(note.displayTitle, "Untitled")
    }

    func testDisplayTitleFallsBackToFirstHeading() {
        let headingID = UUID()
        let note = Note(
            title: "",
            body: DocumentAST(
                blocks: [
                    headingID: Block(
                        type: .heading,
                        attributes: ["level": .number(1)],
                        content: [InlineRun(text: "Auto-derived title")]
                    )
                ],
                rootChildren: [headingID]
            )
        )
        XCTAssertEqual(note.displayTitle, "Auto-derived title")
    }

    func testDisplayTitlePrefersUserTitle() {
        let headingID = UUID()
        let note = Note(
            title: "User Override",
            body: DocumentAST(
                blocks: [
                    headingID: Block(
                        type: .heading,
                        attributes: ["level": .number(1)],
                        content: [InlineRun(text: "Body heading")]
                    )
                ],
                rootChildren: [headingID]
            )
        )
        XCTAssertEqual(note.displayTitle, "User Override")
    }

    // MARK: - Plain text extraction

    func testPlainTextOfEmptyASTIsEmpty() {
        XCTAssertEqual(Note.plainText(of: .empty), "")
    }

    func testPlainTextJoinsParagraphs() {
        let id1 = UUID()
        let id2 = UUID()
        let ast = DocumentAST(
            blocks: [
                id1: Block(
                    type: .paragraph,
                    content: [InlineRun(text: "First paragraph.")]
                ),
                id2: Block(
                    type: .paragraph,
                    content: [InlineRun(text: "Second paragraph.")]
                ),
            ],
            rootChildren: [id1, id2]
        )
        let plain = Note.plainText(of: ast)
        XCTAssertTrue(plain.contains("First paragraph"))
        XCTAssertTrue(plain.contains("Second paragraph"))
    }

    func testPlainTextFlattensListItems() {
        let listID = UUID()
        let item1ID = UUID()
        let item2ID = UUID()
        let ast = DocumentAST(
            blocks: [
                listID: Block(type: .list, children: [item1ID, item2ID]),
                item1ID: Block(
                    type: .listItem,
                    content: [InlineRun(text: "Item one")]
                ),
                item2ID: Block(
                    type: .listItem,
                    content: [InlineRun(text: "Item two")]
                ),
            ],
            rootChildren: [listID]
        )
        let plain = Note.plainText(of: ast)
        XCTAssertTrue(plain.contains("Item one"))
        XCTAssertTrue(plain.contains("Item two"))
    }

    // MARK: - First heading extraction

    func testFirstHeadingReturnsNilForEmpty() {
        XCTAssertNil(Note.firstHeadingText(in: .empty))
    }

    func testFirstHeadingReturnsNilWhenNoHeadings() {
        let id = UUID()
        let ast = DocumentAST(
            blocks: [id: Block(type: .paragraph, content: [InlineRun(text: "Just a paragraph.")])],
            rootChildren: [id]
        )
        XCTAssertNil(Note.firstHeadingText(in: ast))
    }

    func testFirstHeadingFindsTheFirstHeading() {
        let h1 = UUID()
        let h2 = UUID()
        let ast = DocumentAST(
            blocks: [
                h1: Block(
                    type: .heading,
                    attributes: ["level": .number(2)],
                    content: [InlineRun(text: "Second heading")]
                ),
                h2: Block(
                    type: .heading,
                    attributes: ["level": .number(1)],
                    content: [InlineRun(text: "First heading")]
                ),
            ],
            rootChildren: [h1, h2]
        )
        // The "first" is in document order — `h1` comes first
        // in `rootChildren`, so the heading text is "Second
        // heading". The function is order-sensitive.
        XCTAssertEqual(Note.firstHeadingText(in: ast), "Second heading")
    }

    // MARK: - Snippet

    func testSnippetRespectsMaxLength() {
        let id = UUID()
        let longText = String(repeating: "a", count: 500)
        let ast = DocumentAST(
            blocks: [id: Block(type: .paragraph, content: [InlineRun(text: longText)])],
            rootChildren: [id]
        )
        let snippet = Note.plainTextSnippet(from: ast, maxLength: 100)
        XCTAssertTrue(snippet.count <= 101)  // 100 chars + ellipsis
        XCTAssertTrue(snippet.hasSuffix("…"))
    }

    func testSnippetCollapsesWhitespace() {
        let id = UUID()
        let ast = DocumentAST(
            blocks: [id: Block(
                type: .paragraph,
                content: [InlineRun(text: "hello\n\n   world\t\t  !")]
            )],
            rootChildren: [id]
        )
        let snippet = Note.plainTextSnippet(from: ast, maxLength: 200)
        XCTAssertEqual(snippet, "hello world !")
    }

    func testSnippetEmptyAST() {
        XCTAssertEqual(Note.plainTextSnippet(from: .empty, maxLength: 200), "")
    }

    func testSnippetOnNote() {
        let id = UUID()
        let note = Note(
            title: "X",
            body: DocumentAST(
                blocks: [id: Block(type: .paragraph, content: [InlineRun(text: "Hello, world.")])],
                rootChildren: [id]
            )
        )
        XCTAssertEqual(note.snippet(maxLength: 200), "Hello, world.")
    }

    // MARK: - Word count + reading time

    func testWordCountEmpty() {
        XCTAssertEqual(Note.wordCount(of: .empty), 0)
    }

    func testWordCountSimple() {
        let id = UUID()
        let ast = DocumentAST(
            blocks: [id: Block(
                type: .paragraph,
                content: [InlineRun(text: "the quick brown fox")]
            )],
            rootChildren: [id]
        )
        XCTAssertEqual(Note.wordCount(of: ast), 4)
    }

    func testWordCountHandlesNewlines() {
        let id = UUID()
        let ast = DocumentAST(
            blocks: [id: Block(
                type: .paragraph,
                content: [InlineRun(text: "hello\nworld\nfoo bar")]
            )],
            rootChildren: [id]
        )
        XCTAssertEqual(Note.wordCount(of: ast), 4)
    }

    func testReadingTimeIsZeroForEmpty() {
        let note = Note(title: "X", body: .empty)
        XCTAssertEqual(note.readingTimeMinutes, 0)
    }

    func testReadingTimeRoundsUp() {
        let id = UUID()
        let text = String(repeating: "word ", count: 251)  // 251 words -> 2 min
        let ast = DocumentAST(
            blocks: [id: Block(type: .paragraph, content: [InlineRun(text: text)])],
            rootChildren: [id]
        )
        let note = Note(title: "X", body: ast)
        XCTAssertEqual(note.readingTimeMinutes, 2)
    }

    func testReadingTimeAtLeastOneForNonEmpty() {
        let id = UUID()
        let ast = DocumentAST(
            blocks: [id: Block(type: .paragraph, content: [InlineRun(text: "just five words here yes")])],
            rootChildren: [id]
        )
        let note = Note(title: "X", body: ast)
        XCTAssertGreaterThanOrEqual(note.readingTimeMinutes, 1)
    }

    // MARK: - Pin / archive convenience

    func testIsPinnedTracksPinnedAt() {
        var note = Note(title: "X")
        XCTAssertFalse(note.isPinned)
        note.pinnedAt = Date()
        XCTAssertTrue(note.isPinned)
        note.pinnedAt = nil
        XCTAssertFalse(note.isPinned)
    }

    func testIsArchivedTracksArchivedAt() {
        var note = Note(title: "X")
        XCTAssertFalse(note.isArchived)
        note.archivedAt = Date()
        XCTAssertTrue(note.isArchived)
        note.archivedAt = nil
        XCTAssertFalse(note.isArchived)
    }

    // MARK: - Entity type

    func testEntityTypeIsNote() {
        XCTAssertEqual(Note.entityType, "note")
    }

    func testSubtypeIsMarkdown() {
        XCTAssertEqual(Note.subtype, "markdown")
    }
}
