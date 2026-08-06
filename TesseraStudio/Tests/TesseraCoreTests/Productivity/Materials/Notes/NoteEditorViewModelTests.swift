import XCTest
@testable import TesseraCore

/// Tests for the pure / data-driven parts of
/// ``NoteEditorViewModel``: the local document state, the
/// draft fields, the focus mode hooks. The actor-dependent
/// parts (commit, link) are exercised by the integration
/// test.
@MainActor
final class NoteEditorViewModelTests: XCTestCase {

    private func makeEditor(note: Note? = nil) -> NoteEditorViewModel {
        let dataLayer = TesseraDataLayer()
        let store = NoteStore(dataLayer: dataLayer)
        let n = note ?? Note(
            id: UUID(),
            title: "Test Note",
            body: .empty,
            tags: ["q3"],
            createdAt: Date(timeIntervalSince1970: 1_000_000),
            updatedAt: Date(timeIntervalSince1970: 1_000_000)
        )
        return NoteEditorViewModel(note: n, store: store, userID: UUID())
    }

    // MARK: - Init

    func testInitCarriesNote() {
        let note = Note(title: "Hello", body: .empty, tags: ["tag1"])
        let editor = makeEditor(note: note)
        XCTAssertEqual(editor.note.title, "Hello")
        XCTAssertEqual(editor.draftTitle, "Hello")
        XCTAssertEqual(editor.draftTag, "")
    }

    func testInitHandlesEmptyNote() {
        let editor = makeEditor()
        XCTAssertEqual(editor.note.title, "Test Note")
        XCTAssertTrue(editor.document.blocks.isEmpty)
    }

    // MARK: - Document local

    func testSetDocumentLocalUpdatesBinding() {
        let editor = makeEditor()
        let newDoc = DocumentAST(
            blocks: [
                UUID(): Block(type: .paragraph, content: [InlineRun(text: "Hello")])
            ],
            rootChildren: [UUID()]
        )
        editor.setDocumentLocal(newDoc)
        XCTAssertEqual(editor.document.blocks.count, 1)
    }

    // MARK: - Refresh

    func testRefreshReplacesNote() {
        let editor = makeEditor()
        let newNote = Note(
            id: editor.note.id,
            title: "Updated",
            body: .empty,
            tags: ["new-tag"]
        )
        editor.refresh(with: newNote)
        XCTAssertEqual(editor.note.title, "Updated")
        XCTAssertEqual(editor.draftTitle, "Updated")
        XCTAssertEqual(editor.note.tags, ["new-tag"])
    }
}
