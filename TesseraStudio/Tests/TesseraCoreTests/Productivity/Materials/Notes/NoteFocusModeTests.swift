import XCTest
@testable import TesseraCore

/// Tests for the focus mode data flow on the
/// ``NotesViewModel`` and ``NoteEditorViewModel``. The
/// focus mode toggle, the exit condition, and the
/// derived state (word count, reading time) are
/// exercised here. The SwiftUI animation is not
/// exercised — the tests cover the data layer.
@MainActor
final class NoteFocusModeTests: XCTestCase {

    private func makeViewModel() -> NotesViewModel {
        let dataLayer = TesseraDataLayer()
        let store = NoteStore(dataLayer: dataLayer)
        return NotesViewModel(store: store, dataLayer: dataLayer)
    }

    // MARK: - Toggle

    func testToggleFocusModeTurnsOn() {
        let viewModel = makeViewModel()
        XCTAssertFalse(viewModel.isFocusMode)
        viewModel.toggleFocusMode()
        XCTAssertTrue(viewModel.isFocusMode)
    }

    func testToggleFocusModeTurnsOff() {
        let viewModel = makeViewModel()
        viewModel.toggleFocusMode()  // on
        viewModel.toggleFocusMode()  // off
        XCTAssertFalse(viewModel.isFocusMode)
    }

    func testExitFocusModeNoOpWhenAlreadyOff() {
        let viewModel = makeViewModel()
        viewModel.exitFocusMode()
        XCTAssertFalse(viewModel.isFocusMode)
    }

    func testExitFocusModeTurnsOff() {
        let viewModel = makeViewModel()
        viewModel.toggleFocusMode()
        XCTAssertTrue(viewModel.isFocusMode)
        viewModel.exitFocusMode()
        XCTAssertFalse(viewModel.isFocusMode)
    }

    // MARK: - Word count / reading time on the view model

    func testEditorWordCount() {
        let dataLayer = TesseraDataLayer()
        let store = NoteStore(dataLayer: dataLayer)
        let id = UUID()
        let note = Note(
            id: id,
            title: "X",
            body: DocumentAST(
                blocks: [id: Block(
                    type: .paragraph,
                    content: [InlineRun(text: "the quick brown fox")]
                )],
                rootChildren: [id]
            )
        )
        let editor = NoteEditorViewModel(note: note, store: store, userID: UUID())
        XCTAssertEqual(editor.note.wordCount, 4)
    }

    func testEditorReadingTime() {
        let dataLayer = TesseraDataLayer()
        let store = NoteStore(dataLayer: dataLayer)
        let id = UUID()
        // 600 words -> 3 min read (600/250 = 2.4, rounded up = 3)
        let text = String(repeating: "word ", count: 600)
        let note = Note(
            id: id,
            title: "X",
            body: DocumentAST(
                blocks: [id: Block(type: .paragraph, content: [InlineRun(text: text)])],
                rootChildren: [id]
            )
        )
        let editor = NoteEditorViewModel(note: note, store: store, userID: UUID())
        XCTAssertEqual(editor.note.readingTimeMinutes, 3)
    }
}
