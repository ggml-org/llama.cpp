import XCTest
@testable import TesseraCore

/// Tests for the pure / data-driven parts of
/// ``NotesViewModel``: filter application, local search,
/// tag-chip logic. The actor-dependent parts (load, save)
/// are exercised by the integration test.
@MainActor
final class NotesViewModelTests: XCTestCase {

    // MARK: - Setup

    private func makeViewModel() -> NotesViewModel {
        let dataLayer = TesseraDataLayer()
        let store = NoteStore(dataLayer: dataLayer)
        return NotesViewModel(store: store, dataLayer: dataLayer)
    }

    private func makeNote(
        title: String,
        updatedAt: Date = Date(timeIntervalSince1970: 1_000_000),
        tags: [String] = [],
        pinnedAt: Date? = nil,
        archivedAt: Date? = nil
    ) -> Note {
        Note(
            id: UUID(),
            title: title,
            body: .empty,
            tags: Note.normalizeTags(tags),
            pinnedAt: pinnedAt,
            archivedAt: archivedAt,
            createdAt: updatedAt,
            updatedAt: updatedAt
        )
    }

    // MARK: - Initial state

    func testInitialState() {
        let viewModel = makeViewModel()
        XCTAssertEqual(viewModel.allNotes.count, 0)
        XCTAssertEqual(viewModel.rows.count, 0)
        XCTAssertEqual(viewModel.filter, .all)
        XCTAssertNil(viewModel.selectedNoteID)
        XCTAssertNil(viewModel.activeTag)
        XCTAssertFalse(viewModel.isFocusMode)
        XCTAssertFalse(viewModel.isChatDriven)
        XCTAssertNil(viewModel.editor)
    }

    // MARK: - Apply filter (in-memory)

    func testApplyFilterProjectsRows() {
        let viewModel = makeViewModel()
        viewModel.setAllNotesForTesting([
            makeNote(title: "a", updatedAt: Date(timeIntervalSince1970: 2_000)),
            makeNote(title: "b", updatedAt: Date(timeIntervalSince1970: 1_000)),
        ])
        viewModel.applyFilter()
        XCTAssertEqual(viewModel.rows.count, 2)
        XCTAssertEqual(viewModel.rows.map { $0.title }, ["a", "b"])
    }

    func testApplyFilterSortsByUpdatedAt() {
        let viewModel = makeViewModel()
        viewModel.setAllNotesForTesting([
            makeNote(title: "old", updatedAt: Date(timeIntervalSince1970: 1_000)),
            makeNote(title: "new", updatedAt: Date(timeIntervalSince1970: 3_000)),
            makeNote(title: "mid", updatedAt: Date(timeIntervalSince1970: 2_000)),
        ])
        viewModel.applyFilter()
        XCTAssertEqual(viewModel.rows.map { $0.title }, ["new", "mid", "old"])
    }

    // MARK: - Active tag chip

    func testActiveTagFiltersRows() {
        let viewModel = makeViewModel()
        viewModel.setAllNotesForTesting([
            makeNote(title: "A", tags: ["q3", "review"]),
            makeNote(title: "B", tags: ["q3"]),
            makeNote(title: "C", tags: ["other"]),
        ])
        viewModel.setActiveTag("q3")
        XCTAssertEqual(viewModel.rows.map { $0.title }, ["A", "B"])
    }

    func testSetActiveTagNilClearsFilter() {
        let viewModel = makeViewModel()
        viewModel.setAllNotesForTesting([
            makeNote(title: "A", tags: ["q3"]),
            makeNote(title: "B", tags: ["other"]),
        ])
        viewModel.setActiveTag("q3")
        XCTAssertEqual(viewModel.rows.count, 1)
        viewModel.setActiveTag(nil)
        XCTAssertEqual(viewModel.rows.count, 2)
    }

    // MARK: - All tags

    func testAllTagsCollectsDistinctSorted() {
        let viewModel = makeViewModel()
        viewModel.setAllNotesForTesting([
            makeNote(title: "A", tags: ["q3", "review"]),
            makeNote(title: "B", tags: ["Q3", "urgent"]),  // duplicate after normalize
            makeNote(title: "C", tags: []),
        ])
        let tags = viewModel.allTags
        XCTAssertEqual(tags, ["q3", "review", "urgent"])
    }

    // MARK: - Local search

    func testLocalSearchByTitle() {
        let viewModel = makeViewModel()
        viewModel.setAllNotesForTesting([
            makeNote(title: "Q3 Review"),
            makeNote(title: "Q4 Planning"),
            makeNote(title: "Standup notes"),
        ])
        viewModel.applyLocalSearch("q3")
        XCTAssertEqual(viewModel.rows.map { $0.title }, ["Q3 Review"])
    }

    func testLocalSearchCaseInsensitive() {
        let viewModel = makeViewModel()
        viewModel.setAllNotesForTesting([
            makeNote(title: "Q3 Review"),
        ])
        viewModel.applyLocalSearch("Q3")
        XCTAssertEqual(viewModel.rows.map { $0.title }, ["Q3 Review"])
    }

    func testLocalSearchByTag() {
        let viewModel = makeViewModel()
        viewModel.setAllNotesForTesting([
            makeNote(title: "A", tags: ["urgent"]),
            makeNote(title: "B", tags: ["other"]),
        ])
        viewModel.applyLocalSearch("urgent")
        XCTAssertEqual(viewModel.rows.map { $0.title }, ["A"])
    }

    func testLocalSearchEmptyStringClears() {
        let viewModel = makeViewModel()
        viewModel.setAllNotesForTesting([
            makeNote(title: "A"),
            makeNote(title: "B"),
        ])
        viewModel.applyLocalSearch("xyz")
        XCTAssertEqual(viewModel.rows.count, 0)
        viewModel.applyLocalSearch("")
        XCTAssertEqual(viewModel.rows.count, 2)
    }

    // MARK: - Focus mode

    func testToggleFocusMode() {
        let viewModel = makeViewModel()
        XCTAssertFalse(viewModel.isFocusMode)
        viewModel.toggleFocusMode()
        XCTAssertTrue(viewModel.isFocusMode)
        viewModel.toggleFocusMode()
        XCTAssertFalse(viewModel.isFocusMode)
    }

    func testExitFocusModeOnlyWhenActive() {
        let viewModel = makeViewModel()
        viewModel.exitFocusMode()
        XCTAssertFalse(viewModel.isFocusMode)
        viewModel.toggleFocusMode()
        XCTAssertTrue(viewModel.isFocusMode)
        viewModel.exitFocusMode()
        XCTAssertFalse(viewModel.isFocusMode)
    }

    // MARK: - Filter switching

    func testFilterSwitchReprojectsRows() {
        let viewModel = makeViewModel()
        let pinnedNote = makeNote(
            title: "Pinned",
            updatedAt: Date(timeIntervalSince1970: 1_000),
            pinnedAt: Date(timeIntervalSince1970: 2_000)
        )
        let regularNote = makeNote(title: "Regular", updatedAt: Date(timeIntervalSince1970: 3_000))
        viewModel.setAllNotesForTesting([pinnedNote, regularNote])
        viewModel.applyFilter()
        XCTAssertEqual(viewModel.rows.map { $0.title }, ["Regular", "Pinned"])

        viewModel.filter = .pinned
        viewModel.applyFilter()
        XCTAssertEqual(viewModel.rows.map { $0.title }, ["Pinned"])
    }
}
