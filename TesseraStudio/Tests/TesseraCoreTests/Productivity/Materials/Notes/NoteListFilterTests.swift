import XCTest
@testable import TesseraCore

/// Tests for ``NoteListFilter``: filter application, sort
/// order, the per-filter display metadata.
final class NoteListFilterTests: XCTestCase {

    // MARK: - Cases

    func testAllCases() {
        XCTAssertEqual(
            NoteListFilter.allCases,
            [.all, .pinned, .archived]
        )
    }

    // MARK: - Display

    func testDisplayNames() {
        XCTAssertEqual(NoteListFilter.all.displayName, "All")
        XCTAssertEqual(NoteListFilter.pinned.displayName, "Pinned")
        XCTAssertEqual(NoteListFilter.archived.displayName, "Archived")
    }

    func testSystemImages() {
        XCTAssertEqual(NoteListFilter.all.systemImage, "note.text")
        XCTAssertEqual(NoteListFilter.pinned.systemImage, "pin.fill")
        XCTAssertEqual(NoteListFilter.archived.systemImage, "archivebox")
    }

    // MARK: - Apply: All

    func testAllExcludesArchived() {
        let note = makeNote(title: "regular", updatedAt: Date(timeIntervalSince1970: 1_000))
        let archived = makeNote(title: "archived", archivedAt: Date(timeIntervalSince1970: 2_000))
        let result = NoteListFilter.all.apply(to: [note, archived])
        XCTAssertEqual(result.map { $0.title }, ["regular"])
    }

    func testAllSortsByUpdatedAtDescending() {
        let older = makeNote(title: "older", updatedAt: Date(timeIntervalSince1970: 1_000))
        let newer = makeNote(title: "newer", updatedAt: Date(timeIntervalSince1970: 3_000))
        let middle = makeNote(title: "middle", updatedAt: Date(timeIntervalSince1970: 2_000))
        let result = NoteListFilter.all.apply(to: [older, newer, middle])
        XCTAssertEqual(result.map { $0.title }, ["newer", "middle", "older"])
    }

    // MARK: - Apply: Pinned

    func testPinnedIncludesOnlyPinnedAndNotArchived() {
        // The pinned filter is `isPinned && !isArchived`:
        // archived notes (even if pinned) are excluded.
        let pinned = makeNote(title: "pinned", pinnedAt: Date(timeIntervalSince1970: 2_000))
        let pinnedAndArchived = makeNote(
            title: "pinned+archived",
            pinnedAt: Date(timeIntervalSince1970: 3_000),
            archivedAt: Date(timeIntervalSince1970: 1_000)
        )
        let plain = makeNote(title: "plain")
        let result = NoteListFilter.pinned.apply(to: [pinned, pinnedAndArchived, plain])
        // Only the pinned (non-archived) note is in the result.
        XCTAssertEqual(result.map { $0.title }, ["pinned"])
    }

    func testPinnedSortsByPinnedAtDescending() {
        let firstPinned = makeNote(title: "first", pinnedAt: Date(timeIntervalSince1970: 1_000))
        let secondPinned = makeNote(title: "second", pinnedAt: Date(timeIntervalSince1970: 3_000))
        let thirdPinned = makeNote(title: "third", pinnedAt: Date(timeIntervalSince1970: 2_000))
        let result = NoteListFilter.pinned.apply(to: [firstPinned, secondPinned, thirdPinned])
        XCTAssertEqual(result.map { $0.title }, ["second", "third", "first"])
    }

    func testPinnedFilterExcludesUnpinned() {
        let plain = makeNote(title: "plain")
        let result = NoteListFilter.pinned.apply(to: [plain])
        XCTAssertTrue(result.isEmpty)
    }

    func testPinnedFilterExcludesArchived() {
        let pinnedAndArchived = makeNote(
            title: "pinned+archived",
            pinnedAt: Date(timeIntervalSince1970: 1_000),
            archivedAt: Date(timeIntervalSince1970: 2_000)
        )
        let result = NoteListFilter.pinned.apply(to: [pinnedAndArchived])
        XCTAssertTrue(result.isEmpty)
    }

    // MARK: - Apply: Archived

    func testArchivedIncludesOnlyArchived() {
        let archived = makeNote(title: "archived", archivedAt: Date(timeIntervalSince1970: 2_000))
        let plain = makeNote(title: "plain")
        let result = NoteListFilter.archived.apply(to: [archived, plain])
        XCTAssertEqual(result.map { $0.title }, ["archived"])
    }

    func testArchivedSortsByArchivedAtDescending() {
        let first = makeNote(title: "first", archivedAt: Date(timeIntervalSince1970: 1_000))
        let second = makeNote(title: "second", archivedAt: Date(timeIntervalSince1970: 3_000))
        let third = makeNote(title: "third", archivedAt: Date(timeIntervalSince1970: 2_000))
        let result = NoteListFilter.archived.apply(to: [first, second, third])
        XCTAssertEqual(result.map { $0.title }, ["second", "third", "first"])
    }

    // MARK: - Helpers

    private func makeNote(
        title: String,
        updatedAt: Date = Date(timeIntervalSince1970: 1_000_000),
        pinnedAt: Date? = nil,
        archivedAt: Date? = nil
    ) -> Note {
        Note(
            id: UUID(),
            title: title,
            body: .empty,
            tags: [],
            pinnedAt: pinnedAt,
            archivedAt: archivedAt,
            createdAt: updatedAt,
            updatedAt: updatedAt
        )
    }
}
