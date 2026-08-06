import XCTest
@testable import TesseraCore

/// Tests for the Note's linked-entity model. The cross-
/// surface link vocabulary (related_to, references,
/// attendee_of, ...) is shared with the contact / event
/// surfaces; these tests exercise the note-side seam
/// (the `linkedEntityIDs` denormalized cache and the
/// `NoteStore.link` integration).
final class NoteLinkedEntitiesTests: XCTestCase {

    // MARK: - linkedEntityIDs denormalized cache

    func testLinkedEntityIDsInit() {
        let id1 = UUID()
        let id2 = UUID()
        let note = Note(
            id: UUID(),
            title: "X",
            body: .empty,
            linkedEntityIDs: [id1, id2]
        )
        XCTAssertEqual(note.linkedEntityIDs, [id1, id2])
    }

    func testLinkedEntityIDsAppend() {
        var note = Note(title: "X", body: .empty, linkedEntityIDs: [UUID()])
        let newID = UUID()
        note.linkedEntityIDs.append(newID)
        XCTAssertEqual(note.linkedEntityIDs.count, 2)
        XCTAssertTrue(note.linkedEntityIDs.contains(newID))
    }

    func testLinkedEntityIDsRemove() {
        let id1 = UUID()
        let id2 = UUID()
        var note = Note(title: "X", body: .empty, linkedEntityIDs: [id1, id2])
        note.linkedEntityIDs.removeAll { $0 == id1 }
        XCTAssertEqual(note.linkedEntityIDs, [id2])
    }

    // MARK: - JSON round-trip

    func testLinkedEntityIDsRoundTrip() throws {
        let id1 = UUID()
        let id2 = UUID()
        let note = Note(
            id: UUID(),
            title: "X",
            body: .empty,
            linkedEntityIDs: [id1, id2]
        )
        let data = try note.jsonData()
        let decoded = try Note.from(jsonData: data)
        XCTAssertEqual(decoded.linkedEntityIDs, [id1, id2])
    }

    // MARK: - NoteStore.link (smoke)

    func testNoteStoreLinkThrowsWithoutDataLayer() async {
        // The store's `link` call requires the data layer
        // to be connected. Without `start()` the call
        // throws — the test verifies the error path so
        // callers know what to expect.
        let dataLayer = TesseraDataLayer()
        let store = NoteStore(dataLayer: dataLayer)
        do {
            _ = try await store.link(
                noteID: UUID(),
                to: UUID(),
                linkType: "related_to"
            )
            XCTFail("expected error from unconnected data layer")
        } catch {
            // The error path is exercised; the specific
            // error is the data layer's (uninitialized).
            // We just confirm the call throws.
            _ = error
        }
    }
}
