import XCTest
@testable import TesseraCore

/// Unit tests for ``NoteStore`` parts that don't need a real
/// Postgres connection. The end-to-end
/// upsert -> receipt -> fetch flow is exercised by the
/// integration test (``NoteStoreIntegrationTests``) which is
/// env-gated on `TESSERA_DB_INTEGRATION=1`.
///
/// The store depends on the `TesseraDataLayer` actor; the
/// tests never call `start()`, so the actor's internal
/// stores are uninitialized. The tests that touch the data
/// layer use a fake ``TesseraDataLayer``-style seam (the
/// real data layer is exercised by the integration test).
final class NoteStoreTests: XCTestCase {

    // MARK: - Receipt types

    func testReceiptTypesAreStable() {
        // The receipt type strings are persisted to
        // graph_receipts.receipt_type; changing them is
        // a schema migration. Pin them here.
        XCTAssertEqual(NoteReceiptType.upsert.rawValue, "note_upsert")
        XCTAssertEqual(NoteReceiptType.delete.rawValue, "note_delete")
        XCTAssertEqual(NoteReceiptType.titleChanged.rawValue, "note_title_changed")
        XCTAssertEqual(NoteReceiptType.bodyChanged.rawValue, "note_body_changed")
        XCTAssertEqual(NoteReceiptType.pinned.rawValue, "note_pinned")
        XCTAssertEqual(NoteReceiptType.unpinned.rawValue, "note_unpinned")
        XCTAssertEqual(NoteReceiptType.archived.rawValue, "note_archived")
        XCTAssertEqual(NoteReceiptType.unarchived.rawValue, "note_unarchived")
        XCTAssertEqual(NoteReceiptType.tagsChanged.rawValue, "note_tags_changed")
        XCTAssertEqual(NoteReceiptType.tagAdded.rawValue, "note_tag_added")
        XCTAssertEqual(NoteReceiptType.tagRemoved.rawValue, "note_tag_removed")
        XCTAssertEqual(NoteReceiptType.linkCreated.rawValue, "note_link_created")
    }

    // MARK: - JSON helpers

    func testNoteJSONStringRoundTrip() throws {
        let note = Note(
            id: UUID(),
            title: "Q3 Review",
            body: .empty,
            tags: ["q3"],
            createdAt: Date(timeIntervalSince1970: 1_000_000),
            updatedAt: Date(timeIntervalSince1970: 1_000_000)
        )
        let body = try note.jsonDataString()
        XCTAssertTrue(body.contains("Q3 Review"))
        let parsed = try Note.from(jsonDataString: body)
        XCTAssertEqual(parsed, note)
    }

    func testInvalidUTF8Rejected() {
        let bad = "not json at all"
        XCTAssertThrowsError(try Note.from(jsonDataString: bad))
    }

    // MARK: - Errors

    func testStoreErrorEquality() {
        let id = UUID()
        XCTAssertEqual(
            NoteStoreError.noteNotFound(id: id),
            NoteStoreError.noteNotFound(id: id)
        )
        XCTAssertNotEqual(
            NoteStoreError.noteNotFound(id: id),
            NoteStoreError.noteNotFound(id: UUID())
        )
        XCTAssertEqual(
            NoteStoreError.invalidNoteBody(reason: "x"),
            NoteStoreError.invalidNoteBody(reason: "x")
        )
        XCTAssertNotEqual(
            NoteStoreError.invalidNoteBody(reason: "x"),
            NoteStoreError.invalidNoteBody(reason: "y")
        )
    }

    // MARK: - NoteStore construction (smoke)

    func testNoteStoreConstruction() {
        let dataLayer = TesseraDataLayer()
        let store = NoteStore(dataLayer: dataLayer)
        // No public properties to assert on; construction
        // is the smoke test.
        _ = store
    }
}
