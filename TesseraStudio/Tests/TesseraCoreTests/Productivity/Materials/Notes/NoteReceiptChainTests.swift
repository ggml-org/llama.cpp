import XCTest
@testable import TesseraCore

/// Tests for the receipt types emitted by the Note surface.
/// The receipt type vocabulary is part of the data-layer
/// schema (changing the rawValue strings is a migration),
/// so the values are pinned here.
final class NoteReceiptChainTests: XCTestCase {

    // MARK: - Receipt type vocabulary

    func testAllReceiptTypesHaveRawValues() {
        // Each case has a stable string the data layer
        // persists. Changing a rawValue is a migration;
        // adding a new case is not.
        for type in NoteReceiptType.allCases {
            XCTAssertFalse(type.rawValue.isEmpty, "receipt type must have a raw value")
            XCTAssertTrue(type.rawValue.hasPrefix("note_"))
        }
    }

    func testNoteUpsertIsStable() {
        XCTAssertEqual(NoteReceiptType.upsert.rawValue, "note_upsert")
    }

    func testNoteDeleteIsStable() {
        XCTAssertEqual(NoteReceiptType.delete.rawValue, "note_delete")
    }

    func testPinUnpinAreStable() {
        XCTAssertEqual(NoteReceiptType.pinned.rawValue, "note_pinned")
        XCTAssertEqual(NoteReceiptType.unpinned.rawValue, "note_unpinned")
    }

    func testArchiveUnarchiveAreStable() {
        XCTAssertEqual(NoteReceiptType.archived.rawValue, "note_archived")
        XCTAssertEqual(NoteReceiptType.unarchived.rawValue, "note_unarchived")
    }

    func testTagChangeReceiptTypesAreStable() {
        XCTAssertEqual(NoteReceiptType.tagsChanged.rawValue, "note_tags_changed")
        XCTAssertEqual(NoteReceiptType.tagAdded.rawValue, "note_tag_added")
        XCTAssertEqual(NoteReceiptType.tagRemoved.rawValue, "note_tag_removed")
    }

    func testBodyAndTitleChangeReceiptTypesAreStable() {
        XCTAssertEqual(NoteReceiptType.titleChanged.rawValue, "note_title_changed")
        XCTAssertEqual(NoteReceiptType.bodyChanged.rawValue, "note_body_changed")
    }

    func testLinkReceiptTypeIsStable() {
        XCTAssertEqual(NoteReceiptType.linkCreated.rawValue, "note_link_created")
    }

    // MARK: - Receipt types are Codable + Sendable + Equatable

    func testReceiptTypesAreCodable() throws {
        let original: NoteReceiptType = .pinned
        let encoder = JSONEncoder()
        let data = try encoder.encode(original)
        let decoder = JSONDecoder()
        let decoded = try decoder.decode(NoteReceiptType.self, from: data)
        XCTAssertEqual(decoded, original)
    }

    func testReceiptTypesRawValueIsTheStringPersisted() {
        // The data layer persists `receiptType.rawValue`
        // to the `receipt_type` column. The raw value
        // is what shows up in the receipt chain.
        XCTAssertEqual(NoteReceiptType.upsert.rawValue, "note_upsert")
        XCTAssertEqual(NoteReceiptType.bodyChanged.rawValue, "note_body_changed")
        XCTAssertEqual(NoteReceiptType.linkCreated.rawValue, "note_link_created")
    }
}
