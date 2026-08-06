import XCTest
@testable import TesseraCore

/// Unit tests for ``EmailStore`` and the
/// ``EmailReceiptType`` taxonomy. These tests
/// don't need a Postgres connection; the
/// end-to-end upsert + receipt + fetch flow
/// lives in ``EmailStoreIntegrationTests`` (env
/// gated on `TESSERA_DB_INTEGRATION=1`).
final class EmailStoreTests: XCTestCase {

    // MARK: - Receipt types

    func testReceiptTypeStringsAreStable() {
        // The receipt type strings are persisted to
        // graph_receipts.receipt_type; changing
        // them is a schema migration. Pin the
        // vocabulary here.
        XCTAssertEqual(EmailReceiptType.upsert.rawValue, "email_upsert")
        XCTAssertEqual(EmailReceiptType.delete.rawValue, "email_delete")
        XCTAssertEqual(EmailReceiptType.read.rawValue, "email_read")
        XCTAssertEqual(EmailReceiptType.starred.rawValue, "email_starred")
        XCTAssertEqual(EmailReceiptType.folderChanged.rawValue, "email_folder_changed")
        XCTAssertEqual(EmailReceiptType.archived.rawValue, "email_archived")
        XCTAssertEqual(EmailReceiptType.trashed.rawValue, "email_trashed")
        XCTAssertEqual(EmailReceiptType.replied.rawValue, "email_replied")
        XCTAssertEqual(EmailReceiptType.forwarded.rawValue, "email_forwarded")
        XCTAssertEqual(EmailReceiptType.imported.rawValue, "email_imported")
        XCTAssertEqual(EmailReceiptType.linkCreated.rawValue, "email_link_created")
        XCTAssertEqual(EmailReceiptType.linkDeleted.rawValue, "email_link_deleted")
        XCTAssertEqual(EmailReceiptType.draftSaved.rawValue, "email_draft_saved")
        XCTAssertEqual(EmailReceiptType.routedToShareSheet.rawValue, "email_routed_to_share_sheet")
    }

    func testAllReceiptTypesHaveUniqueStrings() {
        // The taxonomy is the union of all cases;
        // two cases sharing a raw value is a
        // bug (the data layer can't
        // disambiguate).
        let raws = EmailReceiptType.allCases.map { $0.rawValue }
        XCTAssertEqual(Set(raws).count, raws.count, "duplicate raw values in EmailReceiptType")
    }

    // MARK: - Store error

    func testStoreErrorEquality() {
        let a = EmailStoreError.invalidEmailBody(reason: "x")
        let b = EmailStoreError.invalidEmailBody(reason: "x")
        XCTAssertEqual(a, b)
        let c = EmailStoreError.invalidEmailBody(reason: "y")
        XCTAssertNotEqual(a, c)
    }
}
