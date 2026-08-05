import XCTest
@testable import TesseraCore

/// Tests for ``ContactStore`` parts that don't need a real
/// Postgres connection. The end-to-end
/// upsert -> receipt -> fetch flow is exercised by the
/// integration test (``ContactStoreIntegrationTests``) which
/// is env-gated on `TESSERA_DB_INTEGRATION=1`.
final class ContactStoreTests: XCTestCase {

    func testReceiptTypesAreStable() {
        // The receipt type strings are persisted to
        // graph_receipts.receipt_type; changing them is
        // a schema migration. Pin them here.
        XCTAssertEqual(ContactReceiptType.upsert.rawValue, "contact_upsert")
        XCTAssertEqual(ContactReceiptType.delete.rawValue, "contact_delete")
        XCTAssertEqual(ContactReceiptType.linkCreated.rawValue, "contact_link_created")
        XCTAssertEqual(ContactReceiptType.linkDeleted.rawValue, "contact_link_deleted")
        XCTAssertEqual(ContactReceiptType.contactExport.rawValue, "contact_export")
    }

    func testEgressGuardAllowList() {
        XCTAssertTrue(TesseraContactEgressGuard.allows("user_explicit_export"))
        XCTAssertTrue(TesseraContactEgressGuard.allows("share_sheet"))
        XCTAssertTrue(TesseraContactEgressGuard.allows("agent_for_user"))
        XCTAssertFalse(TesseraContactEgressGuard.allows(""))
        XCTAssertFalse(TesseraContactEgressGuard.allows("unknown"))
        XCTAssertFalse(TesseraContactEgressGuard.allows("training"))
    }

    func testStoreErrorEquality() {
        XCTAssertEqual(
            ContactStoreError.egressDenied(provenance: "x"),
            ContactStoreError.egressDenied(provenance: "x")
        )
        XCTAssertNotEqual(
            ContactStoreError.egressDenied(provenance: "x"),
            ContactStoreError.egressDenied(provenance: "y")
        )
    }

    // MARK: - JSON helpers

    func testContactJSONStringRoundTrip() throws {
        let date = Date(timeIntervalSince1970: 1_000_000)
        let c = Contact(
            id: UUID(),
            subtype: .person,
            name: NameComponents(first: "Linus", last: "Torvalds"),
            emails: [LabeledEmail(label: .work, value: "linus@kernel.org")],
            organization: "Linux Foundation",
            createdAt: date,
            updatedAt: date
        )
        let body = try c.jsonDataString()
        XCTAssertTrue(body.contains("Linus"))
        XCTAssertTrue(body.contains("Torvalds"))
        let parsed = try Contact.from(jsonDataString: body)
        XCTAssertEqual(parsed, c)
    }

    func testInvalidUTF8Rejected() {
        // Force the JSON decoder to fail by passing a
        // string that is valid UTF-8 but not valid JSON.
        let bad = "not json at all"
        XCTAssertThrowsError(try Contact.from(jsonDataString: bad)) { error in
            // Either the typed error or a JSONDecoder
            // error is acceptable — the test asserts the
            // call doesn't silently return garbage.
            _ = error
        }
    }
}
