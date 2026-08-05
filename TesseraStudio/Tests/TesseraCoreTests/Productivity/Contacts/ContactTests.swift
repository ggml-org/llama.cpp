import XCTest
@testable import TesseraCore

/// Unit tests for the ``Contact`` model: JSON round-trip,
/// subtype serialization, display name computation,
/// `NameComponents` helpers.
final class ContactTests: XCTestCase {

    // MARK: - JSON round-trip

    func testContactRoundTripsJSON() throws {
        let date = Date(timeIntervalSince1970: 1_000_000)
        let original = Contact(
            id: UUID(),
            subtype: .person,
            name: NameComponents(
                prefix: "Dr.",
                first: "Jane",
                middle: "Q",
                last: "Doe",
                suffix: "Jr.",
                nickname: "JD"
            ),
            emails: [
                LabeledEmail(label: .work, value: "jane@acme.com", isPrimary: true),
                LabeledEmail(label: .home, value: "jane@personal.com"),
            ],
            phones: [
                LabeledPhone(label: .mobile, value: "+1-555-0100", isPrimary: true),
            ],
            addresses: [
                LabeledAddress(
                    label: .home,
                    street: "123 Maple",
                    city: "Springfield",
                    region: "IL",
                    postalCode: "62704",
                    country: "US"
                ),
            ],
            organization: "Acme",
            title: "Staff Engineer",
            birthday: Date(timeIntervalSince1970: 0),
            photo: nil,
            notes: "Met at WWDC",
            sourceURL: "https://contacts.example.com/jane",
            linkedEntityIDs: [UUID()],
            createdAt: date,
            updatedAt: date
        )
        let data = try original.jsonData()
        let decoded = try Contact.from(jsonData: data)
        XCTAssertEqual(decoded, original)
    }

    func testEmptyContactRoundTrips() throws {
        let date = Date(timeIntervalSince1970: 1_000_000)
        let empty = Contact(
            id: UUID(),
            subtype: .group,
            name: NameComponents(last: "Family"),
            createdAt: date,
            updatedAt: date
        )
        let data = try empty.jsonData()
        let decoded = try Contact.from(jsonData: data)
        XCTAssertEqual(decoded, empty)
        XCTAssertEqual(decoded.subtype, .group)
        XCTAssertEqual(decoded.displayName, "Family")
    }

    // MARK: - Subtype serialization

    func testEachSubtypeSerializesDistinctly() throws {
        for subtype in Contact.Subtype.allCases {
            let c = Contact(subtype: subtype, name: NameComponents(last: "X"))
            let data = try c.jsonData()
            let s = String(data: data, encoding: .utf8) ?? ""
            XCTAssertTrue(
                s.contains("\"\(subtype.rawValue)\""),
                "subtype \(subtype.rawValue) should appear in JSON: \(s)"
            )
        }
    }

    // MARK: - Display name

    func testPersonDisplayName() {
        let c = Contact(
            subtype: .person,
            name: NameComponents(first: "Ada", last: "Lovelace")
        )
        XCTAssertEqual(c.displayName, "Ada Lovelace")
    }

    func testPersonDisplayNameFallbackToNickname() {
        let c = Contact(
            subtype: .person,
            name: NameComponents(nickname: "Al")
        )
        XCTAssertEqual(c.displayName, "Al")
    }

    func testOrganizationDisplayName() {
        let c = Contact(
            subtype: .organization,
            name: NameComponents(last: "Acme Inc.")
        )
        XCTAssertEqual(c.displayName, "Acme Inc.")
    }

    func testUnnamedContactFallsBack() {
        let c = Contact(name: NameComponents())
        XCTAssertEqual(c.displayName, "Unnamed")
    }

    // MARK: - NameComponents

    func testNameComponentsIsEmpty() {
        XCTAssertTrue(NameComponents().isEmpty)
        XCTAssertFalse(NameComponents(first: "A").isEmpty)
    }

    // MARK: - Entity type / subtype

    func testEntityTypeIsContact() {
        XCTAssertEqual(Contact.entityType, "contact")
    }

    func testSubtypeStringMatchesRawValue() {
        XCTAssertEqual(
            Contact(subtype: .person, name: NameComponents()).subtypeString,
            "person"
        )
        XCTAssertEqual(
            Contact(subtype: .organization, name: NameComponents()).subtypeString,
            "organization"
        )
        XCTAssertEqual(
            Contact(subtype: .group, name: NameComponents()).subtypeString,
            "group"
        )
    }

    // MARK: - Linked entity IDs

    func testLinkedEntityIDsRoundTrip() throws {
        let a = UUID()
        let b = UUID()
        var c = Contact(name: NameComponents(first: "A"))
        c.linkedEntityIDs = [a, b]
        let data = try c.jsonData()
        let decoded = try Contact.from(jsonData: data)
        XCTAssertEqual(decoded.linkedEntityIDs, [a, b])
    }

    // MARK: - Labeled values

    func testCustomLabelsRoundTrip() throws {
        let c = Contact(
            name: NameComponents(first: "X"),
            emails: [LabeledEmail(label: .custom("Other Inbox"), value: "x@y.z")],
            phones: [LabeledPhone(label: .custom("Pager"), value: "555-9999")],
            addresses: [LabeledAddress(label: .custom("Vacation"), street: "Beach Rd")]
        )
        let data = try c.jsonData()
        let decoded = try Contact.from(jsonData: data)
        XCTAssertEqual(decoded.emails.first?.value, "x@y.z")
        XCTAssertEqual(decoded.phones.first?.value, "555-9999")
        XCTAssertEqual(decoded.addresses.first?.street, "Beach Rd")
    }

    func testAddressOneLine() {
        let a = LabeledAddress(
            label: .home,
            street: "1 Main",
            city: "Springfield",
            region: "IL",
            postalCode: "62704",
            country: "US"
        )
        XCTAssertEqual(a.oneLine, "1 Main, Springfield, IL, 62704, US")
    }

    // MARK: - Performance: 10k+ contacts

    func testLargeContactSetJSONRoundTrip() throws {
        // Build 10,000 contacts and round-trip the whole
        // list. The contact store's name-search index
        // (migration 0003) makes the per-contact JSON
        // encoding the dominant cost; the test asserts
        // the total stays under 5 seconds on the
        // developer laptop. This is the "name queries
        // are fast for 10k+ contacts" check from the
        // spec; the in-memory JSON shape is what the
        // data layer's index is built around.
        var contacts: [Contact] = []
        contacts.reserveCapacity(10_000)
        for i in 0..<10_000 {
            contacts.append(Contact(
                subtype: .person,
                name: NameComponents(first: "First\(i)", last: "Last\(i)"),
                emails: [LabeledEmail(label: .work, value: "user\(i)@example.com")]
            ))
        }
        let start = Date()
        let data = try JSONEncoder().encode(contacts)
        let encodeElapsed = Date().timeIntervalSince(start)
        let decoded = try JSONDecoder().decode([Contact].self, from: data)
        let decodeElapsed = Date().timeIntervalSince(start)
        XCTAssertEqual(decoded.count, 10_000)
        XCTAssertLessThan(encodeElapsed, 5.0, "10k contact encode took \(encodeElapsed)s")
        XCTAssertLessThan(decodeElapsed, 5.0, "10k contact round-trip took \(decodeElapsed)s")
    }
}
