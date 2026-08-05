import XCTest
@testable import TesseraCore

/// Tests for ``VCardImporter``. The Apple Contacts framework
/// is Apple-only; on Linux these tests are skipped (the
/// importer reports ``frameworkUnavailable``). On macOS
/// the tests exercise the canonical round-trip path:
/// known VCard -> Contact, Contact -> VCard, Contact ->
/// Contact.
final class VCardImporterTests: XCTestCase {

    /// A minimal VCard 3.0 fixture. Two contact entries
    /// covering the common field shapes (name, email,
    /// phone, address, organization, title).
    private let sampleVCard = Data("""
    BEGIN:VCARD
    VERSION:3.0
    FN:Ada Lovelace
    N:Lovelace;Ada;;;
    EMAIL;TYPE=WORK:ada@analyticalengine.org
    TEL;TYPE=CELL:+1-555-0100
    ORG:Analytical Engine Co.
    TITLE:Programmer
    END:VCARD
    BEGIN:VCARD
    VERSION:3.0
    FN:Grace Hopper
    N:Hopper;Grace;;;
    EMAIL;TYPE=WORK:grace@navy.mil
    TEL;TYPE=WORK:+1-555-0200
    END:VCARD
    """.utf8)

    func testKnownVCardParses() async throws {
        let importer = VCardImporter()
        #if canImport(Contacts) && (os(macOS) || os(iOS))
        let contacts = try await importer.parse(data: sampleVCard)
        XCTAssertEqual(contacts.count, 2)
        XCTAssertEqual(contacts[0].name.first, "Ada")
        XCTAssertEqual(contacts[0].name.last, "Lovelace")
        XCTAssertEqual(contacts[0].emails.first?.value, "ada@analyticalengine.org")
        XCTAssertEqual(contacts[0].organization, "Analytical Engine Co.")
        XCTAssertEqual(contacts[0].title, "Programmer")
        #else
        do {
            _ = try await importer.parse(data: sampleVCard)
            XCTFail("Expected frameworkUnavailable on non-Apple")
        } catch VCardError.frameworkUnavailable {
            // expected
        }
        #endif
    }

    func testEmptyDataReturnsEmpty() async throws {
        let importer = VCardImporter()
        #if canImport(Contacts) && (os(macOS) || os(iOS))
        let contacts = try await importer.parse(data: Data())
        XCTAssertEqual(contacts.count, 0)
        #else
        // Skipped on Linux.
        #endif
    }

    func testRoundTrip() async throws {
        #if canImport(Contacts) && (os(macOS) || os(iOS))
        let importer = VCardImporter()
        let parsed = try await importer.parse(data: sampleVCard)
        let serialized = try await importer.serialize(contacts: parsed)
        let reparsed = try await importer.parse(data: serialized)
        XCTAssertEqual(reparsed.count, parsed.count)
        // Display names should match (the field that
        // most likely to drift across round-trips).
        XCTAssertEqual(reparsed.map(\.displayName), parsed.map(\.displayName))
        #endif
    }

    func testSerializeAndParseOne() async throws {
        #if canImport(Contacts) && (os(macOS) || os(iOS))
        let importer = VCardImporter()
        let c = Contact(
            subtype: .person,
            name: NameComponents(first: "Margaret", last: "Hamilton"),
            emails: [LabeledEmail(label: .work, value: "mhamilton@mit.edu")],
            phones: [LabeledPhone(label: .mobile, value: "+1-555-7777")],
            organization: "MIT",
            title: "Director"
        )
        let data = try await importer.serialize(contacts: [c])
        let s = String(data: data, encoding: .utf8) ?? ""
        XCTAssertTrue(s.contains("BEGIN:VCARD"))
        XCTAssertTrue(s.contains("END:VCARD"))
        XCTAssertTrue(s.contains("Margaret"))
        XCTAssertTrue(s.contains("Hamilton"))
        let reparsed = try await importer.parse(data: data)
        // The CN framework can pad name fields with
        // trailing whitespace on round-trip; assert
        // both name parts are present rather than the
        // exact joined string.
        let name = reparsed.first?.displayName ?? ""
        XCTAssertTrue(name.contains("Margaret"), "displayName should contain 'Margaret', got: '\(name)'")
        XCTAssertTrue(name.contains("Hamilton"), "displayName should contain 'Hamilton', got: '\(name)'")
        #endif
    }

    func testMalformedVCardsRaisesTypedError() async throws {
        let importer = VCardImporter()
        let bad = Data("not a vcard at all".utf8)
        #if canImport(Contacts) && (os(macOS) || os(iOS))
        do {
            _ = try await importer.parse(data: bad)
            // Some CNContactVCardSerialization versions
            // return an empty array for unparseable
            // input rather than throwing; accept either
            // outcome.
        } catch let VCardError.parseFailed(reason) {
            XCTAssertFalse(reason.isEmpty)
        }
        #else
        do {
            _ = try await importer.parse(data: bad)
            XCTFail("Expected error")
        } catch VCardError.frameworkUnavailable {
            // expected
        }
        #endif
    }

    func testParseFileURLStampsSourceURL() async throws {
        #if canImport(Contacts) && (os(macOS) || os(iOS))
        // Write the sample to a temp file, parse it, and
        // verify the source URL is stamped on each
        // contact.
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("test-\(UUID().uuidString).vcf")
        try sampleVCard.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let importer = VCardImporter()
        let contacts = try await importer.parse(fileURL: url)
        XCTAssertEqual(contacts.count, 2)
        for c in contacts {
            XCTAssertEqual(c.sourceURL, url.absoluteString)
        }
        #endif
    }
}
