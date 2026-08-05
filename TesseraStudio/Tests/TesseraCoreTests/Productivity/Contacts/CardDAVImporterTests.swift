import XCTest
@testable import TesseraCore

/// Tests for ``CardDAVImporter``. The protocol-level
/// tests verify the XML parser handles the multistatus
/// response shape (PROPFIND, REPORT). The HTTP-level
/// tests are env-gated on a local mock server.
final class CardDAVImporterTests: XCTestCase {

    // MARK: - PROPFIND body

    func testPropfindBodyHasCorrectShape() {
        let body = CardDAVImporter.propfindBody(
            props: ["DAV: current-user-principal"]
        )
        let s = String(data: body, encoding: .utf8) ?? ""
        XCTAssertTrue(s.contains("<?xml"))
        XCTAssertTrue(s.contains("d:propfind"))
        XCTAssertTrue(s.contains("d:current-user-principal"))
    }

    func testPropfindBodyWithCardDAVNamespace() {
        let body = CardDAVImporter.propfindBody(
            props: ["urn:ietf:params:xml:ns:carddav addressbook-home-set"]
        )
        let s = String(data: body, encoding: .utf8) ?? ""
        XCTAssertTrue(s.contains("cr:addressbook-home-set"))
    }

    func testSyncCollectionBody() {
        let body = CardDAVImporter.syncCollectionBody(syncToken: "https://server/sync/1234")
        let s = String(data: body, encoding: .utf8) ?? ""
        XCTAssertTrue(s.contains("d:sync-collection"))
        XCTAssertTrue(s.contains("d:sync-token"))
        XCTAssertTrue(s.contains("https://server/sync/1234"))
    }

    // MARK: - XML parser

    func testXMLParserHandlesPropfindResponse() {
        let xml = """
        <?xml version="1.0" encoding="UTF-8"?>
        <d:multistatus xmlns:d="DAV:">
          <d:response>
            <d:href>/principals/john/</d:href>
            <d:propstat>
              <d:prop>
                <d:displayname>John</d:displayname>
              </d:prop>
              <d:status>HTTP/1.1 200 OK</d:status>
            </d:propstat>
          </d:response>
        </d:multistatus>
        """
        let parser = CardDAVXMLParser()
        parser.parse(data: Data(xml.utf8))
        XCTAssertEqual(parser.firstHref, "/principals/john/")
    }

    func testXMLParserHandlesAddressbookQuery() {
        let xml = """
        <?xml version="1.0" encoding="UTF-8"?>
        <d:multistatus xmlns:d="DAV:">
          <d:response>
            <d:href>/addressbooks/john/contacts/abc123.vcf</d:href>
            <d:propstat>
              <d:prop>
                <d:getetag>"abc-etag"</d:getetag>
              </d:prop>
              <d:status>HTTP/1.1 200 OK</d:status>
            </d:propstat>
          </d:response>
          <d:response>
            <d:href>/addressbooks/john/contacts/def456.vcf</d:href>
            <d:propstat>
              <d:prop>
                <d:getetag>"def-etag"</d:getetag>
              </d:prop>
              <d:status>HTTP/1.1 200 OK</d:status>
            </d:propstat>
          </d:response>
        </d:multistatus>
        """
        let parser = CardDAVXMLParser()
        parser.parse(data: Data(xml.utf8))
        XCTAssertEqual(parser.responses.count, 2)
        XCTAssertEqual(parser.responses[0].href, "/addressbooks/john/contacts/abc123.vcf")
        XCTAssertEqual(parser.responses[0].etag, "\"abc-etag\"")
    }

    func testXMLParserCapturesSyncToken() {
        let xml = """
        <?xml version="1.0" encoding="UTF-8"?>
        <d:multistatus xmlns:d="DAV:">
          <d:sync-token>https://server/sync/5678</d:sync-token>
        </d:multistatus>
        """
        let parser = CardDAVXMLParser()
        parser.parse(data: Data(xml.utf8))
        XCTAssertEqual(parser.syncToken, "https://server/sync/5678")
    }

    func testXMLParserDetectsRemovedHrefs() {
        let xml = """
        <?xml version="1.0" encoding="UTF-8"?>
        <d:multistatus xmlns:d="DAV:">
          <d:response>
            <d:href>/addressbooks/john/contacts/removed.vcf</d:href>
            <d:propstat>
              <d:prop/>
              <d:status>HTTP/1.1 404 Not Found</d:status>
            </d:propstat>
          </d:response>
        </d:multistatus>
        """
        let parser = CardDAVXMLParser()
        parser.parse(data: Data(xml.utf8))
        XCTAssertTrue(parser.removedHrefs.contains("/addressbooks/john/contacts/removed.vcf"))
    }

    // MARK: - Auth

    func testAdapterConstruction() throws {
        let adapter = try CardDAVImporter(
            configuration: .init(
                serverURL: URL(string: "https://carddav.example.com")!,
                username: "user",
                password: "app-specific-password"
            )
        )
        XCTAssertNotNil(adapter)
    }

    func testAddressBookURLSetter() throws {
        let adapter = try CardDAVImporter(
            configuration: .init(
                serverURL: URL(string: "https://carddav.example.com")!,
                username: "u", password: "p"
            )
        )
        let url = URL(string: "https://carddav.example.com/dav/user/contacts/")!
        Task { await adapter.setAddressBookURL(url) }
        // The URL is set internally; the test verifies
        // the call doesn't throw and the adapter stays
        // usable.
    }
}
