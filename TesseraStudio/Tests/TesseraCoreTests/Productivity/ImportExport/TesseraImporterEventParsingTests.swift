import XCTest
@testable import TesseraCore

/// Tests for the importer's event-stream parser. The parser
/// is pure (no subprocess); we feed it sample JSON and check
/// the resulting events.
final class TesseraImporterEventParsingTests: XCTestCase {

    /// A canonical "import_ok" event parses to .ok with the
    /// expected fields.
    func testParseOkEvent() {
        let json = """
        {"event":"import_ok","path":"/tmp/a.docx","format":"docx",
         "parser":"python-docx",
         "entities":[{"entity_id":"11111111-1111-1111-1111-111111111111","entity_type":"document"}],
         "receipt_ids":["22222222-2222-2222-2222-222222222222"],
         "elapsed_seconds":0.34}
        """
        let events = TesseraImporter.Event.parseStream(json)
        XCTAssertEqual(events.count, 1)
        guard case let .ok(ok) = events[0] else {
            XCTFail("expected .ok, got \(events[0])")
            return
        }
        XCTAssertEqual(ok.path, "/tmp/a.docx")
        XCTAssertEqual(ok.format, "docx")
        XCTAssertEqual(ok.parser, "python-docx")
        XCTAssertEqual(ok.entities.count, 1)
        XCTAssertEqual(ok.entities[0].entityID.uuidString, "11111111-1111-1111-1111-111111111111")
        XCTAssertEqual(ok.entities[0].entityType, "document")
        XCTAssertEqual(ok.receiptIDs.first, "22222222-2222-2222-2222-222222222222")
        XCTAssertEqual(ok.elapsedSeconds, 0.34, accuracy: 0.001)
    }

    /// A canonical "import_failed" event parses to .failed.
    func testParseFailedEvent() {
        let json = """
        {"event":"import_failed","path":"/tmp/bad.bin","reason":"unsupported format"}
        """
        let events = TesseraImporter.Event.parseStream(json)
        XCTAssertEqual(events.count, 1)
        guard case let .failed(f) = events[0] else {
            XCTFail("expected .failed, got \(events[0])")
            return
        }
        XCTAssertEqual(f.path, "/tmp/bad.bin")
        XCTAssertEqual(f.reason, "unsupported format")
    }

    /// A multi-line stream parses to the expected number of
    /// events; non-JSON lines are ignored.
    func testParseStream() {
        let json = """
        {"event":"import_ok","path":"/tmp/a","format":"md","parser":"markdown-it-py","entities":[],"receipt_ids":[],"elapsed_seconds":0.1}
        {"event":"import_failed","path":"/tmp/b","reason":"oops"}
        not json
        {"event":"summary","ok":1,"failed":1,"elapsed":0.2}
        """
        let events = TesseraImporter.Event.parseStream(json)
        XCTAssertEqual(events.count, 3)
    }

    /// An event with a malformed UUID is treated as missing
    /// (the entity is dropped) rather than throwing.
    func testMalformedUUIDIsSkipped() {
        let json = """
        {"event":"import_ok","path":"/tmp/a","format":"md","parser":"markdown-it-py",
         "entities":[{"entity_id":"not-a-uuid","entity_type":"document"}],
         "receipt_ids":[],"elapsed_seconds":0.1}
        """
        let events = TesseraImporter.Event.parseStream(json)
        guard case let .ok(ok) = events[0] else {
            XCTFail("expected .ok")
            return
        }
        XCTAssertTrue(ok.entities.isEmpty, "malformed UUID entity should be dropped")
    }
}
