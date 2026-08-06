import XCTest
@testable import TesseraCore

/// Pure tests for the calendar store's persisted vocabulary
/// (receipt types, link types) and the event defaults. The
/// store's CRUD paths need Postgres and are covered by
/// ``CalendarStoreIntegrationTests`` (env-gated on
/// `TESSERA_DB_INTEGRATION=1`).
final class CalendarStoreTests: XCTestCase {

    /// The receipt strings land in `graph_receipts.receipt_type`;
    /// the receipts drawer and any downstream analytics key off
    /// them. Changing one is a schema event - pin the vocabulary.
    func testReceiptTypeVocabularyIsPinned() {
        XCTAssertEqual(CalendarEventReceiptType.allCases.map(\.rawValue), [
            "event_created",
            "event_updated",
            "event_deleted",
            "event_responded",
            "event_link_created",
        ])
    }

    /// The link types land in `entity_links.link_type`; the
    /// graph view's edge coloring keys off them.
    func testLinkTypeVocabularyIsPinned() {
        XCTAssertEqual(CalendarLinkType.allCases.map(\.rawValue), [
            "attendee_of",
            "prep_document",
            "prep_task",
            "reminder_for",
        ])
    }

    func testDefaultDurationIsOneHour() {
        // Fantastical parity: a start with no end gets one
        // hour. The NLU parser's Defaults borrow this value.
        XCTAssertEqual(CalendarEvent.defaultDuration, 3600)
        XCTAssertEqual(
            CalendarNLUParser.Defaults().defaultDuration,
            CalendarEvent.defaultDuration
        )
    }

    func testStoreErrorsAreEquatable() {
        let id = UUID()
        XCTAssertEqual(CalendarStoreError.eventNotFound(id: id), CalendarStoreError.eventNotFound(id: id))
        XCTAssertNotEqual(CalendarStoreError.eventNotFound(id: id), CalendarStoreError.noAttendees(eventID: id))
    }
}
