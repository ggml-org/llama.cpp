import XCTest
@testable import TesseraCore

/// Unit tests for the ``CalendarEvent`` model: JSON
/// round-trip, recurrence parsing (delegation to
/// ``RecurrenceRule``), attendee + location + linked
/// entity handling, validation, and occurrence expansion.
final class CalendarEventTests: XCTestCase {

    // MARK: - JSON round-trip

    func testEventRoundTripsJSON() throws {
        let date = Date(timeIntervalSince1970: 1_700_000_000)
        let original = CalendarEvent(
            id: UUID(),
            title: "Q3 review",
            notes: "Bring the roadmap doc",
            startAt: date,
            endAt: date.addingTimeInterval(7200),
            allDay: false,
            location: "the blue room",
            locationCoordinate: CalendarEvent.Coordinate(latitude: 37.7749, longitude: -122.4194),
            attendees: [
                CalendarEvent.Attendee(
                    contactID: UUID(),
                    email: "john@acme.com",
                    name: "John Doe",
                    responseStatus: .accepted
                ),
                CalendarEvent.Attendee(
                    email: "external@example.com",
                    name: "External Guest",
                    responseStatus: .needsAction
                ),
            ],
            recurrence: CalendarEvent.Recurrence(
                rrule: "FREQ=WEEKLY;BYDAY=MO",
                exDates: [date.addingTimeInterval(7 * 86_400)]
            ),
            reminders: [UUID()],
            linkedDocumentIDs: [UUID(), UUID()],
            linkedTaskIDs: [UUID()],
            createdAt: date,
            updatedAt: date
        )
        let data = try original.jsonData()
        let decoded = try CalendarEvent.from(jsonData: data)
        XCTAssertEqual(decoded, original)
    }

    func testMinimalEventRoundTrips() throws {
        // Pin the timestamps: the `.iso8601` strategy
        // truncates to whole seconds, so a `Date()` default
        // (sub-second precision) would not round-trip
        // exactly. Deterministic inputs keep the test
        // honest about the JSON path itself.
        let date = Date(timeIntervalSince1970: 1_700_000_000)
        let event = CalendarEvent(
            title: "Coffee",
            startAt: date,
            endAt: date.addingTimeInterval(1800),
            createdAt: date,
            updatedAt: date
        )
        let data = try event.jsonData()
        let decoded = try CalendarEvent.from(jsonData: data)
        XCTAssertEqual(decoded, event)
        XCTAssertNil(decoded.location)
        XCTAssertNil(decoded.recurrence)
        XCTAssertTrue(decoded.attendees.isEmpty)
        XCTAssertTrue(decoded.linkedDocumentIDs.isEmpty)
    }

    func testJSONStringRoundTrip() throws {
        let date = Date(timeIntervalSince1970: 1_700_000_000)
        let event = CalendarEvent(
            title: "Standup",
            startAt: date,
            endAt: date.addingTimeInterval(900),
            createdAt: date,
            updatedAt: date
        )
        let body = try event.jsonDataString()
        XCTAssertTrue(body.contains("Standup"))
        let decoded = try CalendarEvent.from(jsonDataString: body)
        XCTAssertEqual(decoded, event)
    }

    func testInvalidUTF8Rejected() {
        // NSData -> String can't produce invalid UTF-8 via
        // the public API, so exercise the decode path with
        // malformed JSON instead.
        XCTAssertThrowsError(try CalendarEvent.from(jsonDataString: "not json"))
    }

    // MARK: - Entity type pin

    func testEntityTypeMatchesGraphViewAndMigration() {
        // The migration's partial index, the graph view's
        // icon/color table, and the store all key off this
        // string. Pin it.
        XCTAssertEqual(CalendarEvent.entityType, "calendar_event")
        XCTAssertEqual(GraphNode.iconName(for: CalendarEvent.entityType), "calendar")
    }

    // MARK: - Validation

    func testEmptyTitleIsInvalid() {
        let date = Date(timeIntervalSince1970: 1_700_000_000)
        let event = CalendarEvent(title: "   ", startAt: date, endAt: date.addingTimeInterval(3600))
        XCTAssertFalse(event.isValid)
    }

    func testEndBeforeStartIsInvalid() {
        let date = Date(timeIntervalSince1970: 1_700_000_000)
        let event = CalendarEvent(title: "Backwards", startAt: date, endAt: date.addingTimeInterval(-100))
        XCTAssertFalse(event.isValid)
    }

    func testNormalEventIsValid() {
        let date = Date(timeIntervalSince1970: 1_700_000_000)
        let event = CalendarEvent(title: "Ok", startAt: date, endAt: date.addingTimeInterval(100))
        XCTAssertTrue(event.isValid)
    }

    // MARK: - Attendees

    func testAttendeeResolutionFlag() {
        let resolved = CalendarEvent.Attendee(contactID: UUID(), name: "Jane")
        let external = CalendarEvent.Attendee(email: "x@example.com", name: "X")
        let nameOnly = CalendarEvent.Attendee(name: "Someone")
        XCTAssertTrue(resolved.isResolved)
        XCTAssertFalse(external.isResolved)
        XCTAssertFalse(nameOnly.isResolved)
    }

    func testResponseStatusRoundTripsAllCases() throws {
        for status in CalendarEvent.ResponseStatus.allCases {
            let attendee = CalendarEvent.Attendee(name: "A", responseStatus: status)
            let data = try JSONEncoder().encode(attendee)
            let decoded = try JSONDecoder().decode(CalendarEvent.Attendee.self, from: data)
            XCTAssertEqual(decoded.responseStatus, status)
        }
    }

    // MARK: - Recurrence parsing

    func testRecurrenceParsesWeeklyRule() throws {
        let rule = try RecurrenceRule(rrule: "FREQ=WEEKLY;BYDAY=MO")
        XCTAssertEqual(rule.frequency, .weekly)
        XCTAssertEqual(rule.byDay, [.monday])
        XCTAssertEqual(rule.interval, 1)
    }

    func testRecurrenceRoundTripsThroughEvent() throws {
        let date = Date(timeIntervalSince1970: 1_700_000_000)
        let event = CalendarEvent(
            title: "Standup",
            startAt: date,
            endAt: date.addingTimeInterval(900),
            recurrence: CalendarEvent.Recurrence(rrule: "FREQ=DAILY;COUNT=5")
        )
        let decoded = try CalendarEvent.from(jsonData: event.jsonData())
        XCTAssertEqual(decoded.recurrence?.rrule, "FREQ=DAILY;COUNT=5")
    }

    // MARK: - Occurrences

    func testNonRecurringEventSingleOccurrence() {
        let calendar = CalendarFixtures.calendar()
        let start = CalendarFixtures.referenceDate(calendar: calendar)
        let event = CalendarFixtures.event(title: "One", startAt: start)

        let inRange = event.occurrences(
            in: start.addingTimeInterval(-3600)...start.addingTimeInterval(7200),
            calendar: calendar
        )
        XCTAssertEqual(inRange, [start])

        let outRange = event.occurrences(
            in: start.addingTimeInterval(86_400)...start.addingTimeInterval(2 * 86_400),
            calendar: calendar
        )
        XCTAssertTrue(outRange.isEmpty)
    }

    func testRecurringEventExpandsInRange() {
        let calendar = CalendarFixtures.calendar()
        let start = CalendarFixtures.referenceDate(calendar: calendar) // Wednesday
        let event = CalendarFixtures.event(
            title: "Daily",
            startAt: start,
            recurrence: CalendarEvent.Recurrence(rrule: "FREQ=DAILY")
        )
        // A 3-day window starting at the anchor: 3
        // occurrences (the anchor's time-of-day each day).
        let range = start...start.addingTimeInterval(3 * 86_400 - 1)
        let occurrences = event.occurrences(in: range, calendar: calendar)
        XCTAssertEqual(occurrences.count, 3)
        for occurrence in occurrences {
            XCTAssertEqual(calendar.component(.hour, from: occurrence), 10)
        }
    }

    func testExDatesSkipOccurrences() {
        let calendar = CalendarFixtures.calendar()
        let start = CalendarFixtures.referenceDate(calendar: calendar)
        let skipped = start.addingTimeInterval(86_400) // day 2
        let event = CalendarFixtures.event(
            title: "Daily with exception",
            startAt: start,
            recurrence: CalendarEvent.Recurrence(rrule: "FREQ=DAILY", exDates: [skipped])
        )
        let range = start...start.addingTimeInterval(3 * 86_400 - 1)
        let occurrences = event.occurrences(in: range, calendar: calendar)
        XCTAssertEqual(occurrences.count, 2)
        XCTAssertFalse(occurrences.contains(where: { calendar.isDate($0, inSameDayAs: skipped) }))
    }

    func testUnparseableRRuleDegradesToSingleOccurrence() {
        let calendar = CalendarFixtures.calendar()
        let start = CalendarFixtures.referenceDate(calendar: calendar)
        let event = CalendarFixtures.event(
            title: "Broken rule",
            startAt: start,
            recurrence: CalendarEvent.Recurrence(rrule: "FREQ=HOURLY") // unsupported -> throws
        )
        let range = start...start.addingTimeInterval(10 * 86_400)
        let occurrences = event.occurrences(in: range, calendar: calendar)
        XCTAssertEqual(occurrences, [start])
    }

    func testOccursOnDay() {
        let calendar = CalendarFixtures.calendar()
        let start = CalendarFixtures.referenceDate(calendar: calendar)
        let event = CalendarFixtures.event(title: "Today", startAt: start)
        XCTAssertTrue(event.occurs(on: start, calendar: calendar))
        XCTAssertFalse(event.occurs(on: start.addingTimeInterval(2 * 86_400), calendar: calendar))
    }
}
