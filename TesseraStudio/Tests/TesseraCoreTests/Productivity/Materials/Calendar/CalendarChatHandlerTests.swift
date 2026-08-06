import XCTest
@testable import TesseraCore

/// Tests for ``CalendarChatHandler``: the queue lifecycle
/// (pending -> applied / failed) and the five intents
/// (create, list, move, respond, delete). Runs against the
/// in-memory ``CalendarStoring`` fake - no Postgres.
final class CalendarChatHandlerTests: XCTestCase {

    private let calendar = CalendarFixtures.calendar()

    private enum TestError: Error {
        case boom
    }

    private func makeStack(
        contacts: [Contact] = []
    ) -> (handler: CalendarChatHandler, store: InMemoryCalendarStore) {
        let store = InMemoryCalendarStore()
        let parser = CalendarFixtures.parser(contacts: contacts, referenceDate: Date())
        let handler = CalendarChatHandler(store: store, parser: parser, calendar: calendar)
        return (handler, store)
    }

    private func tomorrow(at hour: Int = 12) -> Date {
        let day = calendar.date(byAdding: .day, value: 1, to: calendar.startOfDay(for: Date()))!
        var c = calendar.dateComponents([.year, .month, .day], from: day)
        c.hour = hour
        return calendar.date(from: c)!
    }

    private func nextWeekday(_ weekday: Int) -> Date {
        var day = calendar.startOfDay(for: Date())
        for _ in 1...7 {
            day = calendar.date(byAdding: .day, value: 1, to: day)!
            if calendar.component(.weekday, from: day) == weekday { return day }
        }
        return day
    }

    // MARK: - Create

    func testSubmitCreatesEvent() async throws {
        let (handler, store) = makeStack(
            contacts: [CalendarFixtures.contact(first: "John", last: "Appleseed")]
        )
        let outcome = try await handler.submit("schedule a meeting with John next monday at 2pm")

        XCTAssertEqual(outcome.kind, .created)
        XCTAssertNotNil(outcome.eventID)

        let events = await store.events
        XCTAssertEqual(events.count, 1)
        let event = events.values.first!
        XCTAssertEqual(event.title, "meeting")
        XCTAssertEqual(calendar.component(.hour, from: event.startAt), 14)
        XCTAssertTrue(calendar.isDate(event.startAt, inSameDayAs: nextWeekday(2)))
        XCTAssertEqual(event.attendees.first?.name, "John Appleseed")
        XCTAssertNotNil(event.attendees.first?.contactID)
    }

    // MARK: - List

    func testListShowsEventsInline() async throws {
        let (handler, store) = makeStack()
        _ = try await store.upsert(CalendarFixtures.event(title: "Lunch", startAt: tomorrow()))

        let outcome = try await handler.submit("what's on my calendar tomorrow?")

        XCTAssertEqual(outcome.kind, .listed)
        XCTAssertEqual(outcome.events.count, 1)
        XCTAssertEqual(outcome.events.first?.title, "Lunch")
        XCTAssertTrue(outcome.summary.contains("1 event"))
    }

    func testListEmptyRangeSaysNothing() async throws {
        let (handler, _) = makeStack()
        let outcome = try await handler.submit("what's on my calendar tomorrow?")
        XCTAssertEqual(outcome.kind, .listed)
        XCTAssertTrue(outcome.events.isEmpty)
        XCTAssertTrue(outcome.summary.contains("Nothing"))
    }

    // MARK: - Move

    func testMoveEventPreservesTimeOfDay() async throws {
        let (handler, store) = makeStack()
        let event = CalendarFixtures.event(title: "Q3 review", startAt: tomorrow(at: 14), duration: 7200)
        _ = try await store.upsert(event)

        let outcome = try await handler.submit("move the q3 review to friday")

        XCTAssertEqual(outcome.kind, .updated)
        XCTAssertEqual(outcome.eventID, event.id)

        let moved = await store.events[event.id]
        XCTAssertNotNil(moved)
        XCTAssertTrue(calendar.isDate(moved!.startAt, inSameDayAs: nextWeekday(6)))
        XCTAssertEqual(calendar.component(.hour, from: moved!.startAt), 14)
        XCTAssertEqual(moved!.endAt.timeIntervalSince(moved!.startAt), 7200)
    }

    func testMoveUsesWordOverlapForFuzzyTitles() async throws {
        let (handler, store) = makeStack()
        let event = CalendarFixtures.event(title: "Q3 planning review", startAt: tomorrow(at: 14))
        _ = try await store.upsert(event)

        // "q3 review" is not a substring of the title; the
        // word-overlap fallback resolves it anyway.
        let outcome = try await handler.submit("move q3 review to friday")
        XCTAssertEqual(outcome.kind, .updated)
        XCTAssertEqual(outcome.eventID, event.id)
    }

    func testMoveMissingEventFails() async throws {
        let (handler, _) = makeStack()
        let outcome = try await handler.submit("move the q3 review to friday")
        XCTAssertEqual(outcome.kind, .failed)
        XCTAssertTrue(outcome.summary.contains("Couldn't find"))
    }

    // MARK: - Respond

    func testRespondDecline() async throws {
        let (handler, store) = makeStack()
        let event = CalendarFixtures.event(
            title: "Dentist appointment",
            startAt: tomorrow(at: 9),
            attendees: [CalendarEvent.Attendee(name: "Me")]
        )
        _ = try await store.upsert(event)

        let outcome = try await handler.submit("decline the dentist")

        XCTAssertEqual(outcome.kind, .responded)
        let updated = await store.events[event.id]
        XCTAssertEqual(updated?.attendees.first?.responseStatus, .declined)
        let responses = await store.responses
        XCTAssertEqual(responses.first?.status, .declined)
    }

    func testRespondAccept() async throws {
        let (handler, store) = makeStack()
        let event = CalendarFixtures.event(
            title: "Standup",
            startAt: tomorrow(at: 9),
            attendees: [CalendarEvent.Attendee(name: "Me")]
        )
        _ = try await store.upsert(event)

        let outcome = try await handler.submit("accept standup")
        XCTAssertEqual(outcome.kind, .responded)
        let updated = await store.events[event.id]
        XCTAssertEqual(updated?.attendees.first?.responseStatus, .accepted)
    }

    // MARK: - Delete

    func testDeleteEvent() async throws {
        let (handler, store) = makeStack()
        let event = CalendarFixtures.event(title: "Standup", startAt: tomorrow(at: 9))
        _ = try await store.upsert(event)

        let outcome = try await handler.submit("cancel the standup")

        XCTAssertEqual(outcome.kind, .deleted)
        XCTAssertEqual(outcome.eventID, event.id)
        let events = await store.events
        XCTAssertTrue(events.isEmpty)
        let deleted = await store.deletedIDs
        XCTAssertEqual(deleted, [event.id])
    }

    func testDeleteMissingEventFails() async throws {
        let (handler, _) = makeStack()
        let outcome = try await handler.submit("cancel the standup")
        XCTAssertEqual(outcome.kind, .failed)
    }

    // MARK: - Queue lifecycle

    func testEnqueueThenProcessLifecycle() async throws {
        let (handler, _) = makeStack()
        let item = await handler.enqueue("Coffee with John tomorrow at noon")
        XCTAssertEqual(item.state, .pending)

        let queued = await handler.queue
        XCTAssertEqual(queued.count, 1)
        XCTAssertEqual(queued.first?.state, .pending)

        let outcome = try await handler.processNext()
        XCTAssertEqual(outcome?.kind, .created)

        let after = await handler.queue
        XCTAssertEqual(after.first?.state, .applied)
        XCTAssertNotNil(after.first?.outcome)
    }

    func testProcessNextWithEmptyQueueReturnsNil() async throws {
        let (handler, _) = makeStack()
        let outcome = try await handler.processNext()
        XCTAssertNil(outcome)
    }

    func testStoreFailureMarksItemFailed() async throws {
        let (handler, store) = makeStack()
        await store.setUpsertError(TestError.boom)

        let outcome = try await handler.submit("Lunch with John tomorrow at noon")

        XCTAssertEqual(outcome.kind, .failed)
        let queued = await handler.queue
        XCTAssertEqual(queued.first?.state, .failed)
    }

    // MARK: - Intent classification

    func testClassifyVerbsTakePriorityOverCreate() async {
        let (handler, _) = makeStack()

        if case .delete(let query) = await handler.classify("cancel lunch") {
            XCTAssertEqual(query, "lunch")
        } else {
            XCTFail("expected delete intent")
        }

        if case .respond(let query, let status) = await handler.classify("decline the dentist") {
            XCTAssertEqual(query, "dentist")
            XCTAssertEqual(status, .declined)
        } else {
            XCTFail("expected respond intent")
        }

        if case .move(let query, _) = await handler.classify("move the Q3 review to friday") {
            XCTAssertEqual(query, "Q3 review")
        } else {
            XCTFail("expected move intent")
        }
    }

    func testClassifyListQuestions() async {
        let (handler, _) = makeStack()
        guard case .list(let range) = await handler.classify("what's on my calendar tomorrow?") else {
            return XCTFail("expected list intent")
        }
        let tomorrow = calendar.date(byAdding: .day, value: 1, to: calendar.startOfDay(for: Date()))!
        XCTAssertTrue(range.contains(tomorrow))
        XCTAssertFalse(range.contains(calendar.startOfDay(for: Date()).addingTimeInterval(3600)))
    }

    func testClassifyCreateIsTheDefault() async {
        let (handler, _) = makeStack()
        guard case .create(let parsed) = await handler.classify("Lunch with John tomorrow at noon") else {
            return XCTFail("expected create intent")
        }
        XCTAssertEqual(parsed.title, "Lunch")
    }

    func testClassifyEmptyInputNotUnderstood() async {
        let (handler, _) = makeStack()
        let empty = await handler.classify("")
        let blank = await handler.classify("   ")
        XCTAssertEqual(empty, .notUnderstood)
        XCTAssertEqual(blank, .notUnderstood)
    }
}
