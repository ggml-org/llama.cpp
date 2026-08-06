import Foundation
@testable import TesseraCore

// MARK: - InMemoryCalendarStore

/// A ``CalendarStoring`` fake for the chat handler + view
/// model tests. Records every mutation so tests can assert
/// the create / update / delete / respond flow without
/// Postgres.
actor InMemoryCalendarStore: CalendarStoring {

    private(set) var events: [UUID: CalendarEvent] = [:]
    private(set) var responses: [(eventID: UUID, status: CalendarEvent.ResponseStatus)] = []
    private(set) var deletedIDs: [UUID] = []

    /// Optional failure injection (the "failed" chat state).
    var upsertError: Error?

    func setUpsertError(_ error: Error?) {
        upsertError = error
    }

    func upsert(_ event: CalendarEvent) async throws -> CalendarEvent {
        if let upsertError { throw upsertError }
        events[event.id] = event
        return event
    }

    func get(id: UUID) async throws -> CalendarEvent? {
        events[id]
    }

    func delete(id: UUID) async throws -> Bool {
        guard events.removeValue(forKey: id) != nil else { return false }
        deletedIDs.append(id)
        return true
    }

    func list(limit: Int) async throws -> [CalendarEvent] {
        Array(events.values.sorted { $0.startAt < $1.startAt }.prefix(limit))
    }

    func events(in range: ClosedRange<Date>, calendar: Calendar) async throws -> [CalendarEvent] {
        events.values
            .filter { !$0.occurrences(in: range, calendar: calendar).isEmpty }
            .sorted { $0.startAt < $1.startAt }
    }

    func search(matching query: String, limit: Int) async throws -> [CalendarEvent] {
        let q = query.lowercased()
        return Array(
            events.values
                .filter { $0.title.lowercased().contains(q) }
                .sorted { $0.startAt < $1.startAt }
                .prefix(limit)
        )
    }

    func respond(
        to eventID: UUID,
        attendeeIndex: Int?,
        attendeeName: String?,
        status: CalendarEvent.ResponseStatus
    ) async throws -> CalendarEvent {
        guard var event = events[eventID] else {
            throw CalendarStoreError.eventNotFound(id: eventID)
        }
        if !event.attendees.isEmpty {
            let index = attendeeIndex ?? 0
            if event.attendees.indices.contains(index) {
                event.attendees[index].responseStatus = status
            }
        }
        events[eventID] = event
        responses.append((eventID, status))
        return event
    }
}

// MARK: - Test fixtures

enum CalendarFixtures {

    /// A fixed reference date: Wednesday 2026-08-05 10:00
    /// local. Every NLU test pins "now" to this instant so
    /// relative phrases ("tomorrow", "next monday") are
    /// deterministic.
    static func referenceDate(calendar: Calendar) -> Date {
        var c = DateComponents()
        c.year = 2026
        c.month = 8
        c.day = 5
        c.hour = 10
        c.minute = 0
        return calendar.date(from: c)!
    }

    static func calendar(timeZone: TimeZone = .current) -> Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = timeZone
        calendar.firstWeekday = 1
        return calendar
    }

    static func contact(first: String, last: String, email: String? = nil) -> Contact {
        let now = Date(timeIntervalSince1970: 1_000_000)
        return Contact(
            name: NameComponents(first: first, last: last),
            emails: email.map { [LabeledEmail(label: .work, value: $0, isPrimary: true)] } ?? [],
            createdAt: now,
            updatedAt: now
        )
    }

    static func event(
        title: String,
        startAt: Date,
        duration: TimeInterval = 3600,
        allDay: Bool = false,
        attendees: [CalendarEvent.Attendee] = [],
        recurrence: CalendarEvent.Recurrence? = nil
    ) -> CalendarEvent {
        CalendarEvent(
            title: title,
            startAt: startAt,
            endAt: startAt.addingTimeInterval(duration),
            allDay: allDay,
            attendees: attendees,
            recurrence: recurrence
        )
    }

    static func parser(
        contacts: [Contact] = [],
        documents: [ResolvedDocument] = [],
        coordinates: [String: CalendarEvent.Coordinate] = [:],
        calendar: Calendar = CalendarFixtures.calendar(),
        referenceDate: Date? = nil
    ) -> CalendarNLUParser {
        CalendarNLUParser(
            contacts: StaticContactsAdapter(contacts: contacts),
            documents: StaticDocumentResolver(documents: documents),
            locations: StaticLocationResolver(coordinates: coordinates),
            calendar: calendar,
            referenceDate: referenceDate ?? CalendarFixtures.referenceDate(calendar: calendar)
        )
    }
}
