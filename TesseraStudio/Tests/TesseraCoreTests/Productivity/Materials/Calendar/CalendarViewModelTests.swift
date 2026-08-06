import XCTest
@testable import TesseraCore

/// Tests for ``CalendarViewModel`` + ``CalendarGridModel``
/// (the shared macOS/iOS state) and the graph-view
/// connector. Runs against the in-memory
/// ``CalendarStoring`` fake - no Postgres.
@MainActor
final class CalendarViewModelTests: XCTestCase {

    private let calendar = CalendarFixtures.calendar()
    /// Wednesday 2026-08-05 10:00 - the pinned "now".
    private var referenceDay: Date { CalendarFixtures.referenceDate(calendar: calendar) }

    private func makeStack(
        events: [CalendarEvent] = [],
        initialDate: Date? = nil
    ) async -> (vm: CalendarViewModel, store: InMemoryCalendarStore) {
        let store = InMemoryCalendarStore()
        for event in events {
            _ = try? await store.upsert(event)
        }
        let initial = initialDate ?? referenceDay
        let parser = CalendarFixtures.parser(referenceDate: initial)
        let handler = CalendarChatHandler(
            store: store, parser: parser, calendar: calendar, now: { initial }
        )
        let vm = CalendarViewModel(
            store: store, chatHandler: handler, calendar: calendar, initialDate: initial
        )
        return (vm, store)
    }

    private func day(_ offset: Int, hour: Int = 10) -> Date {
        let base = calendar.startOfDay(for: referenceDay)
        let day = calendar.date(byAdding: .day, value: offset, to: base)!
        var c = calendar.dateComponents([.year, .month, .day], from: day)
        c.hour = hour
        return calendar.date(from: c)!
    }

    // MARK: - Mode switching (spec: preserves the date)

    func testModeSwitchPreservesSelectedDate() async {
        let (vm, _) = await makeStack()
        let focus = day(3)
        vm.selectedDate = focus

        for mode in CalendarViewMode.allCases {
            vm.setViewMode(mode)
            XCTAssertEqual(vm.viewMode, mode)
            XCTAssertTrue(
                calendar.isDate(vm.selectedDate, inSameDayAs: focus),
                "switching to \(mode) must not move the selected date"
            )
        }
    }

    func testVisibleRangeContainsSelectedDateInEveryMode() async {
        let (vm, _) = await makeStack()
        for mode in CalendarViewMode.allCases {
            vm.setViewMode(mode)
            XCTAssertTrue(vm.visibleRange.contains(vm.selectedDate))
        }
    }

    func testDayRangeCoversExactlyOneDay() async {
        let (vm, _) = await makeStack()
        vm.setViewMode(.day)
        let range = vm.visibleRange
        XCTAssertEqual(
            range.upperBound.timeIntervalSince(range.lowerBound),
            86_400
        )
        XCTAssertTrue(calendar.isDate(range.lowerBound, inSameDayAs: referenceDay))
    }

    func testWeekRangeCoversSevenDays() async {
        let (vm, _) = await makeStack()
        vm.setViewMode(.week)
        let days = CalendarGridModel.daysOfWeek(containing: referenceDay, calendar: calendar)
        XCTAssertEqual(days.count, 7)
        let range = vm.visibleRange
        XCTAssertEqual(range.lowerBound, days.first)
        XCTAssertEqual(range.upperBound, calendar.date(byAdding: .day, value: 1, to: days.last!))
    }

    // MARK: - Loading + selection

    func testLoadEventsFiltersToVisibleRange() async {
        let inside = CalendarFixtures.event(title: "Inside", startAt: day(1))
        let outside = CalendarFixtures.event(title: "Outside", startAt: day(30))
        let (vm, _) = await makeStack(events: [inside, outside])
        vm.setViewMode(.week)

        await vm.loadEvents()

        XCTAssertEqual(vm.events.map(\.title), ["Inside"])
        XCTAssertNil(vm.errorMessage)
    }

    func testSelectionDropsWhenEventLeavesRange() async {
        // Seed the event ON the selected day so the .day
        // lens loads it first; then refocus far away.
        let inside = CalendarFixtures.event(title: "Inside", startAt: day(0, hour: 12))
        let (vm, _) = await makeStack(events: [inside])
        vm.setViewMode(.day)
        await vm.loadEvents()
        vm.select(eventID: inside.id)
        XCTAssertEqual(vm.selectedEvent?.id, inside.id)

        // Refocus far away; the selection falls out.
        vm.selectedDate = day(60)
        await vm.loadEvents()
        XCTAssertNil(vm.selectedEventID)
    }

    func testOccurrencesExpandRecurrenceInRange() async {
        let weekly = CalendarFixtures.event(
            title: "Standup",
            startAt: day(0),
            recurrence: CalendarEvent.Recurrence(rrule: "FREQ=WEEKLY")
        )
        let (vm, _) = await makeStack(events: [weekly])
        vm.setViewMode(.month)
        await vm.loadEvents()

        XCTAssertEqual(vm.events.count, 1)
        XCTAssertGreaterThan(vm.occurrences(of: vm.events[0]).count, 1)
    }

    // MARK: - Navigation

    func testStepAdvancesByModeUnit() async {
        let (vm, _) = await makeStack()

        vm.setViewMode(.day)
        vm.step(1)
        XCTAssertTrue(calendar.isDate(vm.selectedDate, inSameDayAs: day(1)))

        vm.setViewMode(.week)
        vm.step(1)
        XCTAssertTrue(calendar.isDate(vm.selectedDate, inSameDayAs: day(8)))

        vm.setViewMode(.month)
        vm.step(-1)
        let backOneMonth = calendar.date(byAdding: .month, value: -1, to: day(8))!
        XCTAssertTrue(calendar.isDate(vm.selectedDate, inSameDayAs: backOneMonth))
    }

    func testGoToToday() async {
        let (vm, _) = await makeStack(initialDate: day(30))
        vm.goToToday()
        XCTAssertTrue(calendar.isDateInToday(vm.selectedDate))
    }

    func testFocusDaySwitchesToDayMode() async {
        let (vm, _) = await makeStack()
        let target = day(5)
        vm.focus(day: target)
        XCTAssertEqual(vm.viewMode, .day)
        XCTAssertTrue(calendar.isDate(vm.selectedDate, inSameDayAs: target))
    }

    func testOpenEventRefocusesAndSelects() async {
        let farAway = CalendarFixtures.event(title: "Offsite", startAt: day(45, hour: 14))
        let (vm, _) = await makeStack(events: [farAway])

        await vm.openEvent(id: farAway.id)

        XCTAssertTrue(calendar.isDate(vm.selectedDate, inSameDayAs: farAway.startAt))
        XCTAssertEqual(vm.selectedEventID, farAway.id)
        XCTAssertTrue(vm.events.contains(where: { $0.id == farAway.id }))
    }

    func testOpenEventWithUnknownIDIsANoOp() async {
        let (vm, _) = await makeStack()
        let before = vm.selectedDate
        await vm.openEvent(id: UUID())
        XCTAssertEqual(vm.selectedDate, before)
        XCTAssertNil(vm.selectedEventID)
    }

    // MARK: - Quick add

    func testQuickAddCreatesSelectsAndClearsText() async {
        let (vm, store) = await makeStack()
        vm.setViewMode(.week)
        vm.quickAddText = "Coffee with John"

        await vm.submitQuickAdd()

        XCTAssertEqual(vm.quickAddText, "")
        XCTAssertEqual(vm.lastChatOutcome?.kind, .created)
        XCTAssertNotNil(vm.selectedEventID)

        let stored = await store.events
        XCTAssertEqual(stored.count, 1)
        let event = stored.values.first!
        // The pinned "now" drives the default window:
        // reference day at 09:00.
        XCTAssertTrue(calendar.isDate(event.startAt, inSameDayAs: referenceDay))
        XCTAssertEqual(calendar.component(.hour, from: event.startAt), 9)
        XCTAssertEqual(vm.selectedEvent?.id, event.id)
        XCTAssertEqual(vm.selectedEvent?.attendees.first?.name, "John")
    }

    func testQuickAddEmptyTextIsANoOp() async {
        let (vm, store) = await makeStack()
        vm.quickAddText = "   "
        await vm.submitQuickAdd()
        let stored = await store.events
        XCTAssertTrue(stored.isEmpty)
        XCTAssertNil(vm.lastChatOutcome)
    }

    func testQuickAddDeleteRemovesEvent() async {
        let event = CalendarFixtures.event(title: "Standup", startAt: day(1))
        let (vm, store) = await makeStack(events: [event])
        vm.setViewMode(.week)
        vm.quickAddText = "cancel the standup"

        await vm.submitQuickAdd()

        XCTAssertEqual(vm.lastChatOutcome?.kind, .deleted)
        let stored = await store.events
        XCTAssertTrue(stored.isEmpty)
    }

    // MARK: - Mutations

    func testDeleteSelected() async {
        let event = CalendarFixtures.event(title: "Doomed", startAt: day(1))
        let (vm, store) = await makeStack(events: [event])
        vm.setViewMode(.week)
        await vm.loadEvents()
        vm.select(eventID: event.id)

        await vm.deleteSelected()

        let stored = await store.events
        XCTAssertTrue(stored.isEmpty)
        XCTAssertNil(vm.selectedEventID)
        XCTAssertTrue(vm.events.isEmpty)
    }

    func testRespondSetsAttendeeStatus() async {
        let event = CalendarFixtures.event(
            title: "Invited",
            startAt: day(1),
            attendees: [CalendarEvent.Attendee(name: "Me")]
        )
        let (vm, store) = await makeStack(events: [event])
        vm.setViewMode(.week)
        await vm.loadEvents()
        vm.select(eventID: event.id)

        await vm.respond(to: .accepted)

        let stored = await store.events[event.id]
        XCTAssertEqual(stored?.attendees.first?.responseStatus, .accepted)
    }

    // MARK: - CalendarGridModel (pure grid math)

    func testHourSlotsCoverTheDay() {
        let slots = CalendarGridModel.hourSlots(for: referenceDay, calendar: calendar)
        XCTAssertEqual(slots.count, 24)
        XCTAssertEqual(slots.first, calendar.startOfDay(for: referenceDay))
        XCTAssertEqual(calendar.component(.hour, from: slots.last!), 23)
    }

    func testDaysOfWeekStartOnFirstWeekday() {
        let days = CalendarGridModel.daysOfWeek(containing: referenceDay, calendar: calendar)
        XCTAssertEqual(days.count, 7)
        XCTAssertEqual(
            calendar.component(.weekday, from: days.first!),
            calendar.firstWeekday
        )
        XCTAssertTrue(days.contains(where: { calendar.isDate($0, inSameDayAs: referenceDay) }))
    }

    func testMonthGridIsRectangularAndCoversTheMonth() {
        let weeks = CalendarGridModel.monthGrid(for: referenceDay, calendar: calendar)
        XCTAssertFalse(weeks.isEmpty)
        for week in weeks {
            XCTAssertEqual(week.count, 7)
        }
        // The grid pads to whole weeks, so it starts on or
        // before the 1st and includes the last day of the
        // month.
        let monthInterval = calendar.dateInterval(of: .month, for: referenceDay)!
        XCTAssertLessThanOrEqual(weeks.first!.first!, monthInterval.start)
        let lastOfMonth = calendar.date(byAdding: .day, value: -1, to: monthInterval.end)!
        let allDays = weeks.flatMap { $0 }
        XCTAssertTrue(allDays.contains(where: { calendar.isDate($0, inSameDayAs: lastOfMonth) }))
        // Every day of the month appears exactly once.
        let monthDays = allDays.filter { calendar.isDate($0, equalTo: referenceDay, toGranularity: .month) }
        XCTAssertEqual(monthDays.count, calendar.range(of: .day, in: .month, for: referenceDay)!.count)
    }

    func testDayFraction() {
        let start = calendar.startOfDay(for: referenceDay)
        XCTAssertEqual(CalendarGridModel.dayFraction(of: start, calendar: calendar), 0)
        let noon = calendar.date(byAdding: .hour, value: 12, to: start)!
        XCTAssertEqual(CalendarGridModel.dayFraction(of: noon, calendar: calendar), 0.5, accuracy: 0.001)
    }

    // MARK: - Graph connector

    func testGraphConnectorOpensCalendarEventNode() async throws {
        let event = CalendarFixtures.event(title: "From graph", startAt: day(2, hour: 14))
        let (vm, _) = await makeStack(events: [event])

        // Construction only - the data layer is never
        // started, so no connection is attempted.
        let dataLayer = TesseraDataLayer(configuration: .init())
        let graph = GraphViewModel(store: GraphStore(dataLayer: dataLayer))
        XCTAssertNil(graph.openEntityHandler)

        CalendarGraphConnector.wire(graph, to: vm)
        XCTAssertNotNil(graph.openEntityHandler)

        let node = GraphNode(
            id: event.id,
            entityType: CalendarEvent.entityType,
            label: event.title,
            importance: 0.5,
            updatedAt: Date()
        )
        graph.open(node)

        // The connector hops through a task; poll for it.
        for _ in 0..<100 {
            if vm.selectedEventID == event.id { break }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertEqual(vm.selectedEventID, event.id)
        XCTAssertTrue(calendar.isDate(vm.selectedDate, inSameDayAs: event.startAt))
    }

    func testGraphConnectorIgnoresNonCalendarNodes() async throws {
        let (vm, _) = await makeStack()
        let dataLayer = TesseraDataLayer(configuration: .init())
        let graph = GraphViewModel(store: GraphStore(dataLayer: dataLayer))
        CalendarGraphConnector.wire(graph, to: vm)

        let node = GraphNode(
            id: UUID(),
            entityType: "document",
            label: "A doc",
            importance: 0.5,
            updatedAt: Date()
        )
        graph.open(node)

        try await Task.sleep(nanoseconds: 50_000_000)
        XCTAssertNil(vm.selectedEventID)
    }
}
