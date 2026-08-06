import Foundation
import SwiftUI

// MARK: - CalendarViewMode

/// The three grid layouts of the calendar surface. The
/// mode is a pure display choice: switching modes NEVER
/// moves `selectedDate` (the spec's "switching between
/// views preserves the date" requirement — the date is the
/// user's context, the mode is just the lens).
public enum CalendarViewMode: String, Codable, Sendable, CaseIterable, Identifiable {
    case day
    case week
    case month

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .day: return "Day"
        case .week: return "Week"
        case .month: return "Month"
        }
    }

    public var systemImage: String {
        switch self {
        case .day: return "rectangle.split.1x3"
        case .week: return "calendar"
        case .month: return "calendar.badge.clock"
        }
    }
}

// MARK: - CalendarGridModel

/// Pure date-grid math shared by the day / week / month
/// views. Kept separate from the views so the grid layout
/// is unit-testable without rendering anything.
public enum CalendarGridModel {

    /// The 24 hour-slot starts (00:00 ... 23:00) for a day.
    public static func hourSlots(for day: Date, calendar: Calendar) -> [Date] {
        let start = calendar.startOfDay(for: day)
        return (0..<24).compactMap { calendar.date(byAdding: .hour, value: $0, to: start) }
    }

    /// The 7 day-starts of the week containing `date`, in
    /// the calendar's first-weekday order.
    public static func daysOfWeek(containing date: Date, calendar: Calendar) -> [Date] {
        guard let interval = calendar.dateInterval(of: .weekOfYear, for: date) else {
            return [calendar.startOfDay(for: date)]
        }
        return (0..<7).compactMap { calendar.date(byAdding: .day, value: $0, to: interval.start) }
    }

    /// The month grid: whole weeks covering the month
    /// containing `date` (leading days from the previous
    /// month and trailing days from the next month are
    /// included so the grid is rectangular, like macOS
    /// Calendar). Each inner array is one week.
    public static func monthGrid(for date: Date, calendar: Calendar) -> [[Date]] {
        guard let monthInterval = calendar.dateInterval(of: .month, for: date),
              let gridStart = calendar.dateInterval(of: .weekOfYear, for: monthInterval.start)?.start,
              let gridEnd = calendar.dateInterval(of: .weekOfYear, for: monthInterval.end - 1)?.end
        else {
            return []
        }
        var weeks: [[Date]] = []
        var cursor = gridStart
        while cursor < gridEnd {
            let week = (0..<7).compactMap { calendar.date(byAdding: .day, value: $0, to: cursor) }
            weeks.append(week)
            guard let next = calendar.date(byAdding: .day, value: 7, to: cursor) else { break }
            cursor = next
        }
        return weeks
    }

    /// Fraction-of-day offset (0..1) for an occurrence,
    /// used to position a timed event block in the hour
    /// grid. All-day events are rendered in a separate
    /// lane and never call this.
    public static func dayFraction(of occurrence: Date, calendar: Calendar) -> Double {
        let day = calendar.startOfDay(for: occurrence)
        let seconds = occurrence.timeIntervalSince(day)
        return min(max(seconds / 86_400.0, 0), 1)
    }
}

// MARK: - CalendarViewModel

/// The calendar surface's shared view model (macOS +
/// iOS). Owns the selected date, the view mode, the
/// visible events, the selection, and the quick-add / chat
/// path.
///
/// The store dependency is the ``CalendarStoring``
/// protocol so the view model is testable with an
/// in-memory fake; production wires ``CalendarStore``.
@MainActor
public final class CalendarViewModel: ObservableObject {

    // MARK: Published state

    /// The date the surface is focused on. Preserved
    /// across mode switches.
    @Published public var selectedDate: Date
    /// Day / week / month lens.
    @Published public var viewMode: CalendarViewMode
    /// Events visible in the current range (expanded for
    /// recurrence).
    @Published public private(set) var events: [CalendarEvent] = []
    /// The event shown in the detail pane / sheet. nil =
    /// no selection.
    @Published public var selectedEventID: UUID?
    /// Quick-add / chat input text.
    @Published public var quickAddText: String = ""
    /// The most recent chat outcome (the panel renders it
    /// inline).
    @Published public private(set) var lastChatOutcome: CalendarChatOutcome?
    @Published public private(set) var isLoading = false
    @Published public var errorMessage: String?

    // MARK: Dependencies

    private let store: CalendarStoring
    private let chatHandler: CalendarChatHandler
    private let calendar: Calendar

    public init(
        store: CalendarStoring,
        chatHandler: CalendarChatHandler,
        calendar: Calendar = .current,
        initialDate: Date = Date(),
        viewMode: CalendarViewMode = .week
    ) {
        self.store = store
        self.chatHandler = chatHandler
        self.calendar = calendar
        self.selectedDate = initialDate
        self.viewMode = viewMode
    }

    // MARK: - Derived state

    /// The date range the current mode + selected date
    /// cover. Day = one day, week = the containing week,
    /// month = the containing month (the grid pads to
    /// whole weeks, but event filtering uses the month
    /// itself plus the pad days).
    public var visibleRange: ClosedRange<Date> {
        switch viewMode {
        case .day:
            let start = calendar.startOfDay(for: selectedDate)
            let end = calendar.date(byAdding: .day, value: 1, to: start) ?? start
            return start...end
        case .week:
            let days = CalendarGridModel.daysOfWeek(containing: selectedDate, calendar: calendar)
            let start = days.first ?? calendar.startOfDay(for: selectedDate)
            let end = calendar.date(byAdding: .day, value: 1, to: days.last ?? start) ?? start
            return start...end
        case .month:
            let weeks = CalendarGridModel.monthGrid(for: selectedDate, calendar: calendar)
            let start = weeks.first?.first ?? calendar.startOfDay(for: selectedDate)
            let lastDay = weeks.last?.last ?? start
            let end = calendar.date(byAdding: .day, value: 1, to: lastDay) ?? start
            return start...end
        }
    }

    /// The event open in the detail pane, if any.
    public var selectedEvent: CalendarEvent? {
        guard let id = selectedEventID else { return nil }
        return events.first(where: { $0.id == id })
    }

    /// Occurrences of `event` inside the visible range
    /// (the grids render one block per occurrence).
    public func occurrences(of event: CalendarEvent) -> [Date] {
        event.occurrences(in: visibleRange, calendar: calendar)
    }

    // MARK: - Mode + navigation

    /// Switch the lens. The selected date is preserved —
    /// switching week -> day keeps the same day, month ->
    /// week keeps the week containing the date.
    public func setViewMode(_ mode: CalendarViewMode) {
        viewMode = mode
    }

    /// Move one unit (day / week / month, per the current
    /// mode) forward (+1) or back (-1).
    public func step(_ direction: Int) {
        let component: Calendar.Component
        switch viewMode {
        case .day: component = .day
        case .week: component = .weekOfYear
        case .month: component = .month
        }
        if let next = calendar.date(byAdding: component, value: direction, to: selectedDate) {
            selectedDate = next
            Task { await loadEvents() }
        }
    }

    /// Jump to today (the toolbar's "Today" button).
    public func goToToday() {
        selectedDate = calendar.startOfDay(for: Date())
        Task { await loadEvents() }
    }

    /// Focus a specific day (month grid tap on iOS).
    /// Switches to the day lens, preserving the tapped
    /// date.
    public func focus(day: Date) {
        selectedDate = day
        viewMode = .day
        Task { await loadEvents() }
    }

    // MARK: - Selection

    /// Open an event in the detail pane (grid tap / graph
    /// node click both route here).
    public func select(eventID: UUID?) {
        selectedEventID = eventID
    }

    /// Open an event by id from another surface (the
    /// graph view routes event node opens here). Refocuses
    /// the surface onto the event's day, loads, and
    /// selects the event.
    public func openEvent(id: UUID) async {
        guard let event = try? await store.get(id: id) else { return }
        selectedDate = event.startAt
        await loadEvents()
        selectedEventID = id
    }

    // MARK: - Loading

    public func loadEvents() async {
        isLoading = true
        defer { isLoading = false }
        do {
            events = try await store.events(in: visibleRange, calendar: calendar)
            // Drop a selection that fell out of the range.
            if let id = selectedEventID, !events.contains(where: { $0.id == id }) {
                selectedEventID = nil
            }
        } catch {
            errorMessage = "Couldn't load events: \(error)"
        }
    }

    // MARK: - Quick add / chat

    /// Submit the quick-add text through the chat handler.
    /// The created event lands in `events` and becomes the
    /// selection (the surface jumps to what the user just
    /// typed, Fantastical-style).
    public func submitQuickAdd() async {
        let text = quickAddText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        quickAddText = ""
        do {
            let outcome = try await chatHandler.submit(text)
            lastChatOutcome = outcome
            if let eventID = outcome.eventID, outcome.kind != .deleted {
                selectedDate = (try? await store.get(id: eventID))?.startAt ?? selectedDate
                await loadEvents()
                selectedEventID = eventID
            } else {
                await loadEvents()
            }
        } catch {
            errorMessage = "Quick add failed: \(error)"
        }
    }

    // MARK: - Mutations (all receipt-bearing via the store)

    /// Delete the selected event.
    public func deleteSelected() async {
        guard let id = selectedEventID else { return }
        do {
            _ = try await store.delete(id: id)
            selectedEventID = nil
            await loadEvents()
        } catch {
            errorMessage = "Delete failed: \(error)"
        }
    }

    /// RSVP the selected event (first attendee = the
    /// user).
    public func respond(to status: CalendarEvent.ResponseStatus) async {
        guard let id = selectedEventID else { return }
        do {
            _ = try await store.respond(
                to: id,
                attendeeIndex: 0,
                attendeeName: nil,
                status: status
            )
            await loadEvents()
        } catch {
            errorMessage = "Respond failed: \(error)"
        }
    }
}
