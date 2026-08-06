import SwiftUI

// MARK: - Event color

/// Stable per-event color derived from the event id, so an
/// event keeps its color across day / week / month views
/// without persisting a color field.
public enum CalendarEventColor {
    public static let palette: [Color] = [
        .blue, .purple, .teal, .indigo, .orange, .pink, .mint, .cyan,
    ]

    public static func color(for eventID: UUID) -> Color {
        let bytes = eventID.uuid.0 ^ eventID.uuid.1 ^ eventID.uuid.2 ^ eventID.uuid.3
        return palette[Int(bytes) % palette.count]
    }
}

// MARK: - Day view

/// Hour-by-hour layout for a single day: an all-day lane
/// on top, then a 24-slot grid with timed events placed as
/// blocks at their hour offset.
public struct CalendarDayView: View {

    public let date: Date
    public let events: [CalendarEvent]
    public let calendar: Calendar
    @Binding public var selectedEventID: UUID?
    public let onSelect: (UUID) -> Void

    public init(
        date: Date,
        events: [CalendarEvent],
        calendar: Calendar = .current,
        selectedEventID: Binding<UUID?>,
        onSelect: @escaping (UUID) -> Void
    ) {
        self.date = date
        self.events = events
        self.calendar = calendar
        self._selectedEventID = selectedEventID
        self.onSelect = onSelect
    }

    /// Events that occur on this day (all-day lane uses
    /// the event itself; the hour grid uses occurrences).
    private var todaysEvents: [CalendarEvent] {
        events.filter { $0.occurs(on: date, calendar: calendar) }
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                allDayLane
                hourGrid
            }
            .padding(.horizontal, 8)
        }
        .accessibilityLabel("Day view for \(date.formatted(date: .abbreviated, time: .omitted))")
    }

    private var allDayLane: some View {
        let allDay = todaysEvents.filter(\.allDay)
        return VStack(alignment: .leading, spacing: 4) {
            if !allDay.isEmpty {
                Text("All day")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                ForEach(allDay) { event in
                    CalendarEventChip(
                        event: event,
                        isSelected: selectedEventID == event.id
                    )
                    .onTapGesture { onSelect(event.id) }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.vertical, 6)
    }

    private var hourGrid: some View {
        let slots = CalendarGridModel.hourSlots(for: date, calendar: calendar)
        let timed = todaysEvents.filter { !$0.allDay }
        return LazyVStack(spacing: 0) {
            ForEach(Array(slots.enumerated()), id: \.offset) { _, slot in
                HStack(alignment: .top, spacing: 8) {
                    Text(slot, format: .dateTime.hour())
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .frame(width: 44, alignment: .trailing)
                    ZStack(alignment: .topLeading) {
                        Rectangle()
                            .fill(.quaternary)
                            .frame(height: 1)
                        ForEach(timed.filter { occursInHour($0, slot) }) { event in
                            CalendarEventBlock(
                                event: event,
                                isSelected: selectedEventID == event.id
                            )
                            .onTapGesture { onSelect(event.id) }
                        }
                    }
                    .frame(maxWidth: .infinity, minHeight: 36, alignment: .topLeading)
                }
                .padding(.vertical, 2)
            }
        }
    }

    private func occursInHour(_ event: CalendarEvent, _ slot: Date) -> Bool {
        event.occurrences(in: slot...slot.addingTimeInterval(3599), calendar: calendar)
            .contains { calendar.isDate($0, equalTo: slot, toGranularity: .hour) }
    }
}

// MARK: - Week view

/// 7-day grid: one column per day, timed events as
/// colored blocks positioned by hour fraction, all-day
/// events stacked in a top lane.
public struct CalendarWeekView: View {

    public let date: Date
    public let events: [CalendarEvent]
    public let calendar: Calendar
    @Binding public var selectedEventID: UUID?
    public let onSelect: (UUID) -> Void
    public let onSelectDay: (Date) -> Void

    public init(
        date: Date,
        events: [CalendarEvent],
        calendar: Calendar = .current,
        selectedEventID: Binding<UUID?>,
        onSelect: @escaping (UUID) -> Void,
        onSelectDay: @escaping (Date) -> Void = { _ in }
    ) {
        self.date = date
        self.events = events
        self.calendar = calendar
        self._selectedEventID = selectedEventID
        self.onSelect = onSelect
        self.onSelectDay = onSelectDay
    }

    private var days: [Date] {
        CalendarGridModel.daysOfWeek(containing: date, calendar: calendar)
    }

    public var body: some View {
        VStack(spacing: 0) {
            headerRow
            Divider()
            ScrollView {
                HStack(alignment: .top, spacing: 4) {
                    ForEach(days, id: \.self) { day in
                        dayColumn(day)
                    }
                }
                .padding(4)
            }
        }
        .accessibilityLabel("Week view containing \(date.formatted(date: .abbreviated, time: .omitted))")
    }

    private var headerRow: some View {
        HStack(spacing: 4) {
            ForEach(days, id: \.self) { day in
                VStack(spacing: 2) {
                    Text(day, format: .dateTime.weekday(.abbreviated))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text(day, format: .dateTime.day())
                        .font(.callout.weight(calendar.isDateInToday(day) ? .bold : .regular))
                }
                .frame(maxWidth: .infinity)
                .contentShape(Rectangle())
                .onTapGesture { onSelectDay(day) }
            }
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 4)
    }

    private func dayColumn(_ day: Date) -> some View {
        let dayEvents = events.filter { $0.occurs(on: day, calendar: calendar) }
        return VStack(alignment: .leading, spacing: 3) {
            ForEach(dayEvents) { event in
                CalendarEventChip(
                    event: event,
                    isSelected: selectedEventID == event.id,
                    compact: true
                )
                .onTapGesture { onSelect(event.id) }
            }
            Spacer(minLength: 24)
        }
        .frame(maxWidth: .infinity, minHeight: 320, alignment: .topLeading)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(calendar.isDateInToday(day) ? Color.accentColor.opacity(0.06) : .clear)
        )
    }
}

// MARK: - Month view

/// Month grid: whole weeks as rows, events as small
/// markers (up to three per cell, then "+N"). Tapping a
/// day cell navigates to the day view (iOS) or focuses
/// the day (macOS).
public struct CalendarMonthView: View {

    public let date: Date
    public let events: [CalendarEvent]
    public let calendar: Calendar
    @Binding public var selectedEventID: UUID?
    public let onSelect: (UUID) -> Void
    public let onSelectDay: (Date) -> Void

    /// Max event markers per day cell before "+N".
    public static let maxMarkersPerCell = 3

    public init(
        date: Date,
        events: [CalendarEvent],
        calendar: Calendar = .current,
        selectedEventID: Binding<UUID?>,
        onSelect: @escaping (UUID) -> Void,
        onSelectDay: @escaping (Date) -> Void
    ) {
        self.date = date
        self.events = events
        self.calendar = calendar
        self._selectedEventID = selectedEventID
        self.onSelect = onSelect
        self.onSelectDay = onSelectDay
    }

    private var weeks: [[Date]] {
        CalendarGridModel.monthGrid(for: date, calendar: calendar)
    }

    public var body: some View {
        VStack(spacing: 4) {
            weekdayHeader
            ForEach(Array(weeks.enumerated()), id: \.offset) { _, week in
                HStack(spacing: 4) {
                    ForEach(week, id: \.self) { day in
                        dayCell(day)
                    }
                }
            }
        }
        .padding(4)
        .accessibilityLabel("Month view for \(date.formatted(.dateTime.month(.wide).year()))")
    }

    private var weekdayHeader: some View {
        let symbols = calendar.veryShortStandaloneWeekdaySymbols
        let first = calendar.firstWeekday - 1
        let ordered = Array(symbols[first...] + symbols[..<first])
        return HStack(spacing: 4) {
            ForEach(Array(ordered.enumerated()), id: \.offset) { _, s in
                Text(s)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity)
            }
        }
    }

    private func dayCell(_ day: Date) -> some View {
        let inMonth = calendar.isDate(day, equalTo: date, toGranularity: .month)
        let dayEvents = events.filter { $0.occurs(on: day, calendar: calendar) }
        let markers = Array(dayEvents.prefix(Self.maxMarkersPerCell))
        let overflow = dayEvents.count - markers.count
        return VStack(alignment: .leading, spacing: 2) {
            Text(day, format: .dateTime.day())
                .font(.caption)
                .fontWeight(calendar.isDateInToday(day) ? .bold : .regular)
                .foregroundStyle(inMonth ? .primary : .tertiary)
            ForEach(markers) { event in
                Circle()
                    .fill(CalendarEventColor.color(for: event.id))
                    .frame(width: 6, height: 6)
                    .accessibilityLabel(event.title)
                    .onTapGesture { onSelect(event.id) }
            }
            if overflow > 0 {
                Text("+\(overflow)")
                    .font(.system(size: 8))
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, minHeight: 56, alignment: .topLeading)
        .padding(3)
        .background(
            RoundedRectangle(cornerRadius: 4)
                .fill(selectedEventID != nil && dayEvents.contains(where: { $0.id == selectedEventID })
                      ? Color.accentColor.opacity(0.12)
                      : (calendar.isDateInToday(day) ? Color.accentColor.opacity(0.06) : .clear))
        )
        .contentShape(Rectangle())
        .onTapGesture { onSelectDay(day) }
    }
}

// MARK: - Event chips + blocks

/// A compact event pill (all-day lane, week columns).
struct CalendarEventChip: View {
    let event: CalendarEvent
    let isSelected: Bool
    var compact: Bool = false

    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(CalendarEventColor.color(for: event.id))
                .frame(width: compact ? 5 : 7, height: compact ? 5 : 7)
            Text(label)
                .font(compact ? .caption2 : .caption)
                .lineLimit(1)
            if event.recurrence != nil {
                Image(systemName: "repeat")
                    .font(.system(size: compact ? 7 : 9))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 5)
        .padding(.vertical, 2)
        .background(
            RoundedRectangle(cornerRadius: 4)
                .fill(CalendarEventColor.color(for: event.id).opacity(isSelected ? 0.35 : 0.15))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 4)
                .strokeBorder(isSelected ? CalendarEventColor.color(for: event.id) : .clear, lineWidth: 1)
        )
        .contentShape(Rectangle())
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(event.title), \(label)")
    }

    private var label: String {
        if event.allDay { return event.title }
        return "\(event.startAt.formatted(date: .omitted, time: .shortened)) \(event.title)"
    }
}

/// A timed event block for the day-view hour grid.
struct CalendarEventBlock: View {
    let event: CalendarEvent
    let isSelected: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(event.title)
                .font(.caption.weight(.medium))
                .lineLimit(1)
            Text("\(event.startAt.formatted(date: .omitted, time: .shortened)) - \(event.endAt.formatted(date: .omitted, time: .shortened))")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(5)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 5)
                .fill(CalendarEventColor.color(for: event.id).opacity(isSelected ? 0.4 : 0.2))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 5)
                .strokeBorder(isSelected ? CalendarEventColor.color(for: event.id) : .clear, lineWidth: 1)
        )
        .contentShape(Rectangle())
        .accessibilityElement(children: .combine)
    }
}
