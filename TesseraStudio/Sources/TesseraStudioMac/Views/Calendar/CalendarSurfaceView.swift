import SwiftUI
import TesseraCore

// MARK: - CalendarSurfaceView

/// The Phase 5 Calendar surface for macOS. Three-column
/// `NavigationSplitView`:
///
///   * **Sidebar** — date picker, day / week / month
///     selector, "Today" shortcut, and the upcoming events
///     list for the visible range.
///   * **Content** — the grid for the current mode (day /
///     week / month) plus the Fantastical-style quick-add
///     bar at the top.
///   * **Detail** — the selected event's detail pane
///     (attendees, location, recurrence, linked materials,
///     receipt history).
///
/// All state lives in ``CalendarViewModel`` (TesseraCore),
/// so the iOS surface (`CalendarMobileView`) and this
/// macOS surface share one model and one store.
public struct CalendarSurfaceView: View {

    @ObservedObject public var model: CalendarViewModel

    public init(model: CalendarViewModel) {
        self.model = model
    }

    public var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 200, ideal: 240)
        } content: {
            contentColumn
                .frame(minWidth: 420)
        } detail: {
            detailColumn
                .navigationSplitViewColumnWidth(min: 280, ideal: 320)
        }
        .navigationTitle("Calendar")
        .task { await model.loadEvents() }
    }

    // MARK: - Sidebar

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 12) {
            DatePicker(
                "Date",
                selection: Binding(
                    get: { model.selectedDate },
                    set: { newValue in
                        model.selectedDate = newValue
                        Task { await model.loadEvents() }
                    }
                ),
                displayedComponents: .date
            )
            .datePickerStyle(.graphical)

            Picker("View", selection: Binding(
                get: { model.viewMode },
                set: { model.setViewMode($0) }
            )) {
                ForEach(CalendarViewMode.allCases) { mode in
                    Label(mode.displayName, systemImage: mode.systemImage).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            Button("Today") { model.goToToday() }
                .buttonStyle(.bordered)
                .controlSize(.small)

            Divider()

            upcomingList
        }
        .padding(10)
    }

    private var upcomingList: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Upcoming")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            if model.events.isEmpty {
                Text("No events in this range.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 4) {
                        ForEach(model.events) { event in
                            Button {
                                model.select(eventID: event.id)
                            } label: {
                                HStack(spacing: 6) {
                                    Circle()
                                        .fill(CalendarEventColor.color(for: event.id))
                                        .frame(width: 7, height: 7)
                                    VStack(alignment: .leading, spacing: 0) {
                                        Text(event.title)
                                            .font(.callout)
                                            .lineLimit(1)
                                        Text(whenLabel(event))
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                    }
                                    Spacer()
                                }
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }
        }
    }

    private func whenLabel(_ event: CalendarEvent) -> String {
        if event.allDay {
            return event.startAt.formatted(date: .abbreviated, time: .omitted)
        }
        return event.startAt.formatted(date: .abbreviated, time: .shortened)
    }

    // MARK: - Content

    private var contentColumn: some View {
        VStack(spacing: 0) {
            quickAddBar
            Divider()
            toolbar
            Divider()
            grid
        }
        .overlay {
            if model.isLoading {
                ProgressView()
                    .controlSize(.large)
            }
        }
    }

    /// Fantastical-style natural language quick-add. The
    /// text goes through ``CalendarChatHandler``: it lands
    /// as a pending chat item, the handler classifies +
    /// executes it, and the created event becomes the
    /// selection.
    private var quickAddBar: some View {
        HStack(spacing: 8) {
            Image(systemName: "sparkles")
                .foregroundStyle(.secondary)
            TextField(
                "\"Lunch with John tomorrow at noon\", \"Q3 review next monday 2pm-4pm in the blue room\"...",
                text: $model.quickAddText
            )
            .textFieldStyle(.plain)
            .onSubmit {
                Task { await model.submitQuickAdd() }
            }
            Button("Add") {
                Task { await model.submitQuickAdd() }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(model.quickAddText.trimmingCharacters(in: .whitespaces).isEmpty)
        }
        .padding(8)
        .background(.bar)
    }

    private var toolbar: some View {
        HStack(spacing: 8) {
            Button {
                model.step(-1)
            } label: {
                Image(systemName: "chevron.left")
            }
            .buttonStyle(.borderless)

            Text(rangeTitle)
                .font(.headline)
                .frame(maxWidth: .infinity)

            Button {
                model.step(1)
            } label: {
                Image(systemName: "chevron.right")
            }
            .buttonStyle(.borderless)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
    }

    private var rangeTitle: String {
        switch model.viewMode {
        case .day:
            return model.selectedDate.formatted(date: .complete, time: .omitted)
        case .week:
            let days = CalendarGridModel.daysOfWeek(containing: model.selectedDate, calendar: .current)
            guard let first = days.first, let last = days.last else {
                return model.selectedDate.formatted(date: .abbreviated, time: .omitted)
            }
            return "\(first.formatted(date: .abbreviated, time: .omitted)) - \(last.formatted(date: .abbreviated, time: .omitted))"
        case .month:
            return model.selectedDate.formatted(.dateTime.month(.wide).year())
        }
    }

    @ViewBuilder
    private var grid: some View {
        switch model.viewMode {
        case .day:
            CalendarDayView(
                date: model.selectedDate,
                events: model.events,
                selectedEventID: Binding(
                    get: { model.selectedEventID },
                    set: { model.selectedEventID = $0 }
                ),
                onSelect: { model.select(eventID: $0) }
            )
        case .week:
            CalendarWeekView(
                date: model.selectedDate,
                events: model.events,
                selectedEventID: Binding(
                    get: { model.selectedEventID },
                    set: { model.selectedEventID = $0 }
                ),
                onSelect: { model.select(eventID: $0) },
                onSelectDay: { model.focus(day: $0) }
            )
        case .month:
            CalendarMonthView(
                date: model.selectedDate,
                events: model.events,
                selectedEventID: Binding(
                    get: { model.selectedEventID },
                    set: { model.selectedEventID = $0 }
                ),
                onSelect: { model.select(eventID: $0) },
                onSelectDay: { model.focus(day: $0) }
            )
        }
    }

    // MARK: - Detail

    @ViewBuilder
    private var detailColumn: some View {
        if let event = model.selectedEvent {
            CalendarEventDetailView(
                event: event,
                onRespond: { status in
                    Task { await model.respond(to: status) }
                },
                onDelete: {
                    Task { await model.deleteSelected() }
                },
                onClose: { model.select(eventID: nil) }
            )
        } else {
            ContentUnavailableView(
                "No event selected",
                systemImage: "calendar.badge.clock",
                description: Text("Select an event in the grid, or type a new one above.")
            )
        }
    }
}
