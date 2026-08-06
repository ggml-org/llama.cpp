import SwiftUI

// MARK: - CalendarEventDetailView

/// The event detail pane (macOS inspector column / iOS
/// pushed screen). Shows the event's fields plus its
/// cross-surface links (attendees -> contacts, prep docs,
/// prep tasks, reminders) and the receipt-history entry
/// point.
public struct CalendarEventDetailView: View {

    public let event: CalendarEvent
    public let receipts: [GraphReceipt]
    public let links: [EntityLink]
    public let onRespond: (CalendarEvent.ResponseStatus) -> Void
    public let onDelete: () -> Void
    public let onClose: () -> Void

    public init(
        event: CalendarEvent,
        receipts: [GraphReceipt] = [],
        links: [EntityLink] = [],
        onRespond: @escaping (CalendarEvent.ResponseStatus) -> Void = { _ in },
        onDelete: @escaping () -> Void = {},
        onClose: @escaping () -> Void = {}
    ) {
        self.event = event
        self.receipts = receipts
        self.links = links
        self.onRespond = onRespond
        self.onDelete = onDelete
        self.onClose = onClose
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                header
                timeSection
                if let location = event.location, !location.isEmpty {
                    locationSection(location)
                }
                if !event.attendees.isEmpty {
                    attendeesSection
                }
                if let recurrence = event.recurrence {
                    recurrenceSection(recurrence)
                }
                linkedSection
                receiptsSection
                dangerZone
            }
            .padding()
        }
        .navigationTitle(event.title)
    }

    // MARK: Sections

    private var header: some View {
        HStack(alignment: .top) {
            Circle()
                .fill(CalendarEventColor.color(for: event.id))
                .frame(width: 10, height: 10)
                .padding(.top, 5)
            Text(event.title)
                .font(.title3.weight(.semibold))
            Spacer()
        }
    }

    private var timeSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Label {
                if event.allDay {
                    Text(event.startAt.formatted(date: .complete, time: .omitted))
                } else {
                    Text("\(event.startAt.formatted(date: .abbreviated, time: .shortened)) - \(event.endAt.formatted(date: .omitted, time: .shortened))")
                }
            } icon: {
                Image(systemName: "clock")
            }
            if event.allDay {
                Label("All day", systemImage: "sun.max")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if !event.notes.isEmpty {
                Text(event.notes)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .padding(.top, 2)
            }
        }
        .font(.callout)
    }

    private func locationSection(_ location: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Label(location, systemImage: "mappin.and.ellipse")
                .font(.callout)
            if let c = event.locationCoordinate {
                Text(String(format: "%.4f, %.4f", c.latitude, c.longitude))
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .padding(.leading, 24)
            }
        }
    }

    private var attendeesSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Attendees")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            ForEach(Array(event.attendees.enumerated()), id: \.offset) { _, attendee in
                HStack(spacing: 6) {
                    Image(systemName: attendee.contactID != nil ? "person.crop.circle.fill" : "person.crop.circle")
                        .foregroundStyle(attendee.contactID != nil ? Color.accentColor : .secondary)
                    VStack(alignment: .leading, spacing: 0) {
                        Text(attendee.name)
                            .font(.callout)
                        if let email = attendee.email {
                            Text(email)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                    Spacer()
                    statusBadge(attendee.responseStatus)
                }
            }
            HStack(spacing: 8) {
                rsvpButton("Accept", .accepted)
                rsvpButton("Maybe", .tentative)
                rsvpButton("Decline", .declined)
            }
            .padding(.top, 2)
        }
    }

    private func rsvpButton(_ title: String, _ status: CalendarEvent.ResponseStatus) -> some View {
        Button(title) { onRespond(status) }
            .buttonStyle(.bordered)
            .controlSize(.small)
    }

    private func statusBadge(_ status: CalendarEvent.ResponseStatus) -> some View {
        let (label, color): (String, Color)
        switch status {
        case .accepted: (label, color) = ("Accepted", .green)
        case .declined: (label, color) = ("Declined", .red)
        case .tentative: (label, color) = ("Maybe", .orange)
        case .needsAction: (label, color) = ("Invited", .secondary)
        }
        return Text(label)
            .font(.caption2.weight(.medium))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(Capsule().fill(color.opacity(0.15)))
            .foregroundStyle(color)
    }

    private func recurrenceSection(_ recurrence: CalendarEvent.Recurrence) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Label(recurrenceSummary(recurrence), systemImage: "repeat")
                .font(.callout)
            if !recurrence.exDates.isEmpty {
                Text("\(recurrence.exDates.count) exception date\(recurrence.exDates.count == 1 ? "" : "s")")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .padding(.leading, 24)
            }
        }
    }

    private func recurrenceSummary(_ recurrence: CalendarEvent.Recurrence) -> String {
        if let rule = try? RecurrenceRule(rrule: recurrence.rrule) {
            switch rule.frequency {
            case .daily: return rule.interval > 1 ? "Every \(rule.interval) days" : "Every day"
            case .weekly:
                let days = rule.byDay.isEmpty ? "week" : rule.byDay.map(\.rawValue).joined(separator: ", ")
                return rule.interval > 1 ? "Every \(rule.interval) weeks (\(days))" : "Weekly (\(days))"
            case .monthly: return rule.interval > 1 ? "Every \(rule.interval) months" : "Monthly"
            case .yearly: return rule.interval > 1 ? "Every \(rule.interval) years" : "Yearly"
            }
        }
        return recurrence.rrule
    }

    /// Cross-surface links (Phase 1 entity_links). The
    /// detail pane lists them grouped by link type; the
    /// graph view draws the same rows as edges.
    private var linkedSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Linked")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            if links.isEmpty {
                Text("No linked materials yet.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            } else {
                ForEach(links, id: \.id) { link in
                    HStack(spacing: 6) {
                        Image(systemName: linkIcon(link.linkType))
                            .foregroundStyle(.secondary)
                        Text(linkTargetName(link))
                            .font(.callout)
                        Spacer()
                        Text(link.linkType)
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                }
            }
            if !event.linkedDocumentIDs.isEmpty || !event.linkedTaskIDs.isEmpty || !event.reminders.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    if !event.linkedDocumentIDs.isEmpty {
                        Label("\(event.linkedDocumentIDs.count) prep document\(event.linkedDocumentIDs.count == 1 ? "" : "s")", systemImage: "doc.text")
                    }
                    if !event.linkedTaskIDs.isEmpty {
                        Label("\(event.linkedTaskIDs.count) prep task\(event.linkedTaskIDs.count == 1 ? "" : "s")", systemImage: "checkmark.square")
                    }
                    if !event.reminders.isEmpty {
                        Label("\(event.reminders.count) reminder\(event.reminders.count == 1 ? "" : "s")", systemImage: "bell")
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
    }

    private func linkIcon(_ linkType: String) -> String {
        switch linkType {
        case CalendarLinkType.attendeeOf.rawValue: return "person.crop.circle"
        case CalendarLinkType.prepDocument.rawValue: return "doc.text"
        case CalendarLinkType.prepTask.rawValue: return "checkmark.square"
        case CalendarLinkType.reminderFor.rawValue: return "bell"
        default: return "link"
        }
    }

    private func linkTargetName(_ link: EntityLink) -> String {
        // The link row carries only the target id; the
        // graph view resolves labels at load time. The
        // detail pane shows the id prefix so the row is
        // still actionable without a second query.
        String(link.targetID.uuidString.prefix(8)) + "..."
    }

    /// The receipt chain for this event (created ->
    /// updated* -> responded* -> deleted). Tapping a row
    /// opens the receipt drawer on macOS.
    private var receiptsSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("History")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            if receipts.isEmpty {
                Text("No receipts yet.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            } else {
                ForEach(receipts) { receipt in
                    HStack(spacing: 6) {
                        Image(systemName: receiptIcon(receipt.receiptType))
                            .foregroundStyle(.secondary)
                        Text(receipt.receiptType.replacingOccurrences(of: "_", with: " "))
                            .font(.callout)
                        Spacer()
                        Text(receipt.witnessedAt.formatted(date: .abbreviated, time: .shortened))
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                }
            }
        }
    }

    private func receiptIcon(_ type: String) -> String {
        switch type {
        case CalendarEventReceiptType.eventCreated.rawValue: return "plus.circle"
        case CalendarEventReceiptType.eventUpdated.rawValue: return "pencil"
        case CalendarEventReceiptType.eventDeleted.rawValue: return "trash"
        case CalendarEventReceiptType.eventResponded.rawValue: return "hand.raised"
        default: return "doc.on.doc"
        }
    }

    private var dangerZone: some View {
        HStack {
            Spacer()
            Button(role: .destructive) {
                onDelete()
            } label: {
                Label("Delete event", systemImage: "trash")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
        .padding(.top, 8)
    }
}

// MARK: - CalendarMobileView

/// The iOS calendar surface: date picker at the top,
/// mode selector, the grid for the current mode, tap to
/// drill into a day / event. Shared code, so it also
/// renders on macOS when embedded (the Mac surface uses
/// ``CalendarSurfaceView`` instead).
public struct CalendarMobileView: View {

    @ObservedObject public var model: CalendarViewModel

    public init(model: CalendarViewModel) {
        self.model = model
    }

    public var body: some View {
        NavigationStack {
            VStack(spacing: 8) {
                controls
                grid
                if let outcome = model.lastChatOutcome {
                    outcomeBanner(outcome)
                }
            }
            .navigationTitle("Calendar")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button("Today") { model.goToToday() }
                }
            }
            .safeAreaInset(edge: .bottom) {
                quickAddBar
            }
            .navigationDestination(item: Binding(
                get: { model.selectedEventID },
                set: { model.selectedEventID = $0 }
            )) { eventID in
                if let event = model.events.first(where: { $0.id == eventID }) {
                    CalendarEventDetailView(
                        event: event,
                        onRespond: { status in
                            model.select(eventID: eventID)
                            Task { await model.respond(to: status) }
                        },
                        onDelete: {
                            Task {
                                model.select(eventID: eventID)
                                await model.deleteSelected()
                            }
                        },
                        onClose: { model.select(eventID: nil) }
                    )
                }
            }
        }
        .task { await model.loadEvents() }
    }

    private var controls: some View {
        VStack(spacing: 8) {
            HStack {
                Button {
                    model.step(-1)
                } label: {
                    Image(systemName: "chevron.left")
                }
                Spacer()
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
                .labelsHidden()
                Spacer()
                Button {
                    model.step(1)
                } label: {
                    Image(systemName: "chevron.right")
                }
            }
            Picker("View", selection: Binding(
                get: { model.viewMode },
                set: { model.setViewMode($0) }
            )) {
                ForEach(CalendarViewMode.allCases) { mode in
                    Text(mode.displayName).tag(mode)
                }
            }
            .pickerStyle(.segmented)
        }
        .padding(.horizontal)
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

    private var quickAddBar: some View {
        HStack(spacing: 8) {
            TextField(
                "Lunch with John tomorrow at noon...",
                text: $model.quickAddText
            )
            .textFieldStyle(.roundedBorder)
            .onSubmit {
                Task { await model.submitQuickAdd() }
            }
            Button {
                Task { await model.submitQuickAdd() }
            } label: {
                Image(systemName: "plus.circle.fill")
                    .font(.title2)
            }
            .disabled(model.quickAddText.trimmingCharacters(in: .whitespaces).isEmpty)
        }
        .padding(.horizontal)
        .padding(.vertical, 6)
        .background(.bar)
    }

    private func outcomeBanner(_ outcome: CalendarChatOutcome) -> some View {
        Text(outcome.summary)
            .font(.caption)
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(outcome.kind == .failed ? Color.red.opacity(0.12) : Color.green.opacity(0.10))
            )
            .padding(.horizontal)
    }
}
