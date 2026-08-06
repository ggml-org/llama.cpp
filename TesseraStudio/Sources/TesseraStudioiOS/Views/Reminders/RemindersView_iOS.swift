#if os(iOS)
import SwiftUI
import UserNotifications
import TesseraCore

// MARK: - RemindersView_iOS

/// The iOS Reminders surface.
///
/// **Layout:** `NavigationStack` with a `TabView`-style
/// filter strip at the top (the four ``ReminderFilter``
/// cases) and a `List` of rows below. Tapping a row
/// pushes the ``ReminderDetailView_iOS``.
///
/// The iOS view mirrors the macOS view's data flow (the
/// same ``ReminderListViewModel`` drives both); the
/// differences are the touch-optimized controls and the
/// modal presentation of the detail (rather than the
/// three-pane split).
public struct RemindersView_iOS: View {

    public init(
        store: any ReminderStoring,
        scheduler: ReminderNotificationScheduler
    ) {
        self.store = store
        self.scheduler = scheduler
        self._viewModel = StateObject(
            wrappedValue: ReminderListViewModel(store: store)
        )
    }

    private let store: any ReminderStoring
    private let scheduler: ReminderNotificationScheduler
    @StateObject private var viewModel: ReminderListViewModel

    @State private var authorizationStatus: UNAuthorizationStatus = .notDetermined
    @State private var showNotificationsAlert = false

    public var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                filterStrip
                Divider()
                list
            }
            .navigationTitle("Reminders")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        Task { await viewModel.load() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
            .task {
                await viewModel.load()
                await refreshAuthorization()
            }
            .alert("Notifications disabled",
                   isPresented: $showNotificationsAlert) {
                Button("Open Settings") {
                    if let url = URL(string: UIApplication.openSettingsURLString) {
                        UIApplication.shared.open(url)
                    }
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Enable notifications in Settings to receive reminder alerts.")
            }
        }
    }

    // MARK: - Filter strip

    private var filterStrip: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 12) {
                ForEach(ReminderFilter.allCases) { filter in
                    Button {
                        viewModel.filter = filter
                    } label: {
                        VStack(spacing: 4) {
                            HStack(spacing: 6) {
                                Image(systemName: filter.systemImage)
                                Text(filter.displayName)
                                    .font(.subheadline)
                                    .fontWeight(.medium)
                            }
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(
                                viewModel.filter == filter
                                    ? Color.accentColor.opacity(0.2)
                                    : Color.gray.opacity(0.1),
                                in: Capsule()
                            )
                            Text("\(filter.apply(to: viewModel.reminders).count)")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 8)
        }
    }

    // MARK: - List

    private var list: some View {
        List(selection: $viewModel.selectedID) {
            ForEach(viewModel.filtered) { reminder in
                NavigationLink(value: reminder.id) {
                    ReminderRow_iOS(reminder: reminder, viewModel: viewModel)
                }
                .swipeActions(edge: .trailing) {
                    if !reminder.isAcknowledged() {
                        Button {
                            Task {
                                await viewModel.acknowledge(reminder)
                                await scheduler.cancel(reminder)
                            }
                        } label: {
                            Label("Ack", systemImage: "checkmark.circle")
                        }
                        .tint(.green)
                    }
                    Button(role: .destructive) {
                        Task {
                            await scheduler.cancel(reminder)
                            await viewModel.delete(reminder)
                        }
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                }
            }
        }
        .navigationDestination(for: UUID.self) { id in
            if let reminder = viewModel.reminders.first(where: { $0.id == id }) {
                ReminderDetailView_iOS(
                    reminder: reminder,
                    store: store,
                    scheduler: scheduler
                )
            }
        }
        .overlay {
            if viewModel.isLoading {
                ProgressView().controlSize(.large)
            } else if viewModel.filtered.isEmpty {
                ContentUnavailableView(
                    emptyTitle,
                    systemImage: emptyIcon,
                    description: Text(emptyDescription)
                )
            } else if let err = viewModel.loadError {
                ContentUnavailableView(
                    "Couldn't load reminders",
                    systemImage: "exclamationmark.triangle",
                    description: Text(err)
                )
            }
        }
    }

    // MARK: - Empty-state copy

    private var emptyTitle: String {
        switch viewModel.filter {
        case .upcoming: return "No upcoming reminders"
        case .acknowledged: return "No acknowledged reminders"
        case .snoozed: return "No snoozed reminders"
        case .all: return "No reminders"
        }
    }

    private var emptyIcon: String {
        switch viewModel.filter {
        case .upcoming: return "bell.slash"
        case .acknowledged: return "checkmark.circle"
        case .snoozed: return "moon.zzz"
        case .all: return "tray"
        }
    }

    private var emptyDescription: String {
        switch viewModel.filter {
        case .upcoming, .all:
            return "Use the chat panel to add one (\"remind me 15 min before the Q3 review\")."
        case .acknowledged:
            return "Acknowledged reminders show up here once you dismiss one."
        case .snoozed:
            return "Snooze an upcoming reminder to see it here."
        }
    }

    // MARK: - Authorization

    private func refreshAuthorization() async {
        authorizationStatus = await scheduler.authorizationStatus()
        if authorizationStatus == .denied {
            showNotificationsAlert = true
        }
    }
}

// MARK: - ReminderRow_iOS

private struct ReminderRow_iOS: View {
    let reminder: Reminder
    let viewModel: ReminderListViewModel

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: rowIcon)
                .font(.title3)
                .foregroundStyle(rowColor)
                .frame(width: 28)
                .padding(.top, 2)
            VStack(alignment: .leading, spacing: 2) {
                Text(reminder.title)
                    .font(.body)
                    .strikethrough(reminder.isAcknowledged())
                Text(reminder.displayLine())
                    .font(.caption)
                    .foregroundStyle(.secondary)
                HStack(spacing: 6) {
                    if !reminder.priority.shortLabel.isEmpty {
                        Text(reminder.priority.shortLabel)
                            .font(.caption2)
                            .fontWeight(.bold)
                            .foregroundStyle(priorityColor)
                    }
                    Text(timeDescription)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
            Spacer()
        }
        .padding(.vertical, 2)
    }

    private var rowIcon: String {
        if reminder.isAcknowledged() { return "checkmark.circle" }
        if reminder.isSnoozed() { return "moon.zzz" }
        return "bell"
    }

    private var rowColor: Color {
        if reminder.isAcknowledged() { return .green }
        if reminder.isSnoozed() { return .orange }
        return .yellow
    }

    private var priorityColor: Color {
        switch reminder.priority {
        case .high: return .red
        case .medium: return .orange
        case .low: return .secondary
        case .none: return .clear
        }
    }

    private var timeDescription: String {
        if reminder.isAcknowledged(),
           let ack = reminder.acknowledgedAt {
            return "Acknowledged \(viewModel.relativeTime(for: ack))"
        }
        if reminder.isSnoozed(),
           let snooze = reminder.snoozedUntil {
            return "Snoozed until \(viewModel.relativeTime(for: snooze))"
        }
        return viewModel.relativeTime(for: reminder.triggerAt)
    }
}
#endif
