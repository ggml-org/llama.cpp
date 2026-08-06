#if os(macOS)
import AppKit
import SwiftUI
import UserNotifications
import TesseraCore

// MARK: - RemindersView (macOS)

/// The macOS Reminders surface.
///
/// **Layout:** `NavigationSplitView` with a sidebar (the
/// four ``ReminderFilter`` cases), a list (the filtered
/// reminder rows), and a detail pane (the selected
/// reminder's metadata + linked entities + receipt chain).
///
/// **Data:** the view reads from ``ReminderStore`` (which
/// wraps ``TesseraDataLayer``). Mutations go through the
/// same store so every change is a constitutional receipt.
/// Notifications are scheduled via
/// ``ReminderNotificationScheduler``; the view wires the
/// store + scheduler together so an acknowledge call
/// cancels the pending notification, a snooze call
/// reschedules it, and a delete call cancels it.
public struct RemindersView: View {

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

    public var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 200, ideal: 220)
        } content: {
            list
        } detail: {
            if let id = viewModel.selectedID,
               let reminder = viewModel.reminders.first(where: { $0.id == id }) {
                ReminderDetailView(
                    reminder: reminder,
                    store: store,
                    scheduler: scheduler
                )
            } else {
                emptyState
            }
        }
        .navigationTitle("Reminders")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    Task {
                        await viewModel.load()
                        await refreshAuthorization()
                    }
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .help("Reload reminders")
            }
        }
        .overlay(alignment: .bottom) {
            if authorizationStatus == .denied {
                notificationsDisabledBanner
            }
        }
        .task {
            await viewModel.load()
            await refreshAuthorization()
        }
    }

    // MARK: - Sidebar

    private var sidebar: some View {
        List(selection: $viewModel.filter) {
            Section("Reminders") {
                ForEach(ReminderFilter.allCases) { filter in
                    Label(filter.displayName, systemImage: filter.systemImage)
                        .badge(badgeCount(for: filter))
                        .tag(filter)
                }
            }
        }
        .listStyle(.sidebar)
    }

    private func badgeCount(for filter: ReminderFilter) -> Int {
        filter.apply(to: viewModel.reminders).count
    }

    // MARK: - List

    private var list: some View {
        List(selection: $viewModel.selectedID) {
            ForEach(viewModel.filtered) { reminder in
                ReminderRow(reminder: reminder, viewModel: viewModel)
                    .tag(reminder.id as UUID?)
                    .contextMenu {
                        Button("Acknowledge") {
                            Task {
                                await viewModel.acknowledge(reminder)
                                await scheduler.cancel(reminder)
                            }
                        }
                        .disabled(reminder.isAcknowledged())
                        Menu("Snooze") {
                            ForEach([5, 10, 15, 30, 60], id: \.self) { minutes in
                                Button("\(minutes) min") {
                                    Task {
                                        let until = Date().addingTimeInterval(Double(minutes) * 60)
                                        await viewModel.snooze(reminder, until: until)
                                        try? await scheduler.snooze(reminder, until: until)
                                    }
                                }
                            }
                        }
                        .disabled(reminder.isAcknowledged())
                        Divider()
                        Button("Delete", role: .destructive) {
                            Task {
                                await scheduler.cancel(reminder)
                                await viewModel.delete(reminder)
                            }
                        }
                    }
            }
        }
        .overlay {
            if viewModel.isLoading {
                ProgressView().controlSize(.large)
            } else if viewModel.filtered.isEmpty {
                emptyListState
            } else if let err = viewModel.loadError {
                ContentUnavailableView(
                    "Couldn't load reminders",
                    systemImage: "exclamationmark.triangle",
                    description: Text(err)
                )
            }
        }
    }

    @ViewBuilder
    private var emptyListState: some View {
        switch viewModel.filter {
        case .upcoming:
            ContentUnavailableView(
                "No upcoming reminders",
                systemImage: "bell.slash",
                description: Text("Use the chat panel or the Reminders import to add one.")
            )
        case .acknowledged:
            ContentUnavailableView(
                "No acknowledged reminders",
                systemImage: "checkmark.circle",
                description: Text("Acknowledged reminders show up here once you dismiss one.")
            )
        case .snoozed:
            ContentUnavailableView(
                "No snoozed reminders",
                systemImage: "moon.zzz",
                description: Text("Snooze an upcoming reminder to see it here.")
            )
        case .all:
            ContentUnavailableView(
                "No reminders",
                systemImage: "tray",
                description: Text("Add a reminder via the chat panel or the New button.")
            )
        }
    }

    // MARK: - Detail empty state

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "bell")
                .font(.system(size: 64))
                .foregroundStyle(.tertiary)
            Text("Select a reminder")
                .font(.title3)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Notifications-disabled banner

    private var notificationsDisabledBanner: some View {
        HStack(spacing: 8) {
            Image(systemName: "bell.slash")
            Text("Notifications are disabled. Open System Settings to enable them.")
                .font(.caption)
            Spacer()
            Button("Open Settings") {
                if let url = URL(string: "x-apple.systempreferences:com.apple.preference.notifications") {
                    NSWorkspace.shared.open(url)
                }
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
        .padding(8)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 6))
        .padding()
        .transition(.opacity)
    }

    private func refreshAuthorization() async {
        authorizationStatus = await scheduler.authorizationStatus()
    }
}

// MARK: - ReminderRow

private struct ReminderRow: View {
    let reminder: Reminder
    let viewModel: ReminderListViewModel

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
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
