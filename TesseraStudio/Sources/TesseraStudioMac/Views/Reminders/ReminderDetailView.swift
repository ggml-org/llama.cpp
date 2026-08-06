#if os(macOS)
import SwiftUI
import TesseraCore

// MARK: - ReminderDetailView (macOS)

/// The detail pane for a single reminder. Renders the
/// reminder's metadata, its linked entities (calendar
/// event, task, contacts), and its constitutional receipt
/// chain.
///
/// The view is deliberately read-only on the right pane;
/// mutations happen via the toolbar's buttons (which call
/// the store + scheduler). The receipt chain below the
/// metadata is the audit trail the user can scroll.
public struct ReminderDetailView: View {

    public init(
        reminder: Reminder,
        store: any ReminderStoring,
        scheduler: ReminderNotificationScheduler
    ) {
        self.reminder = reminder
        self.store = store
        self.scheduler = scheduler
    }

    let reminder: Reminder
    let store: any ReminderStoring
    let scheduler: ReminderNotificationScheduler

    @State private var receipts: [GraphReceipt] = []
    @State private var showError: String?

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                Divider()
                metadata
                Divider()
                receiptChain
            }
            .padding()
        }
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                if !reminder.isAcknowledged() {
                    Button {
                        Task {
                            await scheduler.cancel(reminder)
                            _ = try? await store.acknowledge(id: reminder.id)
                        }
                    } label: {
                        Label("Acknowledge", systemImage: "checkmark.circle")
                    }
                    .help("Mark as acknowledged and cancel the notification")

                    Menu {
                        ForEach([5, 10, 15, 30, 60], id: \.self) { m in
                            Button("\(m) min") { snooze(minutes: m) }
                        }
                    } label: {
                        Label("Snooze", systemImage: "moon.zzz")
                    }
                    .help("Snooze the reminder")
                }
                Button(role: .destructive) {
                    Task {
                        await scheduler.cancel(reminder)
                        _ = try? await store.delete(id: reminder.id)
                    }
                } label: {
                    Label("Delete", systemImage: "trash")
                }
                .help("Delete the reminder")
            }
        }
        .task {
            await loadReceipts()
        }
        .alert("Error",
               isPresented: Binding(
                get: { showError != nil },
                set: { if !$0 { showError = nil } }
               )) {
            Button("OK") { showError = nil }
        } message: {
            Text(showError ?? "")
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack(alignment: .center, spacing: 16) {
            Image(systemName: "bell")
                .font(.system(size: 48))
                .foregroundStyle(.yellow)
            VStack(alignment: .leading, spacing: 4) {
                Text(reminder.title)
                    .font(.title2)
                Text(reminder.displayLine())
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                Text(reminder.priority.rawValue.capitalized)
                    .font(.caption2)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(.quaternary, in: Capsule())
            }
            Spacer()
        }
    }

    // MARK: - Metadata

    private var metadata: some View {
        VStack(alignment: .leading, spacing: 12) {
            row("Trigger", reminder.triggerAt.formatted(date: .abbreviated, time: .shortened))
            row("Offset", reminder.offsetLabel)
            if let snooze = reminder.snoozedUntil {
                row("Snoozed until", snooze.formatted(date: .abbreviated, time: .shortened))
            }
            if let ack = reminder.acknowledgedAt {
                row("Acknowledged", ack.formatted(date: .abbreviated, time: .shortened))
            }
            row("Calendar event", reminder.calendarEventID.uuidString)
            if !reminder.notes.isEmpty {
                Divider()
                Text("Notes")
                    .font(.subheadline)
                    .fontWeight(.medium)
                Text(reminder.notes)
                    .font(.body)
                    .textSelection(.enabled)
            }
        }
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 120, alignment: .leading)
            Text(value)
                .font(.caption)
                .textSelection(.enabled)
            Spacer()
        }
    }

    // MARK: - Receipts

    private var receiptChain: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Receipts")
                .font(.subheadline)
                .fontWeight(.medium)
            if receipts.isEmpty {
                Text("No receipts yet.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(receipts) { r in
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "doc.text")
                            .foregroundStyle(.tertiary)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(r.receiptType)
                                .font(.caption)
                            Text(r.witnessedAt.formatted(date: .abbreviated, time: .shortened))
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                            if !r.payload.isEmpty {
                                Text(Self.formatPayload(r.payload))
                                    .font(.caption2)
                                    .foregroundStyle(.tertiary)
                                    .lineLimit(2)
                            }
                        }
                        Spacer()
                    }
                }
            }
        }
    }

    private static func formatPayload(_ payload: [String: JSONValue]) -> String {
        let parts = payload
            .sorted { $0.key < $1.key }
            .map { (k, v) in
                "\(k)=\(v.shortDescription)"
            }
        return parts.joined(separator: ", ")
    }

    // MARK: - Actions

    private func snooze(minutes: Int) {
        let until = Date().addingTimeInterval(Double(minutes) * 60)
        Task {
            do {
                _ = try await store.snooze(id: reminder.id, until: until)
                try await scheduler.snooze(reminder, until: until)
            } catch {
                showError = String(describing: error)
            }
        }
    }

    private func loadReceipts() async {
        do {
            receipts = try await store.receipts(forReminder: reminder.id)
        } catch {
            receipts = []
        }
    }
}
#endif
