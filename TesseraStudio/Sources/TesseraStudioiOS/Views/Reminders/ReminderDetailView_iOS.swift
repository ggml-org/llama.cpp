#if os(iOS)
import SwiftUI
import TesseraCore

// MARK: - ReminderDetailView_iOS

/// The iOS detail view for a single reminder. Mirrors
/// ``ReminderDetailView`` (macOS) but in a single-pane
/// layout optimized for touch — the receipt chain scrolls
/// inline rather than in a side inspector, and the
/// acknowledge / snooze / delete actions are in a bottom
/// toolbar.
public struct ReminderDetailView_iOS: View {

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
    @State private var showSnoozeSheet = false
    @State private var showError: String?

    public var body: some View {
        Form {
            Section {
                HStack(spacing: 12) {
                    Image(systemName: "bell")
                        .font(.system(size: 36))
                        .foregroundStyle(.yellow)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(reminder.title)
                            .font(.headline)
                        Text(reminder.displayLine())
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            Section("Details") {
                labelRow("Trigger", reminder.triggerAt.formatted(date: .abbreviated, time: .shortened))
                labelRow("Offset", reminder.offsetLabel)
                if let snooze = reminder.snoozedUntil {
                    labelRow("Snoozed until", snooze.formatted(date: .abbreviated, time: .shortened))
                }
                if let ack = reminder.acknowledgedAt {
                    labelRow("Acknowledged", ack.formatted(date: .abbreviated, time: .shortened))
                }
                labelRow("Priority", reminder.priority.rawValue.capitalized)
            }
            if !reminder.notes.isEmpty {
                Section("Notes") {
                    Text(reminder.notes)
                }
            }
            Section("Receipts") {
                if receipts.isEmpty {
                    Text("No receipts yet.")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(receipts) { r in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(r.receiptType)
                                .font(.subheadline)
                            Text(r.witnessedAt.formatted(date: .abbreviated, time: .shortened))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            if !r.payload.isEmpty {
                                Text(Self.formatPayload(r.payload))
                                    .font(.caption2)
                                    .foregroundStyle(.tertiary)
                            }
                        }
                    }
                }
            }
        }
        .navigationTitle("Reminder")
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Menu {
                    if !reminder.isAcknowledged() {
                        Button {
                            Task {
                                await scheduler.cancel(reminder)
                                _ = try? await store.acknowledge(id: reminder.id)
                            }
                        } label: {
                            Label("Acknowledge", systemImage: "checkmark.circle")
                        }
                        Button {
                            showSnoozeSheet = true
                        } label: {
                            Label("Snooze", systemImage: "moon.zzz")
                        }
                    }
                    Button(role: .destructive) {
                        Task {
                            await scheduler.cancel(reminder)
                            _ = try? await store.delete(id: reminder.id)
                        }
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
            }
        }
        .sheet(isPresented: $showSnoozeSheet) {
            SnoozePickerSheet { minutes in
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
        }
        .alert("Error", isPresented: Binding(
            get: { showError != nil },
            set: { if !$0 { showError = nil } }
        )) {
            Button("OK") { showError = nil }
        } message: {
            Text(showError ?? "")
        }
        .task { await loadReceipts() }
    }

    // MARK: - Helpers

    private func labelRow(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label).foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .multilineTextAlignment(.trailing)
        }
    }

    private static func formatPayload(_ payload: [String: JSONValue]) -> String {
        payload
            .sorted { $0.key < $1.key }
            .map { "\($0.key)=\($1.value.shortDescription)" }
            .joined(separator: ", ")
    }

    private func loadReceipts() async {
        do {
            receipts = try await store.receipts(forReminder: reminder.id)
        } catch {
            receipts = []
        }
    }
}

// MARK: - SnoozePickerSheet

private struct SnoozePickerSheet: View {
    let onPick: (Int) -> Void
    @Environment(\.dismiss) private var dismiss

    private let presets = [5, 10, 15, 30, 60]

    var body: some View {
        NavigationStack {
            List {
                ForEach(presets, id: \.self) { m in
                    Button("\(m) min") {
                        onPick(m)
                        dismiss()
                    }
                }
            }
            .navigationTitle("Snooze")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
        .presentationDetents([.medium])
    }
}
#endif
