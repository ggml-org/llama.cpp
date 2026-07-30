import SwiftUI
import SwiftData

/// Run history table showing past quantization / evaluation runs.
struct RunsView: View {
    @Query(sort: \RunRecord.timestamp, order: .reverse)
    private var runs: [RunRecord]

    @State private var selectedRun: RunRecord?
    @State private var statusFilter: RunStatus?

    var filteredRuns: [RunRecord] {
        if let status = statusFilter {
            return runs.filter { $0.status == status }
        }
        return runs
    }

    var body: some View {
        Group {
            if filteredRuns.isEmpty {
                ContentUnavailableView(
                    "No Runs",
                    systemImage: "clock.badge.questionmark",
                    description: Text("Quantization and evaluation runs will appear here.")
                )
            } else {
                #if os(macOS)
                Table(filteredRuns, selection: Binding(
                    get: { selectedRun?.persistentModelID },
                    set: { id in selectedRun = runs.first { $0.persistentModelID == id } }
                )) {
                    TableColumn("Model", value: \.modelName)
                    TableColumn("Runtime") { run in
                        Text(run.runtime.displayName)
                    }
                    TableColumn("Status") { run in
                        StatusBadge(status: run.status)
                    }
                    TableColumn("Duration") { run in
                        Text(formatDuration(run.durationSeconds))
                    }
                    TableColumn("Date") { run in
                        Text(run.timestamp, style: .relative)
                    }
                }
                #else
                List(filteredRuns) { run in
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text(run.modelName)
                                .font(.headline)
                            Spacer()
                            StatusBadge(status: run.status)
                        }
                        HStack {
                            Text(run.runtime.displayName)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Spacer()
                            Text(run.timestamp, style: .relative)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .padding(.vertical, 4)
                }
                #endif
            }
        }
        .navigationTitle("Runs")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Picker("Status", selection: $statusFilter) {
                    Text("All").tag(RunStatus?.none)
                    ForEach([RunStatus.running, .completed, .failed, .cancelled], id: \.self) { status in
                        Text(status.rawValue.capitalized).tag(RunStatus?.some(status))
                    }
                }
            }
        }
        .sheet(item: $selectedRun) { run in
            RunDetailSheet(run: run)
        }
    }

    private func formatDuration(_ seconds: Double) -> String {
        if seconds < 60 { return String(format: "%.1fs", seconds) }
        let mins = Int(seconds) / 60
        let secs = Int(seconds) % 60
        return "\(mins)m \(secs)s"
    }
}

struct StatusBadge: View {
    let status: RunStatus

    var body: some View {
        Text(status.rawValue)
            .font(.caption2.bold())
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(color.opacity(0.15), in: Capsule())
            .foregroundStyle(color)
    }

    private var color: Color {
        switch status {
        case .running: .blue
        case .completed: .green
        case .failed: .red
        case .cancelled: .orange
        }
    }
}

struct RunDetailSheet: View {
    let run: RunRecord
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section("Model") {
                    LabeledContent("Name", value: run.modelName)
                    LabeledContent("Runtime", value: run.runtime.displayName)
                    LabeledContent("Status", value: run.status.rawValue)
                    LabeledContent("Duration", value: formatDuration(run.durationSeconds))
                    LabeledContent("Date", value: run.timestamp.formatted())
                }
                Section("Configuration") {
                    ForEach(run.config.sorted(by: { $0.key < $1.key }), id: \.key) { key, value in
                        LabeledContent(key, value: describe(value))
                    }
                }
                Section("Metrics") {
                    ForEach(run.metrics.sorted(by: { $0.key < $1.key }), id: \.key) { key, value in
                        LabeledContent(key, value: describe(value))
                    }
                }
            }
            .navigationTitle("Run Details")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        #if os(macOS)
        .frame(minWidth: 450, minHeight: 400)
        #endif
    }

    private func formatDuration(_ seconds: Double) -> String {
        if seconds < 60 { return String(format: "%.1fs", seconds) }
        let mins = Int(seconds) / 60
        let secs = Int(seconds) % 60
        return "\(mins)m \(secs)s"
    }

    private func describe(_ value: JSONValue) -> String {
        switch value {
        case .string(let s): s
        case .number(let n): String(format: "%.4g", n)
        case .bool(let b): b ? "true" : "false"
        case .null: "null"
        case .array(let a): "[\(a.count) items]"
        case .object(let o): "{\(o.count) keys}"
        }
    }
}
