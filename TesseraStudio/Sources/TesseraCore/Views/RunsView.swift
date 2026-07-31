import SwiftUI
import SwiftData

/// Run history table showing past quantization / evaluation runs.
public struct RunsView: View {
    @Query(sort: \RunRecord.timestamp, order: .reverse)
    private var runs: [RunRecord]

    @State private var selectedRun: RunRecord?
    @State private var statusFilter: RunStatus?

    public init() {}

    var filteredRuns: [RunRecord] {
        if let status = statusFilter {
            return runs.filter { $0.status == status }
        }
        return runs
    }

    public var body: some View {
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
    @State private var exportItem: ExportItem?

    /// Best-effort decode of a quantization receipt from the run's metrics.
    private var receipt: QuantizationReceipt? {
        guard let data = run.metricsJSON.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(QuantizationReceipt.self, from: data)
    }

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
                        LabeledContent(key, value: value.shortDescription)
                    }
                }
                Section("Metrics") {
                    ForEach(run.metrics.sorted(by: { $0.key < $1.key }), id: \.key) { key, value in
                        LabeledContent(key, value: value.shortDescription)
                    }
                }
                if let receipt {
                    Section("Receipt") {
                        ReceiptView(receipt: receipt)
                            .listRowInsets(EdgeInsets())
                        Button("Export Receipt as PDF") { exportReceiptPDF(receipt) }
                        Button("Export Charts as PNG") { exportChartsPNG(receipt) }
                    }
                }
            }
            .navigationTitle("Run Details")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .sheet(item: $exportItem) { item in
                ExportView(item: item)
            }
        }
        #if os(macOS)
        .frame(minWidth: 450, minHeight: 400)
        #endif
    }

    private func exportReceiptPDF(_ receipt: QuantizationReceipt) {
        let data = ReceiptPDFRenderer.pdfData(for: receipt)
        exportItem = ExportItem(
            title: "Receipt - \(receipt.model.name)",
            filename: "receipt-\(receipt.model.name).pdf",
            data: data
        )
    }

    private func exportChartsPNG(_ receipt: QuantizationReceipt) {
        guard let data = ChartImageRenderer.pngData(receipt: receipt) else { return }
        exportItem = ExportItem(
            title: "Charts - \(receipt.model.name)",
            filename: "charts-\(receipt.model.name).png",
            data: data
        )
    }

    private func formatDuration(_ seconds: Double) -> String {
        if seconds < 60 { return String(format: "%.1fs", seconds) }
        let mins = Int(seconds) / 60
        let secs = Int(seconds) % 60
        return "\(mins)m \(secs)s"
    }
}
