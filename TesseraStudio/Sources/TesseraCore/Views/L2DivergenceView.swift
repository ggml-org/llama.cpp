import SwiftUI
import Charts

/// Renders an L2 BF16-vs-quantized divergence report
/// (llama.tessera.runtime-probe.v1): per-tensor divergence sorted worst-first,
/// flagged tensors (relative Frobenius above their type's tolerance) highlighted,
/// and a chart comparing each tensor's divergence against its type-specific
/// flag threshold.
public struct L2DivergenceView: View {
    @State private var report: L2Report?
    @State private var flaggedOnly = false
    @State private var showImporter = false
    @State private var loadError: String?

    public init(report: L2Report? = nil) {
        _report = State(initialValue: report)
    }

    public var body: some View {
        Group {
            if let report {
                content(report)
            } else {
                emptyState
            }
        }
        .navigationTitle("L2 Divergence")
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.json]) { result in
            load(result)
        }
        .alert("Load failed", isPresented: Binding(
            get: { loadError != nil },
            set: { if !$0 { loadError = nil } }
        )) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(loadError ?? "")
        }
    }

    private func rows(_ report: L2Report) -> [L2TensorResult] {
        let source = flaggedOnly ? report.tensors.filter(\.flagged) : report.tensors
        return source.sorted { $0.divergence.relativeFrobenius > $1.divergence.relativeFrobenius }
    }

    private func content(_ report: L2Report) -> some View {
        let rows = rows(report)
        return VStack(alignment: .leading, spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    summary(report)
                    chart(rows)
                }
                .padding()
            }
            table(rows)
        }
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Toggle("Flagged only", isOn: $flaggedOnly)
                    .toggleStyle(.switch)
            }
        }
    }

    private var emptyState: some View {
        ContentUnavailableView {
            Label("No L2 Report", systemImage: "waveform.path.ecg")
        } description: {
            Text("Load an L2 divergence JSON to inspect per-tensor weight divergence.")
        } actions: {
            Button("Load Report...") { showImporter = true }
        }
    }

    // MARK: Sections

    private func summary(_ report: L2Report) -> some View {
        HStack(spacing: 12) {
            stat("Flagged", "\(report.nFlagged)/\(report.nTensors)", report.nFlagged > 0 ? .red : .green)
            stat("Flag multiplier", String(format: "%.2fx", report.flagMultiplier), .primary)
            stat("Layer", report.layer, .primary)
        }
    }

    private func stat(_ label: String, _ value: String, _ color: Color) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(.title3, design: .rounded).bold())
                .monospacedDigit()
                .foregroundStyle(color)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    /// Bars = relative Frobenius divergence; dots = each tensor's type-specific
    /// flag threshold. A bar above its dot is flagged.
    private func chart(_ rows: [L2TensorResult]) -> some View {
        let top = Array(rows.prefix(20))
        return VStack(alignment: .leading, spacing: 6) {
            Text("Divergence vs type tolerance (top \(top.count))")
                .font(.caption.bold())
            Chart {
                ForEach(top) { row in
                    BarMark(
                        x: .value("Rel. Frobenius", row.divergence.relativeFrobenius),
                        y: .value("Tensor", shortName(row.tensor))
                    )
                    .foregroundStyle(row.flagged ? Color.red.gradient : Color.blue.gradient)
                }
                ForEach(top) { row in
                    PointMark(
                        x: .value("Threshold", row.flagThreshold),
                        y: .value("Tensor", shortName(row.tensor))
                    )
                    .symbol(.diamond)
                    .foregroundStyle(.orange)
                }
            }
            .frame(height: CGFloat(max(top.count, 1)) * 22 + 40)
            Text("Diamond = per-type flag threshold (\(String(format: "%.2fx", report?.flagMultiplier ?? 1.5)) baseline).")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
    }

    @ViewBuilder
    private func table(_ rows: [L2TensorResult]) -> some View {
        #if os(macOS)
        Table(rows) {
            TableColumn("Tensor") { row in
                HStack(spacing: 4) {
                    if row.flagged {
                        Image(systemName: "flag.fill").foregroundStyle(.red).font(.caption2)
                    }
                    Text(row.tensor).font(.system(.caption, design: .monospaced)).lineLimit(1)
                }
            }
            TableColumn("Type") { row in
                Text(row.qtype).font(.caption2.monospaced())
            }
            TableColumn("Rel. Frob") { row in
                Text(String(format: "%.4g", row.divergence.relativeFrobenius))
                    .font(.caption2.monospaced())
                    .foregroundStyle(row.flagged ? .red : .primary)
            }
            TableColumn("Baseline") { row in
                Text(String(format: "%.4g", row.expectedFrob)).font(.caption2.monospaced())
            }
            TableColumn("Threshold") { row in
                Text(String(format: "%.4g", row.flagThreshold)).font(.caption2.monospaced())
            }
        }
        .frame(minHeight: 200)
        #else
        List(rows) { row in
            VStack(alignment: .leading, spacing: 3) {
                HStack {
                    if row.flagged {
                        Image(systemName: "flag.fill").foregroundStyle(.red).font(.caption2)
                    }
                    Text(row.tensor).font(.system(.caption, design: .monospaced)).lineLimit(1)
                    Spacer()
                    Text(row.qtype).font(.caption2.monospaced()).foregroundStyle(.secondary)
                }
                HStack {
                    Text(String(format: "frob %.4g", row.divergence.relativeFrobenius))
                        .foregroundStyle(row.flagged ? .red : .primary)
                    Spacer()
                    Text(String(format: "thr %.4g", row.flagThreshold))
                }
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
            }
            .padding(.vertical, 2)
            .listRowBackground(row.flagged ? Color.red.opacity(0.08) : Color.clear)
        }
        #endif
    }

    // MARK: Helpers

    private func shortName(_ name: String) -> String {
        name.count > 28 ? "..." + name.suffix(25) : name
    }

    private func load(_ result: Result<URL, Error>) {
        switch result {
        case .success(let url):
            do {
                if case .l2(let value) = try AnalyticsReport.load(from: url) {
                    report = value
                    loadError = nil
                } else {
                    loadError = "That file is not an L2 divergence report."
                }
            } catch {
                loadError = error.localizedDescription
            }
        case .failure(let error):
            loadError = error.localizedDescription
        }
    }
}
