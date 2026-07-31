import SwiftUI
import Charts

/// Renders an A/B harness receipt: the offline-proxy composite against the
/// kernel-direct composite, a per-tensor scatter (offline vs kernel-direct),
/// the Kendall-tau ranking agreement, and the tensors whose proxy/kernel
/// ranks disagree the most.
public struct ABReceiptView: View {
    @State private var report: ABReport?
    @State private var showImporter = false
    @State private var loadError: String?

    public init(report: ABReport? = nil) {
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
        .navigationTitle("A/B Receipt")
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

    private func content(_ report: ABReport) -> some View {
        let disagreed = report.mostDisagreedTensors
        return ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header(report)
                compositePanel(report)
                scatterPanel(report, disagreed: disagreed)
                if !disagreed.isEmpty {
                    disagreementPanel(report, disagreed: disagreed)
                }
            }
            .padding()
        }
    }

    private var emptyState: some View {
        ContentUnavailableView {
            Label("No A/B Receipt", systemImage: "scale.3d")
        } description: {
            Text("Load an A/B harness receipt JSON to compare offline proxy vs kernel-direct fitness.")
        } actions: {
            Button("Load Receipt...") { showImporter = true }
        }
    }

    // MARK: Panels

    private func header(_ report: ABReport) -> some View {
        HStack(spacing: 12) {
            badge("Kendall tau", String(format: "%.3f", report.kendallTau), .blue)
            badge("Disagreement", String(format: "%.0f%%", report.rankingDisagreement * 100), .orange)
            badge("Tensors", "\(report.nTensors)", .secondary)
            Spacer()
            if report.compositeBeatsSingle {
                Label("composite beats single", systemImage: "checkmark.circle.fill")
                    .font(.caption.bold())
                    .foregroundStyle(.green)
            }
        }
    }

    private func compositePanel(_ report: ABReport) -> some View {
        HStack(spacing: 12) {
            compositeCard("Offline proxy", report.compositeOffline, .teal)
            compositeCard("Kernel-direct", report.compositeKernel, .indigo)
        }
    }

    private func compositeCard(_ title: String, _ value: Double, _ color: Color) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(String(format: "%.5g", value))
                .font(.system(.title2, design: .rounded).bold())
                .monospacedDigit()
                .foregroundStyle(color)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))
    }

    private func scatterPanel(_ report: ABReport, disagreed: Set<String>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Per-tensor: offline proxy (x) vs kernel-direct (y)")
                .font(.caption.bold())
            Chart {
                ForEach(report.scores) { score in
                    PointMark(
                        x: .value("Offline proxy", score.offlineProxyMSE),
                        y: .value("Kernel-direct", score.kernelDirectT2)
                    )
                    .foregroundStyle(disagreed.contains(score.name) ? Color.red : Color.blue)
                    .symbolSize(disagreed.contains(score.name) ? 90 : 40)
                }
                if let line = diagonal(report) {
                    LineMark(
                        x: .value("Offline proxy", line.min),
                        y: .value("Kernel-direct", line.min)
                    )
                    LineMark(
                        x: .value("Offline proxy", line.max),
                        y: .value("Kernel-direct", line.max)
                    )
                    .foregroundStyle(.gray.opacity(0.5))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 4]))
                }
            }
            .frame(height: 240)
            Text("Dashed line = perfect proxy/kernel agreement; red points rank differently under the two signals.")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
    }

    private func disagreementPanel(_ report: ABReport, disagreed: Set<String>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Label("Largest ranking disagreements", systemImage: "exclamationmark.triangle")
                .font(.subheadline.bold())
                .foregroundStyle(.red)
            ForEach(report.scores.filter { disagreed.contains($0.name) }) { score in
                HStack {
                    Text(score.name)
                        .font(.system(.caption, design: .monospaced))
                        .lineLimit(1)
                    Spacer()
                    Text(String(format: "off %.4g", score.offlineProxyMSE))
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                    Text(String(format: "kern %.4g", score.kernelDirectT2))
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
    }

    // MARK: Helpers

    private func badge(_ label: String, _ value: String, _ color: Color) -> some View {
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
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    /// The y=x reference extent spanning the observed scores.
    private func diagonal(_ report: ABReport) -> (min: Double, max: Double)? {
        let values = report.scores.flatMap { [$0.offlineProxyMSE, $0.kernelDirectT2] }
        guard let lo = values.min(), let hi = values.max(), hi > lo else { return nil }
        return (lo, hi)
    }

    private func load(_ result: Result<URL, Error>) {
        switch result {
        case .success(let url):
            do {
                if case .ab(let value) = try AnalyticsReport.load(from: url) {
                    report = value
                    loadError = nil
                } else {
                    loadError = "That file is not an A/B harness receipt."
                }
            } catch {
                loadError = error.localizedDescription
            }
        case .failure(let error):
            loadError = error.localizedDescription
        }
    }
}
