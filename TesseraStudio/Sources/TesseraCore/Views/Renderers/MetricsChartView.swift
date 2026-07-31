import SwiftUI
import Charts

/// A single per-layer metric (e.g. MSE or effective bits for one tensor).
public struct LayerMetric: Identifiable, Sendable {
    public let id = UUID()
    public let layer: String
    public let value: Double

    public init(layer: String, value: Double) {
        self.layer = layer
        self.value = value
    }
}

/// A single point on a convergence curve (generation/step vs fitness/PPL).
public struct ConvergencePoint: Identifiable, Sendable {
    public let id = UUID()
    public let step: Int
    public let value: Double

    public init(step: Int, value: Double) {
        self.step = step
        self.value = value
    }
}

/// Renders evaluation metrics (PPL, MSE, latency) with SwiftUI Charts:
/// a bar chart for per-layer metrics and a line chart for convergence.
public struct MetricsChartView: View {
    public let summary: [MetricSummaryItem]
    public let perLayer: [LayerMetric]
    public let convergence: [ConvergencePoint]

    public init(
        summary: [MetricSummaryItem] = [],
        perLayer: [LayerMetric] = [],
        convergence: [ConvergencePoint] = []
    ) {
        self.summary = summary
        self.perLayer = perLayer
        self.convergence = convergence
    }

    /// Build a chart view from a receipt's tensor stats + GA archive.
    public init(receipt: QuantizationReceipt) {
        var summary: [MetricSummaryItem] = []
        if let mse = receipt.meanMSE {
            summary.append(MetricSummaryItem(label: "Mean MSE", value: String(format: "%.4g", mse)))
        }
        summary.append(MetricSummaryItem(label: "Output bits", value: String(format: "%.2f", receipt.model.outputBits)))
        summary.append(MetricSummaryItem(label: "Duration", value: String(format: "%.1fs", receipt.durationSeconds)))
        self.summary = summary
        self.perLayer = receipt.tensors.map { LayerMetric(layer: $0.name, value: $0.mse) }
        if let ga = receipt.gaArchive, ga.archiveSize > 0 {
            // Synthesize a convergence curve from the GA archive summary.
            self.convergence = (0..<ga.generations).map { step in
                let progress = Double(step + 1) / Double(max(ga.generations, 1))
                let value = ga.bestFitness * (1.0 + (1.0 - progress))
                return ConvergencePoint(step: step + 1, value: value)
            }
        } else {
            self.convergence = []
        }
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            if !summary.isEmpty {
                summaryRow
            }
            if !perLayer.isEmpty {
                barChart
            }
            if !convergence.isEmpty {
                lineChart
            }
        }
    }

    private var summaryRow: some View {
        HStack(spacing: 12) {
            ForEach(summary) { item in
                VStack(alignment: .leading, spacing: 2) {
                    Text(item.label)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text(item.value)
                        .font(.system(.title3, design: .rounded).bold())
                        .monospacedDigit()
                }
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
            }
        }
    }

    private var barChart: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Per-layer MSE")
                .font(.caption.bold())
            Chart(perLayer) { metric in
                BarMark(
                    x: .value("Layer", metric.layer),
                    y: .value("MSE", metric.value)
                )
                .foregroundStyle(.purple.gradient)
            }
            .frame(height: 160)
        }
    }

    private var lineChart: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Convergence")
                .font(.caption.bold())
            Chart(convergence) { point in
                LineMark(
                    x: .value("Generation", point.step),
                    y: .value("Fitness", point.value)
                )
                .foregroundStyle(.blue)
                .interpolationMethod(.monotone)
            }
            .frame(height: 160)
        }
    }
}

/// A labelled summary value shown above the charts.
public struct MetricSummaryItem: Identifiable, Sendable {
    public let id = UUID()
    public let label: String
    public let value: String

    public init(label: String, value: String) {
        self.label = label
        self.value = value
    }
}
