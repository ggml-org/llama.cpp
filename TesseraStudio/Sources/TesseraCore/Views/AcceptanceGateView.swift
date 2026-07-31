import SwiftUI
import Charts

/// Renders the G6 acceptance gate result (llama.tessera.acceptance.v1):
/// a pass/fail verdict, Test 1 (composite beats best single proxy),
/// Test 2 (offline/kernel ranking disagreement), and the per-proxy breakdown.
public struct AcceptanceGateView: View {
    @State private var verdict: AcceptanceVerdict?
    @State private var showImporter = false
    @State private var loadError: String?

    public init(verdict: AcceptanceVerdict? = nil) {
        _verdict = State(initialValue: verdict)
    }

    public var body: some View {
        Group {
            if let verdict {
                content(verdict)
            } else {
                emptyState
            }
        }
        .navigationTitle("Acceptance Gate")
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

    private func content(_ verdict: AcceptanceVerdict) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                verdictHeader(verdict)
                test1Panel(verdict)
                test2Panel(verdict)
                perProxyPanel(verdict)
                provenance(verdict)
            }
            .padding()
        }
    }

    private var emptyState: some View {
        ContentUnavailableView {
            Label("No Acceptance Report", systemImage: "checkmark.seal")
        } description: {
            Text("Load an acceptance gate JSON to see the novelty verdict.")
        } actions: {
            Button("Load Report...") { showImporter = true }
        }
    }

    // MARK: Panels

    private func verdictHeader(_ verdict: AcceptanceVerdict) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Image(systemName: verdict.acceptancePassed ? "checkmark.circle.fill" : "xmark.circle.fill")
                    .font(.title)
                Text(verdict.acceptancePassed ? "PASS" : "FAIL")
                    .font(.system(.title, design: .rounded).bold())
            }
            .foregroundStyle(verdict.acceptancePassed ? .green : .red)

            if !verdict.verdict.isEmpty {
                Text(verdict.verdict)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background((verdict.acceptancePassed ? Color.green : Color.red).opacity(0.1), in: RoundedRectangle(cornerRadius: 10))
    }

    private func test1Panel(_ verdict: AcceptanceVerdict) -> some View {
        let bars: [(name: String, value: Double)] = [
            ("Composite", verdict.compositeT2),
            ("Best single", verdict.bestSingleT2),
        ]
        return card("Test 1 - composite beats single proxy", icon: "chart.bar.fill") {
            Chart(bars, id: \.name) { bar in
                BarMark(
                    x: .value("t_l^2", bar.value),
                    y: .value("Config", bar.name)
                )
                .foregroundStyle(bar.name == "Composite"
                    ? (verdict.compositeWins ? Color.green.gradient : Color.red.gradient)
                    : Color.secondary.gradient)
                .annotation(position: .trailing) {
                    Text(String(format: "%.4g", bar.value))
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                }
            }
            .frame(height: 110)

            HStack {
                Label(
                    String(format: "Improvement: %.1f%%", verdict.improvementPct),
                    systemImage: verdict.compositeWins ? "arrow.down.right" : "arrow.up.right"
                )
                .font(.caption.bold())
                .foregroundStyle(verdict.compositeWins ? .green : .red)
                Spacer()
                passChip(verdict.compositeWins, label: "composite wins")
            }
        }
    }

    private func test2Panel(_ verdict: AcceptanceVerdict) -> some View {
        card("Test 2 - ranking disagreement", icon: "arrow.left.arrow.right") {
            HStack(spacing: 20) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Kendall tau")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Gauge(value: verdict.kendallTau, in: -1...1) {
                        EmptyView()
                    } currentValueLabel: {
                        Text(String(format: "%.2f", verdict.kendallTau))
                            .font(.caption.monospaced())
                    }
                    .gaugeStyle(.accessoryCircular)
                    .tint(.blue)
                }
                VStack(alignment: .leading, spacing: 4) {
                    Text("Ranking disagreement")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Gauge(value: verdict.rankingDisagreement, in: 0...1) {
                        EmptyView()
                    } currentValueLabel: {
                        Text(String(format: "%.0f%%", verdict.rankingDisagreement * 100))
                            .font(.caption.monospaced())
                    }
                    .gaugeStyle(.accessoryCircular)
                    .tint(verdict.noveltySurvives ? .green : .orange)
                }
                Spacer()
                passChip(verdict.noveltySurvives, label: "novelty survives")
            }
            Text("Disagreement > 5% means the kernel-direct signal ranks candidates differently from the offline proxy.")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
    }

    private func perProxyPanel(_ verdict: AcceptanceVerdict) -> some View {
        card("Per-proxy mean t_l^2 (held-out)", icon: "slider.horizontal.3") {
            Chart(verdict.perProxy.labeled, id: \.label) { item in
                BarMark(
                    x: .value("t_l^2", item.value),
                    y: .value("Proxy", item.label)
                )
                .foregroundStyle(.purple.gradient)
                .annotation(position: .trailing) {
                    Text(String(format: "%.4g", item.value))
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                }
            }
            .frame(height: 130)
        }
    }

    private func provenance(_ verdict: AcceptanceVerdict) -> some View {
        card("Provenance", icon: "info.circle") {
            LabeledContent("Tensors (total)", value: "\(verdict.nTensorsTotal)")
            LabeledContent("Tensors (held-out)", value: "\(verdict.nTensorsHeldout)")
            LabeledContent("Schema", value: verdict.schema)
        }
    }

    // MARK: Helpers

    private func passChip(_ passed: Bool, label: String) -> some View {
        Text(label)
            .font(.caption2.bold())
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background((passed ? Color.green : Color.red).opacity(0.15), in: Capsule())
            .foregroundStyle(passed ? .green : .red)
    }

    private func card<Content: View>(_ title: String, icon: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: icon)
                .font(.subheadline.bold())
            content()
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
    }

    private func load(_ result: Result<URL, Error>) {
        switch result {
        case .success(let url):
            do {
                if case .acceptance(let value) = try AnalyticsReport.load(from: url) {
                    verdict = value
                    loadError = nil
                } else {
                    loadError = "That file is not an acceptance gate report."
                }
            } catch {
                loadError = error.localizedDescription
            }
        case .failure(let error):
            loadError = error.localizedDescription
        }
    }
}
