import SwiftUI

/// Analytics dashboard combining the four runtime-aware pipeline surfaces:
/// MAP-Elites archive browser, G6 acceptance gate, A/B harness receipt, and
/// L2 divergence. The "Load Analytics..." importer sniffs each file's schema
/// and routes it to the matching tab.
public struct AnalyticsDashboardView: View {
    @State private var selection: AnalyticsTab = .archive
    @State private var archive: ArchiveReport?
    @State private var acceptance: AcceptanceVerdict?
    @State private var ab: ABReport?
    @State private var l2: L2Report?
    @State private var showImporter = false
    @State private var loadError: String?

    public init(
        archive: ArchiveReport? = nil,
        acceptance: AcceptanceVerdict? = nil,
        ab: ABReport? = nil,
        l2: L2Report? = nil
    ) {
        _archive = State(initialValue: archive)
        _acceptance = State(initialValue: acceptance)
        _ab = State(initialValue: ab)
        _l2 = State(initialValue: l2)
    }

    public var body: some View {
        TabView(selection: $selection) {
            ArchiveBrowserView(report: archive)
                .id(archive?.id)
                .tabItem { Label("Archive", systemImage: "square.grid.3x3") }
                .tag(AnalyticsTab.archive)

            AcceptanceGateView(verdict: acceptance)
                .id(acceptance?.id)
                .tabItem { Label("Acceptance", systemImage: "checkmark.seal") }
                .tag(AnalyticsTab.acceptance)

            ABReceiptView(report: ab)
                .id(ab?.id)
                .tabItem { Label("A/B Receipt", systemImage: "scale.3d") }
                .tag(AnalyticsTab.ab)

            L2DivergenceView(report: l2)
                .id(l2?.id)
                .tabItem { Label("L2 Divergence", systemImage: "waveform.path.ecg") }
                .tag(AnalyticsTab.l2)
        }
        .navigationTitle("Analytics")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button("Load Analytics...", systemImage: "square.and.arrow.down") {
                    showImporter = true
                }
            }
        }
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

    private func load(_ result: Result<URL, Error>) {
        switch result {
        case .success(let url):
            do {
                switch try AnalyticsReport.load(from: url) {
                case .archive(let report):
                    archive = report
                    selection = .archive
                case .acceptance(let report):
                    acceptance = report
                    selection = .acceptance
                case .ab(let report):
                    ab = report
                    selection = .ab
                case .l2(let report):
                    l2 = report
                    selection = .l2
                }
                loadError = nil
            } catch {
                loadError = error.localizedDescription
            }
        case .failure(let error):
            loadError = error.localizedDescription
        }
    }
}

/// The four analytics surfaces, used as the dashboard's tab selection.
public enum AnalyticsTab: Hashable, CaseIterable, Sendable {
    case archive
    case acceptance
    case ab
    case l2
}
