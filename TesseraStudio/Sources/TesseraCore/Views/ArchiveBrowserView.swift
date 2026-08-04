import SwiftUI

/// Browses a MAP-Elites archive (tessera.map-elites-archive.v1) as a
/// kurtosis x effective-rank grid. The archive is 4D (kurtosis, eff-rank,
/// family, modality); this projects onto the kurtosis/rank plane for the
/// selected modality, keeping the best-fitness cell at each position and
/// aggregating over tensor families. Empty grid positions are shown in gray.
public struct ArchiveBrowserView: View {
    @State private var report: ArchiveReport?
    @State private var modalityFilter: ModalityFilter = .all
    @State private var selected: ArchiveCell?
    @State private var showImporter = false
    @State private var loadError: String?

    public init(report: ArchiveReport? = nil) {
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
        .navigationTitle("Archive")
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.json]) { result in
            load(result)
        }
        .sheet(item: $selected) { cell in
            CellDetailSheet(cell: cell)
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

    // MARK: Content

    private func content(_ report: ArchiveReport) -> some View {
        let filtered = filteredCells(report)
        let grid = gridCells(report, from: filtered)
        let scale = FitnessScale(cells: filtered)

        return ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                summaryBar(report.summary)
                controls
                gridView(report, grid: grid, scale: scale)
                legend(scale)
            }
            .padding()
        }
    }

    private var emptyState: some View {
        ContentUnavailableView {
            Label("No Archive", systemImage: "square.grid.3x3")
        } description: {
            Text("Load a MAP-Elites archive JSON to browse regime-indexed elites.")
        } actions: {
            Button("Load Archive...") { showImporter = true }
        }
    }

    private func summaryBar(_ summary: ArchiveSummary) -> some View {
        HStack(spacing: 12) {
            stat("Occupied", "\(summary.occupiedCells)/\(summary.totalCells)")
            stat("Mean fitness", String(format: "%.4g", summary.meanFitness))
            stat("Best", String(format: "%.4g", summary.bestFitness))
            stat("Worst", String(format: "%.4g", summary.worstFitness))
        }
    }

    private func stat(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(.title3, design: .rounded).bold())
                .monospacedDigit()
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private var controls: some View {
        Picker("Modality", selection: $modalityFilter) {
            ForEach(ModalityFilter.allCases) { filter in
                Text(filter.label).tag(filter)
            }
        }
        .pickerStyle(.segmented)
        .onChange(of: modalityFilter) { _, _ in selected = nil }
    }

    private func gridView(_ report: ArchiveReport, grid: [GridCell?], scale: FitnessScale) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Kurtosis (x) vs effective rank (y)")
                .font(.caption.bold())
            Grid(horizontalSpacing: 4, verticalSpacing: 4) {
                // Highest effective-rank bin at the top.
                ForEach((0..<report.nRankBins).reversed(), id: \.self) { r in
                    GridRow {
                        ForEach(0..<report.nKurtosisBins, id: \.self) { k in
                            tile(grid[r * report.nKurtosisBins + k], scale: scale)
                        }
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func tile(_ cell: GridCell?, scale: FitnessScale) -> some View {
        if let cell {
            Button {
                selected = cell.representative
            } label: {
                RoundedRectangle(cornerRadius: 4)
                    .fill(scale.color(for: cell.representative.bestFitness))
                    .frame(width: 34, height: 34)
                    .overlay(
                        Text(String(format: "%.3g", cell.representative.bestFitness))
                            .font(.system(size: 7, design: .monospaced))
                            .foregroundStyle(scale.textColor(for: cell.representative.bestFitness))
                            .lineLimit(1)
                            .minimumScaleFactor(0.4)
                    )
            }
            .buttonStyle(.plain)
            .help(cell.representative.tensorName)
        } else {
            RoundedRectangle(cornerRadius: 4)
                .fill(.quaternary)
                .frame(width: 34, height: 34)
        }
    }

    private func legend(_ scale: FitnessScale) -> some View {
        HStack(spacing: 8) {
            Text("better")
                .font(.caption2)
                .foregroundStyle(.secondary)
            LinearGradient(
                colors: [scale.color(for: scale.best), scale.color(for: scale.worst)],
                startPoint: .leading,
                endPoint: .trailing
            )
            .frame(width: 120, height: 10)
            .clipShape(Capsule())
            Text("worse")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: Projection

    private func filteredCells(_ report: ArchiveReport) -> [ArchiveCell] {
        report.cells.filter { cell in
            cell.evalCount > 0 && modalityFilter.matches(cell.modalityBucket)
        }
    }

    /// Best-fitness cell at each (rank, kurtosis) position, aggregated over
    /// families (and modalities when the filter is .all). Indexed row-major
    /// as grid[r * nKurtosisBins + k]; nil = empty position.
    private func gridCells(_ report: ArchiveReport, from cells: [ArchiveCell]) -> [GridCell?] {
        var best: [Int: ArchiveCell] = [:]
        for cell in cells {
            let k = Int(cell.kurtosisBucket)
            let r = Int(cell.effRankBucket)
            guard (0..<report.nKurtosisBins).contains(k), (0..<report.nRankBins).contains(r) else { continue }
            let key = r * report.nKurtosisBins + k
            if let existing = best[key] {
                if cell.bestFitness < existing.bestFitness { best[key] = cell }
            } else {
                best[key] = cell
            }
        }
        let count = report.nKurtosisBins * report.nRankBins
        return (0..<count).map { i in
            best[i].map { GridCell(k: i % report.nKurtosisBins, r: i / report.nKurtosisBins, representative: $0) }
        }
    }

    private func load(_ result: Result<URL, Error>) {
        switch result {
        case .success(let url):
            do {
                if case .archive(let archive) = try AnalyticsReport.load(from: url) {
                    report = archive
                    loadError = nil
                } else {
                    loadError = "That file is not a MAP-Elites archive."
                }
            } catch {
                loadError = error.localizedDescription
            }
        case .failure(let error):
            loadError = error.localizedDescription
        }
    }
}

// MARK: - Supporting types

private struct GridCell: Identifiable {
    let k: Int
    let r: Int
    let representative: ArchiveCell
    var id: String { "\(k)-\(r)" }
}

/// Maps a fitness value to a green (best) -> red (worst) color.
/// Internal (not private) so the contrast tests in AnalyticsTests
/// can pin the tile text-color contract.
struct FitnessScale {
    let best: Double
    let worst: Double

    init(cells: [ArchiveCell]) {
        let values = cells.map(\.bestFitness)
        self.best = values.min() ?? 0
        self.worst = values.max() ?? 0
    }

    func color(for fitness: Double) -> Color {
        guard worst > best else { return .green }
        let t = (fitness - best) / (worst - best)
        // hue 0.33 = green, 0 = red
        return Color(hue: (1 - t) * 0.33, saturation: 0.75, brightness: 0.85)
    }

    /// Text color with enough contrast against ``color(for:)``.
    /// The fill spans red (low luminance) to green (high
    /// luminance), so a single fixed text color fails at one end
    /// or the other. Pick white or black from the fill's relative
    /// luminance instead (WCAG contrast, HIG color guidance).
    func textColor(for fitness: Double) -> Color {
        // Degenerate archive (all-equal fitness) renders green,
        // which needs dark text.
        guard worst > best else { return .black }
        let t = (fitness - best) / (worst - best)
        let hue = (1 - t) * 0.33
        return Self.luminance(hue: hue, saturation: 0.75, brightness: 0.85) > 0.45
            ? .black : .white
    }

    /// Relative luminance (0 = black, 1 = white) of an HSB color,
    /// via the standard HSB -> RGB -> linear-luma conversion.
    private static func luminance(hue: Double, saturation: Double, brightness: Double) -> Double {
        let h = hue * 6
        let c = brightness * saturation
        let x = c * (1 - abs(h.truncatingRemainder(dividingBy: 2) - 1))
        let m = brightness - c
        let (r, g, b): (Double, Double, Double)
        switch h {
        case ..<1: (r, g, b) = (c, x, 0)
        case ..<2: (r, g, b) = (x, c, 0)
        case ..<3: (r, g, b) = (0, c, x)
        case ..<4: (r, g, b) = (0, x, c)
        case ..<5: (r, g, b) = (x, 0, c)
        default:   (r, g, b) = (c, 0, x)
        }
        // sRGB relative luminance.
        func lin(_ v: Double) -> Double {
            v <= 0.03928 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4)
        }
        return 0.2126 * lin(r + m) + 0.7152 * lin(g + m) + 0.0722 * lin(b + m)
    }
}

enum ModalityFilter: CaseIterable, Identifiable {
    case all, text, image, audio

    var id: Self { self }

    var label: String {
        switch self {
        case .all: "All"
        case .text: "Text"
        case .image: "Image"
        case .audio: "Audio"
        }
    }

    func matches(_ bucket: Int) -> Bool {
        switch self {
        case .all: true
        case .text: bucket == 0
        case .image: bucket == 1
        case .audio: bucket == 2
        }
    }
}

private struct CellDetailSheet: View {
    let cell: ArchiveCell
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section("Regime cell") {
                    LabeledContent("Tensor", value: cell.tensorName.isEmpty ? "-" : cell.tensorName)
                    LabeledContent("Modality", value: cell.modalityName)
                    LabeledContent("Kurtosis bin", value: "\(Int(cell.kurtosisBucket))")
                    LabeledContent("Eff-rank bin", value: "\(Int(cell.effRankBucket))")
                    LabeledContent("Family bin", value: "\(cell.familyBucket)")
                }
                Section("Best policy") {
                    LabeledContent("Fitness", value: String(format: "%.6g", cell.bestFitness))
                    LabeledContent("AWQ alpha", value: String(format: "%.4g", cell.bestAlpha))
                    LabeledContent("AWQ clip", value: String(format: "%.4g", cell.bestClip))
                    LabeledContent("Evaluations", value: "\(cell.evalCount)")
                }
            }
            .navigationTitle("Archive Cell")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        #if os(macOS)
        .frame(minWidth: 360, minHeight: 380)
        #endif
    }
}
