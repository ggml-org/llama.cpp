import SwiftUI

/// Model library browser. Shows model cards with badges and metadata.
public struct LibraryView: View {
    @State private var models: [ModelInfo] = []
    @State private var searchText = ""
    @State private var runtimeFilter: TesseraRuntime?

    public init() {}

    var filteredModels: [ModelInfo] {
        var result = models
        if let runtime = runtimeFilter {
            result = result.filter { $0.runtime == runtime }
        }
        if !searchText.isEmpty {
            result = result.filter {
                $0.name.localizedCaseInsensitiveContains(searchText) ||
                $0.family.localizedCaseInsensitiveContains(searchText)
            }
        }
        return result
    }

    public var body: some View {
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 280, maximum: 400))], spacing: 16) {
                ForEach(filteredModels) { model in
                    ModelCardView(model: model)
                }
            }
            .padding()
        }
        .navigationTitle("Library")
        .searchable(text: $searchText, prompt: "Search models")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Picker("Runtime", selection: $runtimeFilter) {
                    Text("All").tag(TesseraRuntime?.none)
                    ForEach(TesseraRuntime.allCases, id: \.self) { rt in
                        Text(rt.displayName).tag(TesseraRuntime?.some(rt))
                    }
                }
            }
            ToolbarItem(placement: .primaryAction) {
                Button("Scan", systemImage: "arrow.clockwise") {
                    scanModels()
                }
            }
        }
        .onAppear { scanModels() }
        .overlay {
            if filteredModels.isEmpty {
                ContentUnavailableView(
                    "No Models",
                    systemImage: "cube.transparent",
                    description: Text("Scan a directory or add models to get started.")
                )
            }
        }
    }

    private func scanModels() {
        let fm = FileManager.default
        let primary = NSString(string: TesseraSettings.modelDirectory).expandingTildeInPath
        let dirs = [primary, NSString(string: "~/Models").expandingTildeInPath]

        var found: [ModelInfo] = []
        for dir in dirs {
            guard let contents = try? fm.contentsOfDirectory(atPath: dir) else { continue }
            for file in contents where file.hasSuffix(".gguf") || file.hasSuffix(".mlmodelc") {
                let path = (dir as NSString).appendingPathComponent(file)
                let attrs = try? fm.attributesOfItem(atPath: path)
                let size = (attrs?[.size] as? Int64) ?? 0
                let isTessera = file.contains("tessera") || file.contains("TSQ")
                let isMLC = file.hasSuffix(".mlmodelc")

                found.append(ModelInfo(
                    name: file.replacingOccurrences(of: ".gguf", with: "")
                        .replacingOccurrences(of: ".mlmodelc", with: ""),
                    family: guessFamily(from: file),
                    parameterCount: guessParams(from: file),
                    quantization: isTessera ? "Tessera" : "stock",
                    effectiveBits: isTessera ? 3.5 : 4.5,
                    fileSizeBytes: size,
                    runtime: isMLC ? .onDevice : .mlx,
                    isTesseraQuantized: isTessera,
                    hasMLModelC: isMLC,
                    ggufPath: file.hasSuffix(".gguf") ? path : nil,
                    mlmodelcPath: isMLC ? path : nil
                ))
            }
        }
        models = found.sorted { $0.name < $1.name }
    }

    private func guessFamily(from filename: String) -> String {
        let lower = filename.lowercased()
        if lower.contains("gemma") { return "Gemma" }
        if lower.contains("llama") { return "LLaMA" }
        if lower.contains("mistral") { return "Mistral" }
        if lower.contains("phi") { return "Phi" }
        if lower.contains("qwen") { return "Qwen" }
        return "Unknown"
    }

    private func guessParams(from filename: String) -> String {
        let lower = filename.lowercased()
        if lower.contains("12b") { return "12B" }
        if lower.contains("7b") { return "7B" }
        if lower.contains("4b") { return "4B" }
        if lower.contains("3b") { return "3B" }
        if lower.contains("1b") { return "1B" }
        return "?"
    }
}

/// A single model card in the library grid.
struct ModelCardView: View {
    let model: ModelInfo

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(model.name)
                    .font(.headline)
                    .lineLimit(1)
                Spacer()
                Image(systemName: model.runtime.icon)
                    .foregroundStyle(.secondary)
            }

            HStack(spacing: 4) {
                ForEach(model.badges, id: \.self) { badge in
                    BadgeView(badge: badge)
                }
                if model.hasSidecar {
                    BadgeView(badge: .ane)
                }
            }

            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
                GridRow {
                    Text("Family").foregroundStyle(.secondary)
                    Text(model.family)
                }
                GridRow {
                    Text("Params").foregroundStyle(.secondary)
                    Text(model.parameterCount)
                }
                GridRow {
                    Text("Bits").foregroundStyle(.secondary)
                    Text(String(format: "%.1f", model.effectiveBits))
                }
                GridRow {
                    Text("Size").foregroundStyle(.secondary)
                    Text(model.fileSizeFormatted)
                }
            }
            .font(.caption)
        }
        .padding()
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 12))
    }
}

struct BadgeView: View {
    let badge: ModelBadge

    var body: some View {
        Text(badge.rawValue)
            .font(.caption2.bold())
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.15), in: Capsule())
            .foregroundStyle(color)
    }

    private var color: Color {
        switch badge.color {
        case "green": .green
        case "blue": .blue
        case "purple": .purple
        case "orange": .orange
        case "pink": .pink
        default: .gray
        }
    }
}
