import SwiftUI

/// Renders a quantization receipt (from `tessera receipts` JSON) as
/// structured cards: model info, per-tensor stats, calibration config,
/// and the GA archive summary. See design doc 14.10.
public struct ReceiptView: View {
    public let receipt: QuantizationReceipt
    public let archive: ArchiveReport?

    public init(receipt: QuantizationReceipt, archive: ArchiveReport? = nil) {
        self.receipt = receipt
        self.archive = archive
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            modelCard
            if !receipt.tensors.isEmpty {
                tensorCard
            }
            calibrationCard
            if let ga = receipt.gaArchive {
                gaCard(ga)
            }
            if let archive {
                archiveCard(archive)
            }
            MetricsChartView(receipt: receipt)
        }
        .padding()
        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 12))
    }

    private var header: some View {
        HStack {
            Label("Quantization Receipt", systemImage: "doc.badge.gearshape")
                .font(.headline)
            Spacer()
            Text(receipt.schemaVersion)
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
        }
    }

    private var modelCard: some View {
        ReceiptCard(title: "Model", icon: "cube.box") {
            ReceiptRow(label: "Name", value: receipt.model.name)
            ReceiptRow(label: "Family", value: receipt.model.family)
            ReceiptRow(label: "Parameters", value: receipt.model.parameterCount)
            ReceiptRow(label: "Bits", value: String(format: "%.1f -> %.2f", receipt.model.sourceBits, receipt.model.outputBits))
            ReceiptRow(label: "Size", value: ByteCountFormatter.string(fromByteCount: receipt.model.fileSizeBytes, countStyle: .file))
        }
    }

    private var tensorCard: some View {
        ReceiptCard(title: "Per-tensor stats (\(receipt.tensors.count))", icon: "list.bullet.rectangle") {
            ForEach(receipt.tensors.prefix(12)) { tensor in
                HStack {
                    Text(tensor.name)
                        .font(.system(.caption, design: .monospaced))
                        .lineLimit(1)
                    Spacer()
                    Text(String(format: "%.2fb", tensor.bits))
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                    Text(String(format: "mse %.4g", tensor.mse))
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                    Text(String(format: "snr %.1fdB", tensor.snrDB))
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                }
            }
            if receipt.tensors.count > 12 {
                Text("+ \(receipt.tensors.count - 12) more")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    private var calibrationCard: some View {
        ReceiptCard(title: "Calibration", icon: "slider.horizontal.3") {
            ReceiptRow(label: "Corpus", value: receipt.calibration.corpus.isEmpty ? "-" : receipt.calibration.corpus)
            ReceiptRow(label: "Tokens", value: "\(receipt.calibration.tokenCount)")
            ReceiptRow(label: "Modality", value: receipt.calibration.modality)
            ReceiptRow(label: "Dequant", value: receipt.calibration.dequantMode)
        }
    }

    private func gaCard(_ ga: ReceiptGAArchive) -> some View {
        ReceiptCard(title: "GA archive", icon: "point.topleft.down.curvedto.point.bottomright.up") {
            ReceiptRow(label: "Generations", value: "\(ga.generations)")
            ReceiptRow(label: "Population", value: "\(ga.population)")
            ReceiptRow(label: "Best fitness", value: String(format: "%.4g", ga.bestFitness))
            ReceiptRow(label: "Archive size", value: "\(ga.archiveSize)")
        }
    }

    private func archiveCard(_ archive: ArchiveReport) -> some View {
        let summary = archive.summary
        return ReceiptCard(title: "MAP-Elites archive", icon: "square.grid.3x3") {
            ReceiptRow(label: "Occupied cells", value: "\(summary.occupiedCells)/\(summary.totalCells)")
            ReceiptRow(label: "Mean fitness", value: String(format: "%.4g", summary.meanFitness))
            ReceiptRow(label: "Best fitness", value: String(format: "%.4g", summary.bestFitness))
            ReceiptRow(label: "Worst fitness", value: String(format: "%.4g", summary.worstFitness))
        }
    }
}

private struct ReceiptCard<Content: View>: View {
    let title: String
    let icon: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label(title, systemImage: icon)
                .font(.subheadline.bold())
            content
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
    }
}

private struct ReceiptRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack(alignment: .top) {
            Text(label)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .multilineTextAlignment(.trailing)
                .textSelection(.enabled)
        }
        .font(.caption)
    }
}
