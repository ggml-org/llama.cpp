import SwiftUI
import CoreGraphics
import CoreText
#if canImport(AppKit)
import AppKit
#endif
#if canImport(UIKit)
import UIKit
#endif

/// Export formats supported by the studio.
public enum ExportFormat: String, CaseIterable, Sendable {
    case markdown
    case json
    case pdf
    case png

    public var fileExtension: String {
        switch self {
        case .markdown: "md"
        case .json: "json"
        case .pdf: "pdf"
        case .png: "png"
        }
    }
}

/// A prepared export payload: bytes plus a suggested filename.
public struct ExportItem: Identifiable {
    public let id = UUID()
    public let title: String
    public let filename: String
    public let data: Data

    public init(title: String, filename: String, data: Data) {
        self.title = title
        self.filename = filename
        self.data = data
    }
}

/// Builds Markdown / JSON representations of a conversation.
public enum ConversationExporter {
    public static func markdown(title: String, messages: [ChatMessage]) -> String {
        var lines = ["# \(title)", ""]
        for message in messages {
            lines.append("## \(roleName(message.role))")
            lines.append("")
            lines.append(message.content)
            for call in message.toolCalls {
                lines.append("")
                lines.append("**Tool:** `\(call.toolName)`")
                if let result = call.result {
                    lines.append(result.success ? "_(success)_" : "_(failed: \(result.error ?? "unknown"))_")
                }
            }
            lines.append("")
        }
        return lines.joined(separator: "\n")
    }

    public static func json(title: String, messages: [ChatMessage]) -> String {
        let payload: [String: JSONValue] = [
            "title": .string(title),
            "messages": .array(messages.map { message in
                .object([
                    "role": .string(message.role.rawValue),
                    "content": .string(message.content),
                    "timestamp": .string(message.timestamp.ISO8601Format()),
                    "tool_calls": .array(message.toolCalls.map { call in
                        .object([
                            "tool": .string(call.toolName),
                            "success": .bool(call.result?.success ?? false),
                        ])
                    }),
                ])
            }),
        ]
        guard let data = try? JSONEncoder().encode(payload),
              let str = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return str
    }

    private static func roleName(_ role: ChatRole) -> String {
        switch role {
        case .user: "User"
        case .assistant: "Tessera Agent"
        case .system: "System"
        case .tool: "Tool"
        }
    }
}

/// Renders a quantization receipt to a single-page PDF (CoreText, cross-platform).
public enum ReceiptPDFRenderer {
    public static func pdfData(for receipt: QuantizationReceipt) -> Data {
        var lines: [String] = []
        lines.append("Schema: \(receipt.schemaVersion)")
        lines.append("")
        lines.append("Model")
        lines.append("  Name:        \(receipt.model.name)")
        lines.append("  Family:      \(receipt.model.family)")
        lines.append("  Parameters:  \(receipt.model.parameterCount)")
        lines.append("  Bits:        \(String(format: "%.1f -> %.2f", receipt.model.sourceBits, receipt.model.outputBits))")
        lines.append("")
        lines.append("Calibration")
        lines.append("  Corpus:      \(receipt.calibration.corpus)")
        lines.append("  Tokens:      \(receipt.calibration.tokenCount)")
        lines.append("  Modality:    \(receipt.calibration.modality)")
        lines.append("  Dequant:     \(receipt.calibration.dequantMode)")
        if let ga = receipt.gaArchive {
            lines.append("")
            lines.append("GA Archive")
            lines.append("  Generations: \(ga.generations)")
            lines.append("  Population:  \(ga.population)")
            lines.append("  Best fitness:\(String(format: " %.4g", ga.bestFitness))")
            lines.append("  Archive size:\(ga.archiveSize)")
        }
        if !receipt.tensors.isEmpty {
            lines.append("")
            lines.append("Per-tensor stats (\(receipt.tensors.count))")
            for tensor in receipt.tensors.prefix(40) {
                let name = String(tensor.name.prefix(28)).padding(toLength: 28, withPad: " ", startingAt: 0)
                lines.append(String(format: "  %@ %4.2fb  mse %.4g  snr %.1fdB", name, tensor.bits, tensor.mse, tensor.snrDB))
            }
        }
        return pdfData(title: "Tessera Quantization Receipt - \(receipt.model.name)", lines: lines)
    }

    static func pdfData(title: String, lines: [String]) -> Data {
        let pdfData = NSMutableData()
        guard let consumer = CGDataConsumer(data: pdfData as CFMutableData) else { return Data() }
        var mediaBox = CGRect(x: 0, y: 0, width: 612, height: 792)
        guard let ctx = CGContext(consumer: consumer, mediaBox: &mediaBox, nil) else { return Data() }

        ctx.beginPDFPage(nil)

        let body = ([title, ""] + lines).joined(separator: "\n")
        let font = CTFontCreateWithName("Menlo" as CFString, 10, nil)
        let attributes: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): font,
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(gray: 0.1, alpha: 1),
        ]
        let attrString = NSAttributedString(string: body, attributes: attributes)
        let framesetter = CTFramesetterCreateWithAttributedString(attrString)
        let textRect = CGRect(x: 40, y: 40, width: 532, height: 712)
        let path = CGPath(rect: textRect, transform: nil)
        let frame = CTFramesetterCreateFrame(framesetter, CFRange(location: 0, length: 0), path, nil)

        // CoreText draws in a bottom-left origin; flip to top-left for the page.
        ctx.textMatrix = .identity
        ctx.translateBy(x: 0, y: 792)
        ctx.scaleBy(x: 1, y: -1)
        CTFrameDraw(frame, ctx)

        ctx.endPDFPage()
        ctx.closePDF()
        return pdfData as Data
    }
}

/// Renders a metrics chart to PNG via SwiftUI's ImageRenderer.
@MainActor
public enum ChartImageRenderer {
    public static func pngData(receipt: QuantizationReceipt) -> Data? {
        let content = MetricsChartView(receipt: receipt)
            .padding(24)
            .frame(width: 720, height: 560)
            .background(Color.white)
            // The export canvas is always white; pin the chart's
            // semantic colors to their light variants so a
            // dark-mode host does not render dark-mode ink on a
            // light canvas.
            .environment(\.colorScheme, .light)
        let renderer = ImageRenderer(content: content)
        renderer.scale = 2.0
        #if canImport(AppKit)
        guard let image = renderer.nsImage,
              let tiff = image.tiffRepresentation,
              let rep = NSBitmapImageRep(data: tiff) else { return nil }
        return rep.representation(using: .png, properties: [:])
        #elseif canImport(UIKit)
        return renderer.uiImage?.pngData()
        #else
        return nil
        #endif
    }
}

/// Platform-adaptive export surface: NSSavePanel on macOS, ShareLink on iOS.
public struct ExportView: View {
    public let item: ExportItem
    @Environment(\.dismiss) private var dismiss
    @State private var status: String?

    public init(item: ExportItem) {
        self.item = item
    }

    public var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                Image(systemName: "square.and.arrow.up")
                    .font(.largeTitle)
                    .foregroundStyle(.secondary)
                Text(item.title)
                    .font(.headline)
                Text(item.filename)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                Text(ByteCountFormatter.string(fromByteCount: Int64(item.data.count), countStyle: .file))
                    .font(.caption)
                    .foregroundStyle(.tertiary)

                if let status {
                    Text(status)
                        .font(.caption)
                        .foregroundStyle(.green)
                }

                Spacer()

                exportControl
            }
            .padding()
            .navigationTitle("Export")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        #if os(macOS)
        .frame(minWidth: 360, minHeight: 300)
        #endif
    }

    @ViewBuilder
    private var exportControl: some View {
        #if os(macOS)
        Button("Save As...") { savePanel() }
            .buttonStyle(.borderedProminent)
        #elseif os(iOS)
        if let url = stagedFileURL() {
            ShareLink(item: url) {
                Label("Share", systemImage: "square.and.arrow.up")
            }
            .buttonStyle(.borderedProminent)
        }
        #endif
    }

    #if os(macOS)
    private func savePanel() {
        let panel = NSSavePanel()
        panel.canCreateDirectories = true
        panel.nameFieldStringValue = item.filename
        panel.title = "Export \(item.title)"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try item.data.write(to: url)
            status = "Saved."
        } catch {
            status = "Failed: \(error.localizedDescription)"
        }
    }
    #endif

    #if os(iOS)
    private func stagedFileURL() -> URL? {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(item.filename)
        do {
            try item.data.write(to: url, options: .atomic)
            return url
        } catch {
            return nil
        }
    }
    #endif
}
