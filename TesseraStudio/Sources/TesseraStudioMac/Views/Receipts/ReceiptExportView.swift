import SwiftUI
import AppKit
import UniformTypeIdentifiers
import TesseraCore

// MARK: - ReceiptExportView

/// The receipt export UI (per spec §7.5). The user
/// picks one of three formats (signed JSON / Markdown /
/// C2PA-signed document), confirms the export, and the
/// service builds the artifact and logs a `Receipt` of
/// type `export` to the chain. The export is gated by
/// the configured `EgressPolicy` and the user
/// confirmation dialog.
///
/// **macOS:** the export view is the third tab of the
/// receipts drawer. The user picks a format, the view
/// shows a confirmation dialog, and the artifact is
/// saved via `NSSavePanel`.
///
/// **iOS:** the export view is a list in the modal
/// sheet. The save uses `fileExporter`.
public struct ReceiptExportView: View {

    public let documentID: UUID
    public let documentTitle: String
    public let service: ReceiptExportService
    public let userID: UserID
    public let onExported: (ExportArtifact) -> Void

    @State private var format: ReceiptExportFormat = .signedJSON
    @State private var isExporting: Bool = false
    @State private var lastArtifact: ExportArtifact?
    @State private var errorMessage: String?
    @State private var showConfirmation: Bool = false
    @State private var showSavePanel: Bool = false

    public init(
        documentID: UUID,
        documentTitle: String,
        service: ReceiptExportService,
        userID: UserID,
        onExported: @escaping (ExportArtifact) -> Void
    ) {
        self.documentID = documentID
        self.documentTitle = documentTitle
        self.service = service
        self.userID = userID
        self.onExported = onExported
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Export receipt chain")
                .font(.system(size: 13, weight: .semibold))

            Text("Export the full receipt chain for this document. The export is signed with your device's signing key, logged as a chain entry, and saved to disk.")
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            VStack(alignment: .leading, spacing: 8) {
                ForEach(ReceiptExportFormat.allCases) { fmt in
                    formatRow(fmt)
                }
            }
            .padding(8)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(Color.secondary.opacity(0.06))
            )

            if let last = lastArtifact {
                lastExportBox(last)
            }

            if let error = errorMessage {
                Text(error)
                    .font(.system(size: 11))
                    .foregroundStyle(.red)
            }

            Spacer()

            HStack {
                Spacer()
                Button("Export…") {
                    showConfirmation = true
                }
                .buttonStyle(.borderedProminent)
                .disabled(isExporting)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .alert("Export receipt chain?", isPresented: $showConfirmation) {
            Button("Cancel", role: .cancel) {}
            Button("Export") { Task { await runExport() } }
        } message: {
            Text("This will create a \(format.displayName) for '\(documentTitle)'. The export is logged as a receipt in the chain.")
        }
    }

    // MARK: - Pieces

    private func formatRow(_ fmt: ReceiptExportFormat) -> some View {
        Button {
            format = fmt
        } label: {
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: format == fmt ? "largecircle.fill.circle" : "circle")
                    .font(.system(size: 14))
                    .foregroundStyle(format == fmt ? Color.accentColor : .secondary)
                VStack(alignment: .leading, spacing: 2) {
                    Text(fmt.displayName)
                        .font(.system(size: 12, weight: .medium))
                    Text(formatDescription(fmt))
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                }
            }
        }
        .buttonStyle(.plain)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func formatDescription(_ fmt: ReceiptExportFormat) -> String {
        switch fmt {
        case .signedJSON: return "Default. Single JSON file with full chain + ed25519 signatures + C2PA manifests."
        case .markdown: return "Human-readable Markdown. Suitable for non-technical reviewers."
        case .c2paDocument: return "The document itself, with the C2PA manifest embedded. Verifiable by any C2PA-aware tool."
        }
    }

    private func lastExportBox(_ artifact: ExportArtifact) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                Text("Exported")
                    .font(.system(size: 11, weight: .medium))
                Spacer()
            }
            Text(artifact.filename)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(.secondary)
            Text("Receipt: \(artifact.receiptID.uuidString.prefix(12))…")
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.tertiary)
        }
        .padding(8)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(Color.green.opacity(0.08))
        )
    }

    // MARK: - Export

    private func runExport() async {
        isExporting = true
        errorMessage = nil
        defer { isExporting = false }
        do {
            let artifact = try await service.export(
                documentID: documentID,
                format: format,
                documentTitle: documentTitle,
                userID: userID,
                userConfirmed: true
            )
            lastArtifact = artifact
            onExported(artifact)
            await saveArtifact(artifact)
        } catch ExportError.userDenied {
            // The user cancelled; not an error.
        } catch {
            errorMessage = "Export failed: \(error)"
        }
    }

    private func saveArtifact(_ artifact: ExportArtifact) async {
        await MainActor.run {
            let panel = NSSavePanel()
            panel.title = "Save \(artifact.format.displayName)"
            panel.nameFieldStringValue = artifact.filename
            panel.allowedContentTypes = [Self.contentType(for: artifact.format)]
            panel.canCreateDirectories = true
            let response = panel.runModal()
            if response == .OK, let url = panel.url {
                do {
                    try artifact.payload.write(to: url)
                } catch {
                    errorMessage = "Failed to write: \(error)"
                }
            }
        }
    }

    static func contentType(for format: ReceiptExportFormat) -> UTType {
        switch format {
        case .signedJSON: return .json
        case .markdown: return UTType(filenameExtension: "md") ?? .plainText
        case .c2paDocument: return .plainText
        }
    }
}
