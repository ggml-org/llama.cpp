import Foundation

// MARK: - ProductivityExportFormat

/// The set of formats the productivity exporter can produce.
///
/// Mirrors the Python side's ``SUPPORTED_FORMATS`` set; the
/// strings are the same as the CLI's ``--format`` argument.
/// ``rawValue`` is the wire name used by the HTTP API and the
/// Python subprocess; ``displayName`` is the human-readable
/// form for the share-sheet UI.
///
/// Note: the conversation exporter (``ExportView.swift``) has
/// its own ``ExportFormat`` enum (markdown / json / pdf / png).
/// We deliberately don't unify the two: the conversation
/// exporter is a single-source helper for the chat panel,
/// the productivity exporter is the multi-format pipeline
/// from the spec's §11. Both have the same name in spirit
/// (a file format the user can export to) but their
/// format sets are disjoint.
public enum ProductivityExportFormat: String, Codable, Sendable, CaseIterable, Hashable {
    case pdf
    case docx
    case xlsx
    case pptx
    case html
    case md
    case eml

    public var displayName: String {
        switch self {
        case .pdf: return "PDF"
        case .docx: return "Word Document"
        case .xlsx: return "Excel Spreadsheet"
        case .pptx: return "PowerPoint"
        case .html: return "HTML"
        case .md: return "Markdown"
        case .eml: return "Email (.eml)"
        }
    }

    /// The file extension for the format.
    public var fileExtension: String { rawValue }
}

// MARK: - ShareTarget

/// A single share target the user can pick from the share sheet.
///
/// ``ShareTarget`` is the v1 abstraction over both
/// ``NSSharingServicePicker`` (which wraps the system share
/// sheet) and a custom in-app target (like the Slack webhook).
/// Each target carries:

/// * a display name and icon (the system share sheet uses
///   these for the picker),
/// * the formats it accepts (the share sheet filters the
///   available targets by what the document can be exported
///   to),
/// * a handler that runs when the user picks the target.
///
/// The handler receives the exported file's URL; the target
/// is responsible for whatever it does with the file
/// (uploading, opening in another app, posting to a webhook,
/// etc.).
public struct ShareTarget: Identifiable, Sendable, Hashable {
    public let id: String
    public var name: String
    public var icon: Data?
    public var accepts: Set<ProductivityExportFormat>
    public var handler: @Sendable (URL) async throws -> Void

    public init(
        id: String,
        name: String,
        icon: Data? = nil,
        accepts: Set<ProductivityExportFormat>,
        handler: @escaping @Sendable (URL) async throws -> Void
    ) {
        self.id = id
        self.name = name
        self.icon = icon
        self.accepts = accepts
        self.handler = handler
    }

    public static func == (lhs: ShareTarget, rhs: ShareTarget) -> Bool {
        lhs.id == rhs.id && lhs.name == rhs.name
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
        hasher.combine(name)
    }
}

// MARK: - TesseraExporter

/// The Swift-side actor that runs the Python exporter pipeline.
///
/// Mirrors ``TesseraImporter``: it spawns a Python subprocess
/// to do the actual file writing, then emits a ``graph_receipt``
/// via the data layer. The actor is the seam between the
/// system share sheet (which calls into the actor with a
/// target URL) and the Python exporter (which writes the
/// file).
///
/// The actor also exposes ``export(entityID:shareWith:)``,
/// the system share sheet entry point: it determines the
/// available share targets, exports the document to the
/// target's preferred format, and hands the file to the
/// target's handler.
public actor TesseraExporter {
    private let runner: PythonSubprocessRunner
    private let env: [String: String]
    private let outputDir: URL
    private let importer: TesseraImporter

    public init(
        pythonExecutable: URL? = nil,
        env: [String: String] = [:],
        outputDir: URL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("tessera-exports", isDirectory: true),
        importer: TesseraImporter? = nil
    ) {
        self.runner = PythonSubprocessRunner()
        self.env = env
        self.outputDir = outputDir
        self.importer = importer ?? TesseraImporter(env: env)
        // Ensure the output dir exists for the first call.
        try? FileManager.default.createDirectory(
            at: outputDir, withIntermediateDirectories: true
        )
    }

    /// Export an entity to the given format and write it to
    /// ``outputURL``. The Python CLI writes the file to
    /// ``outputURL``; the data layer gets a ``receipt_type =
    /// "export"`` receipt appended.
    public func export(
        entityID: UUID,
        to format: ProductivityExportFormat,
        outputURL: URL
    ) async throws {
        var args: [String] = [
            "export",
            entityID.uuidString,
            "--format", format.rawValue,
            "--output", outputURL.path,
        ]
        args += ["--output-dir", outputDir.path]
        let result = try runner.run(
            script: TesseraCLIPath.exporterScript,
            args: args,
            env: env
        )
        if result.exitCode != 0 {
            throw TesseraExporterError.subprocessFailed(
                exitCode: result.exitCode,
                stderr: result.stderr
            )
        }
        // The CLI's stdout is a JSON event stream; we
        // surface only the "export_ok" / "summary" events.
        // The error path is the exit code.
    }

    /// System share sheet entry point. Exports the document
    /// to a format the target accepts and hands the file URL
    /// to the target's handler. The default target picker
    /// filters the available targets by what the document
    /// can be exported to.
    public func export(
        entityID: UUID,
        shareWith target: ShareTarget
    ) async throws {
        guard let format = target.accepts.first else {
            throw TesseraExporterError.targetRefusedFormat
        }
        // Stage the file in the output dir with a stable name
        // (entity id + extension) so the share sheet sees a
        // fresh file even if the user re-exports the same
        // document.
        let staged = outputDir.appendingPathComponent(
            "\(entityID.uuidString).\(format.fileExtension)"
        )
        try await export(entityID: entityID, to: format, outputURL: staged)
        try await target.handler(staged)
    }
}

// MARK: - Errors

public enum TesseraExporterError: Error, LocalizedError {
    case subprocessFailed(exitCode: Int32, stderr: String)
    case targetRefusedFormat
    case exportFailed(reason: String)

    public var errorDescription: String? {
        switch self {
        case .subprocessFailed(let code, let stderr):
            return "Export: subprocess exited with code \(code): \(stderr)"
        case .targetRefusedFormat:
            return "Export: the share target does not accept any of the entity's formats"
        case .exportFailed(let reason):
            return "Export failed: \(reason)"
        }
    }
}
