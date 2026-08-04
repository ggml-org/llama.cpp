import SwiftUI
import UniformTypeIdentifiers
import TesseraCore

/// FileDocument wrapper for a workflow + its node positions.
/// The on-disk JSON is the workflow + positions sidecar; the
/// ``Workflow`` struct itself is unchanged (so a hand-built
/// workflow JSON without positions still loads).
///
/// JSON shape (schema `tessera.workflow.document.v1`):
/// ```json
/// {
///   "schema": "tessera.workflow.document.v1",
///   "workflow": { ...standard Workflow Codable... },
///   "positions": { "calib": { "x": 220, "y": 220 }, ... }
/// }
/// ```
///
/// New fields are additive (default values); the executor +
/// editor both read the standard ``Workflow`` block.
///
/// The on-disk file is registered as the custom `com.tessera.workflow`
/// UTI (declared in `Support/Mac/Info.plist`; Swift constant
/// ``UTType.tesseraWorkflow``). The bytes are still JSON, so the
/// UTI conforms to `public.json` — Launch Services treats the
/// `.tessera-workflow` extension as a first-class Tessera file
/// type and SwiftUI's file pickers filter to it automatically.
/// `Equatable` so the editor derives "edited" by comparing the
/// live document against the last saved snapshot - no separate
/// dirty flag that could drift from the real content.
struct WorkflowDocument: FileDocument, Equatable {
    static let currentSchema = "tessera.workflow.document.v1"

    static var readableContentTypes: [UTType] {
        [.tesseraWorkflow]
    }

    static var writableContentTypes: [UTType] {
        [.tesseraWorkflow]
    }

    var workflow: Workflow
    var positions: WorkflowPositionMap

    init(workflow: Workflow, positions: WorkflowPositionMap = [:]) {
        self.workflow = workflow
        self.positions = positions
    }

    init(configuration: ReadConfiguration) throws {
        guard let data = configuration.file.regularFileContents else {
            throw CocoaError(.fileReadCorruptFile)
        }
        let envelope = try JSONDecoder().decode(Envelope.self, from: data)
        self.workflow = envelope.workflow
        self.positions = envelope.positions ?? [:]
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        let envelope = Envelope(
            schema: WorkflowDocument.currentSchema,
            workflow: workflow,
            positions: positions.isEmpty ? nil : positions
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(envelope)
        return FileWrapper(regularFileWithContents: data)
    }

    /// The on-disk JSON envelope. The `positions` field is
    /// optional so a hand-built workflow JSON without
    /// positions (just the standard Workflow block) still
    /// round-trips through the document loader. Exposed
    /// publicly so `WorkflowsView.loadDocument` can decode
    /// the envelope directly when reading from a URL.
    struct Envelope: Codable {
        let schema: String
        let workflow: Workflow
        let positions: WorkflowPositionMap?

        init(
            schema: String = WorkflowDocument.currentSchema,
            workflow: Workflow,
            positions: WorkflowPositionMap?
        ) {
            self.schema = schema
            self.workflow = workflow
            self.positions = positions
        }
    }
}
