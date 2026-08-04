import Foundation
import UniformTypeIdentifiers

/// Uniform Type Identifier for Tessera Studio workflow documents.
///
/// The custom UTI is declared in `Support/Mac/Info.plist` under
/// `UTExportedTypeDeclarations`, which makes the OS Launch Services
/// treat `.tessera-workflow` as a first-class file type (own icon in
/// Finder, Quick Look, Spotlight metadata, "Open With" integration).
/// The Info.plist also lists this UTI in `CFBundleDocumentTypes` so
/// double-clicking a workflow file launches Tessera Studio and
/// routes the file through the `WorkflowDocument` FileDocument loader.
///
/// The Swift constant here lets `WorkflowDocument` and the file
/// picker modifiers (`fileExporter` / `fileImporter`) refer to the
/// type without stringly-typed `UTType("com.tessera.workflow")`
/// lookups at every call site. Lives in `TesseraCore` (not
/// `TesseraStudioMac`) so the contract is testable from
/// `TesseraCoreTests` and reachable from the iOS surface when
/// workflows ship there.
extension UTType {
    /// The Tessera Studio workflow document type.
    ///
    /// - Identifier: `com.tessera.workflow`
    /// - Conforms to: `public.json`, `public.data`
    /// - File extension: `tessera-workflow`
    /// - MIME type: `application/vnd.tessera.workflow+json`
    public static let tesseraWorkflow = UTType(
        exportedAs: "com.tessera.workflow",
        conformingTo: .json
    )
}
