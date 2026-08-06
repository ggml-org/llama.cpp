import Foundation

// MARK: - EmailImporter

/// The Swift-side actor that drives email imports.
/// v1 supports three input formats:
///
/// * `.eml` — a single RFC 5322 message.
/// * `.mbox` — a sequence of "From " separated
///   messages (Apple Mail's "Export Mailbox"
///   produces a .mbox file).
/// * Apple Mail "Export Mailbox" — also produces
///   `.mbox` (the user picks "Save As: Mail
///   Archive" or "Raw Message Source" in Mail's
///   File menu; both produce RFC 5322 mbox). The
///   `importAppleMailMailbox` entry point is a
///   thin alias for `importMBOX`.
///
/// The actor wraps ``TesseraImporter`` (Phase 4).
/// The Phase 4 Python CLI's `mailbox` / `email`
/// stdlib parsers do the actual work; the actor's
/// job is to:
/// 1. Hand the URL to the Phase 4 importer.
/// 2. Receive the new entity ids back.
/// 3. Re-fetch the email rows and normalize the
///    `folder` + `threadID` (the Python parser
///    already sets threadID in `meta`, but the
///    Swift normalization is the source of truth
///    so the threading helpers are unit-tested).
///
/// **Why wrap, not re-implement.** RFC 5322 is a
/// big spec (multipart bodies, encoded-word
/// headers, MIME types, character sets, ...).
/// Re-implementing in Swift is a multi-week
/// project; the Python stdlib does it correctly
/// in 200 lines. The Swift wrapper pays for the
/// subprocess cost (a few hundred ms per email
/// batch) and the Python dependency. Both are
/// well-bounded.
public actor EmailImporter {

    public enum ImportError: Error, Sendable, Equatable {
        case importerFailed(reason: String)
        case fileNotFound(path: String)
        case noEmailsExtracted
    }

    private let importer: TesseraImporter
    private let store: EmailStore
    private let mediaDir: URL?

    public init(
        importer: TesseraImporter,
        store: EmailStore,
        mediaDir: URL? = nil
    ) {
        self.importer = importer
        self.store = store
        self.mediaDir = mediaDir
    }

    // MARK: - EML

    /// Import a single .eml file. Returns the new
    /// email's id. Throws on a parse failure or a
    /// subprocess error.
    public func importEML(fileURL: URL) async throws -> [UUID] {
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            throw ImportError.fileNotFound(path: fileURL.path)
        }
        return try await runImport(urls: [fileURL])
    }

    // MARK: - MBOX

    /// Import a .mbox file. The Python parser
    /// returns one document per message; the
    /// Phase 4 CLI emits one `import_ok` event per
    /// message. The Swift side returns the list of
    /// new email ids in the order the parser
    /// emitted them.
    public func importMBOX(fileURL: URL) async throws -> [UUID] {
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            throw ImportError.fileNotFound(path: fileURL.path)
        }
        return try await runImport(urls: [fileURL])
    }

    /// Apple Mail's "Export Mailbox" produces a
    /// .mbox file. The user picks "Mail > File >
    /// Save As" with format "Raw Message Source"
    /// or uses "Mailbox > Export Mailbox…". Both
    /// produce a standard RFC 5322 mbox. This
    /// entry point is the alias for ``importMBOX``;
    /// it exists so the UI's import menu can label
    /// the entry "Apple Mail mailbox" without
    /// confusing the user.
    public func importAppleMailMailbox(fileURL: URL) async throws -> [UUID] {
        try await importMBOX(fileURL: fileURL)
    }

    // MARK: - Batch

    /// Import a batch of email files. The Python
    /// CLI handles format detection (magic bytes
    /// + extension); the Swift side just hands
    /// over the URLs. Failures are surfaced per
    /// file; the successful ids are returned.
    public func importFiles(_ urls: [URL]) async throws -> [UUID] {
        let existing = urls.filter { FileManager.default.fileExists(atPath: $0.path) }
        guard !existing.isEmpty else {
            throw ImportError.noEmailsExtracted
        }
        return try await runImport(urls: existing)
    }

    // MARK: - Internals

    private func runImport(urls: [URL]) async throws -> [UUID] {
        // The Phase 4 importer emits an `import_ok`
        // event per entity. For .eml, the entity
        // type is "email" with the email's body
        // fields in the body JSON. We hand the
        // URLs over and read the new ids back.
        let newIDs: [UUID]
        do {
            newIDs = try await importer.importDragAndDrop(urls: urls)
        } catch {
            throw ImportError.importerFailed(reason: String(describing: error))
        }
        guard !newIDs.isEmpty else {
            throw ImportError.noEmailsExtracted
        }
        // The Python parser already populated the
        // JSON body with the email fields; the
        // store's `upsert` from the Phase 4 path
        // was the one that wrote the row. The
        // Swift side re-fetches each email and
        // re-saves it to apply the `Folder` +
        // `threadID` normalization (the Python
        // parser stores the raw RFC 5322
        // message-id, not the bare id; the Swift
        // side strips the brackets).
        var out: [UUID] = []
        for id in newIDs {
            guard let email = try await store.get(id: id) else { continue }
            var normalized = email
            normalized.messageID = Threading.stripBrackets(email.messageID)
            // The Python parser set threadID in
            // body.meta but the Swift `EmailMessage`
            // re-derives it via Threading.normalize.
            // The store's upsert is idempotent.
            _ = try await store.upsert(normalized)
            out.append(id)
        }
        return out
    }
}
