import Foundation

// MARK: - CodeStore

/// The code-specific seam between the file-system
/// watcher / agent / SwiftUI view and the data layer
/// facade (``TesseraDataLayer``). Mirrors
/// ``ContactStore``'s role: own the
/// CodeFile <-> data layer integration, enforce the
/// constitutional receipt invariant for every mutation,
/// expose a domain-shaped API to the rest of the
/// productivity surface.
///
/// **Receipt contract.** every mutation that changes
/// code state (import, modify, delete, rename, tag,
/// link) appends a signed receipt to `graph_receipts`
/// with a `receiptType` from ``CodeReceiptType`` and a
/// payload that names the affected `entity_id` plus a
/// structural summary. The agent's `hybrid_search`
/// over `entity_type = 'code'` returns these receipts
/// when assembling context.
///
/// **No Postgres in v1 unit tests.** The store has an
/// `init(dataLayer:)` that takes a real
/// ``TesseraDataLayer`` and an `init()` that uses a
/// no-op data layer for tests. The `apply` and `query`
/// methods work against an in-memory file index; the
/// data layer is the durable mirror. Tests of the pure
/// logic (mutation engine, receipt shape) don't need a
/// database.
///
/// **Why a class, not a struct.** The in-memory
/// `index` + `pathToID` are mutable and the methods
/// that mutate them are `async` (they await on the
/// data layer for the receipt write). A class with
/// internal locking is the standard pattern; the lock
/// is held only across the in-memory mutation, not
/// across the `await` (the data layer is its own
/// actor with its own serialization).
public final class CodeStore: @unchecked Sendable {

    private let dataLayer: TesseraDataLayer?
    private let lock = NSLock()

    /// The in-memory file index. The store keeps a
    /// snapshot of the `CodeFile`s the user has open +
    /// the watcher has imported; queries (search, list
    /// by path, list by language) hit the index, not
    /// the data layer. The index is rebuilt from the
    /// data layer on `loadAll()`.
    private var index: [UUID: CodeFile]
    /// A `path -> id` lookup. The store's `get(path:)`
    /// uses this; the `apply` path uses it to detect
    /// "is this an update or an insert?".
    private var pathToID: [String: UUID]

    /// Construct a store backed by the data layer. The
    /// in-memory index is empty until `loadAll()` is
    /// called.
    public init(dataLayer: TesseraDataLayer) {
        self.dataLayer = dataLayer
        self.index = [:]
        self.pathToID = [:]
    }

    /// Construct a store without a data layer. Used by
    /// unit tests that exercise the mutation engine
    /// and the in-memory index; the `apply` path
    /// short-circuits the receipt write when the data
    /// layer is nil.
    public init() {
        self.dataLayer = nil
        self.index = [:]
        self.pathToID = [:]
    }

    // MARK: - Import / upsert

    /// Insert or update a file by path. If the path is
    /// already in the index, the file is replaced
    /// (the watcher emits `.modified` for changes;
    /// the import path emits `.created` once and
    /// `.modified` after). The receipt type is
    /// `code_file_imported` for new files and
    /// `code_file_modified` for existing files.
    public func upsert(_ file: CodeFile) async throws -> CodeFile {
        let isNew: Bool = lock.withLock { pathToID[file.path] == nil }
        lock.withLock {
            index[file.id] = file
            pathToID[file.path] = file.id
        }
        let body = try file.jsonDataString()
        if let dataLayer {
            _ = try await dataLayer.upsertEntity(
                GraphEntityUpsert(
                    id: file.id,
                    entityType: CodeFile.entityType,
                    subtype: file.subtypeString,
                    label: file.filename,
                    body: body,
                    sourceURL: file.path,
                    embedding: nil
                )
            )
        }
        let receiptType: CodeReceiptType = isNew ? .imported : .modified
        let payload: [String: JSONValue] = [
            "path": .string(file.path),
            "language": .string(file.language),
            "size": .number(Double(file.size)),
            "checksum": .string(file.checksum),
        ]
        try await appendReceipt(
            entityID: file.id,
            receiptType: receiptType.rawValue,
            payload: payload
        )
        return file
    }

    /// Get a file by id.
    public func get(id: UUID) -> CodeFile? {
        return lock.withLock { index[id] }
    }

    /// Get a file by path.
    public func get(path: String) -> CodeFile? {
        return lock.withLock {
            guard let id = pathToID[path] else { return nil }
            return index[id]
        }
    }

    /// All files in the index, sorted by path.
    public func listAll() -> [CodeFile] {
        return lock.withLock {
            index.values.sorted { $0.path < $1.path }
        }
    }

    /// All files whose language matches `language`.
    public func list(language: String) -> [CodeFile] {
        return lock.withLock {
            index.values
                .filter { $0.language == language }
                .sorted { $0.path < $1.path }
        }
    }

    /// All files whose path or body matches `query`
    /// (case-insensitive substring). Used by the
    /// sidebar's quick-filter box.
    public func search(_ query: String) -> [CodeFile] {
        let q = query.lowercased()
        guard !q.isEmpty else { return listAll() }
        return lock.withLock {
            index.values
                .filter {
                    $0.path.lowercased().contains(q) ||
                    $0.filename.lowercased().contains(q) ||
                    $0.body.lowercased().contains(q.prefix(1000).description)
                }
                .sorted { $0.path < $1.path }
        }
    }

    // MARK: - Mutation

    /// Apply a `CodeMutation` to the file. The engine
    /// produces a `CodeMutationApplyResult`; the store
    /// persists the updated file, writes the receipt,
    /// and returns the pre-mutation snapshot (so the
    /// caller can build the undo receipt).
    public func apply(
        _ mutation: CodeMutation,
        to fileID: UUID
    ) async throws -> CodeMutationApplyResult {
        let file: CodeFile? = lock.withLock { index[fileID] }
        guard let file else {
            throw CodeStoreError.fileNotFound(id: fileID)
        }
        let engine = CodeMutationEngine()
        let result = try engine.apply(mutation, to: file)
        // Update the in-memory index.
        lock.withLock {
            index[result.updated.id] = result.updated
            pathToID[result.updated.path] = result.updated.id
        }
        // Persist + receipt.
        let body = try result.updated.jsonDataString()
        if let dataLayer {
            _ = try await dataLayer.upsertEntity(
                GraphEntityUpsert(
                    id: result.updated.id,
                    entityType: CodeFile.entityType,
                    subtype: result.updated.subtypeString,
                    label: result.updated.filename,
                    body: body,
                    sourceURL: result.updated.path,
                    embedding: nil
                )
            )
        }
        let payload: [String: JSONValue] = [
            "path": .string(result.updated.path),
            "summary": .string(mutation.shortDescription),
            "preChecksum": .string(file.checksum),
            "postChecksum": .string(result.updated.checksum),
            "preSize": .number(Double(file.size)),
            "postSize": .number(Double(result.updated.size)),
        ]
        try await appendReceipt(
            entityID: result.updated.id,
            receiptType: mutation.receiptType,
            payload: payload
        )
        return result
    }

    // MARK: - Delete / rename

    /// Delete a file by id. The receipt is
    /// `code_file_deleted`; the data layer's row is
    /// removed. The receipt chain is preserved (the
    /// row count on the receipts table doesn't
    /// decrease; receipts are append-only).
    public func delete(id: UUID) async throws {
        let file: CodeFile? = lock.withLock { index.removeValue(forKey: id) }
        guard let file else {
            throw CodeStoreError.fileNotFound(id: id)
        }
        lock.withLock { pathToID.removeValue(forKey: file.path) }
        if let dataLayer {
            _ = try await dataLayer.deleteEntity(id: id)
        }
        try await appendReceipt(
            entityID: id,
            receiptType: CodeReceiptType.deleted.rawValue,
            payload: [
                "path": .string(file.path),
                "filename": .string(file.filename),
            ]
        )
    }

    /// Rename a file. The id is stable; the path
    /// changes. The data layer's `sourceURL` is
    /// updated; the receipt is `code_file_renamed`.
    public func rename(id: UUID, to newPath: String) async throws -> CodeFile {
        let existingFile: CodeFile? = lock.withLock { index[id] }
        guard var file = existingFile else {
            throw CodeStoreError.fileNotFound(id: id)
        }
        let oldPath = file.path
        lock.withLock {
            pathToID.removeValue(forKey: oldPath)
        }
        file.path = newPath
        file.filename = CodeFile.filenameFromPath(newPath)
        file.updatedAt = Date()
        lock.withLock {
            index[id] = file
            pathToID[newPath] = id
        }
        let body = try file.jsonDataString()
        if let dataLayer {
            _ = try await dataLayer.upsertEntity(
                GraphEntityUpsert(
                    id: id,
                    entityType: CodeFile.entityType,
                    subtype: file.subtypeString,
                    label: file.filename,
                    body: body,
                    sourceURL: newPath,
                    embedding: nil
                )
            )
        }
        try await appendReceipt(
            entityID: id,
            receiptType: CodeReceiptType.renamed.rawValue,
            payload: [
                "oldPath": .string(oldPath),
                "newPath": .string(newPath),
            ]
        )
        return file
    }

    // MARK: - Linking

    /// Link the file to another graph entity. Creates
    /// an `entity_link` row + a `code_file_linked`
    /// receipt. The link is a no-op if a row with the
    /// same (source, target, type) tuple already
    /// exists.
    @discardableResult
    public func link(
        _ fileID: UUID,
        to otherEntityID: UUID,
        linkType: String = "related_to",
        weight: Float = 1.0
    ) async throws -> EntityLink? {
        let fileExists: Bool = lock.withLock { index[fileID] != nil }
        guard fileExists else {
            throw CodeStoreError.fileNotFound(id: fileID)
        }
        guard let dataLayer else { return nil }
        let link = try await dataLayer.linkEntities(
            sourceID: fileID,
            targetID: otherEntityID,
            linkType: linkType,
            weight: weight
        )
        try await appendReceipt(
            entityID: fileID,
            receiptType: CodeReceiptType.linked.rawValue,
            payload: [
                "targetEntityID": .string(otherEntityID.uuidString),
                "linkType": .string(linkType),
            ]
        )
        return link
    }

    // MARK: - Receipts

    /// All constitutional receipts for a file, oldest
    /// first. The receipt chain is the audit trail the
    /// user sees in the receipt drawer.
    public func receipts(forFile fileID: UUID) async throws -> [GraphReceipt] {
        guard let dataLayer else { return [] }
        return try await dataLayer.receipts(forEntity: fileID)
    }

    // MARK: - Load

    /// Load every `code` entity from the data layer
    /// into the in-memory index. Called once at app
    /// start; the watcher + chat panel keep the index
    /// in sync after.
    public func loadAll() async throws {
        guard let dataLayer else { return }
        let rows = try await dataLayer.listByEntityType(
            entityType: CodeFile.entityType, limit: 10_000
        )
        var newIndex: [UUID: CodeFile] = [:]
        var newPathToID: [String: UUID] = [:]
        for row in rows {
            guard let body = row.body, !body.isEmpty else { continue }
            guard let file = try? CodeFile.from(jsonDataString: body) else { continue }
            newIndex[file.id] = file
            newPathToID[file.path] = file.id
        }
        lock.withLock {
            self.index = newIndex
            self.pathToID = newPathToID
        }
    }

    // MARK: - Helpers

    private func appendReceipt(
        entityID: UUID,
        receiptType: String,
        payload: [String: JSONValue]
    ) async throws {
        guard let dataLayer else { return }
        _ = try await dataLayer.appendReceipt(
            entityID: entityID,
            receiptType: receiptType,
            payload: payload
        )
    }
}

// MARK: - Errors

public enum CodeStoreError: Error, Sendable, Equatable {
    case fileNotFound(id: UUID)
    case invalidBody(reason: String)
}

// MARK: - NSLock helper

private extension NSLock {
    /// `lock.withLock { ... }` is in the stdlib in
    /// Swift 5.9+; this extension is the in-tree
    /// polyfill for older toolchains. The function
    /// takes a closure that returns a value; the
    /// lock is held for the duration of the
    /// closure's execution.
    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock()
        defer { unlock() }
        return try body()
    }
}
