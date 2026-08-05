import Foundation

// MARK: - DocumentStore

/// The productivity surface's high-level wrapper around
/// ``TesseraDataLayer``. Owns the document ↔ data layer
/// integration: load a `DocumentAST` from a `graph_entity.body`,
/// apply mutations to the in-memory AST, persist the updated
/// AST + a new signed ``Receipt`` to the data layer.
///
/// The store is the seam between the in-memory mutation engine
/// (Phase 1) and the durable Postgres tables (Phase 1's migration
/// `0002_productivity_receipts.sql`). It enforces the hexagonal
/// boundary: no raw SQL or JSON reaches the caller; no raw
/// `TesseraDataLayer` API leaks into the productivity code that
/// sits above this wrapper.
///
/// The store is a value type because it composes with the
/// underlying `TesseraDataLayer` actor and has no state of its
/// own beyond the references it holds.
public struct DocumentStore: Sendable {

    private let dataLayer: TesseraDataLayer

    public init(dataLayer: TesseraDataLayer) {
        self.dataLayer = dataLayer
    }

    // MARK: - Load / save AST

    /// Load the document AST from the data layer. Reads
    /// `graph_entities.body` (JSONB) for the entity whose id
    /// matches `id`, decodes it into a `DocumentAST`. Returns
    /// an empty AST if the entity exists but has no body yet
    /// (a freshly-created document).
    public func loadDocument(id: UUID) async throws -> DocumentAST {
        guard let entity = try await dataLayer.getEntity(id: id) else {
            throw DocumentStoreError.documentNotFound(id: id)
        }
        guard let body = entity.body, !body.isEmpty else {
            return DocumentAST.empty
        }
        guard let data = body.data(using: .utf8) else {
            throw DocumentStoreError.invalidBody(
                documentID: id,
                reason: "body is not valid UTF-8"
            )
        }
        do {
            return try DocumentAST.from(jsonData: data)
        } catch {
            throw DocumentStoreError.invalidBody(
                documentID: id,
                reason: String(describing: error)
            )
        }
    }

    /// Persist the AST back to the data layer. Updates only the
    /// `body` column on the `graph_entity`; the entity's other
    /// columns (label, source_url, embedding) are not touched.
    /// Caller is expected to have called `apply(...)` already,
    /// so this is the final persist step.
    public func saveDocument(id: UUID, ast: DocumentAST) async throws {
        let data = try ast.jsonData()
        guard let body = String(data: data, encoding: .utf8) else {
            throw DocumentStoreError.invalidBody(
                documentID: id,
                reason: "AST is not valid UTF-8"
            )
        }
        // The data layer's upsertEntity takes the full row shape;
        // we re-read the existing entity first to preserve the
        // other columns. For now, label and source_url default to
        // empty strings if the entity doesn't exist (the caller
        // is expected to have created it via the importer path).
        let existing = try await dataLayer.getEntity(id: id)
        let label = existing?.label ?? ""
        let subtype = existing?.subtype ?? "doc"
        let entityType = existing?.entityType ?? "document"
        let sourceURL = existing?.sourceURL
        _ = try await dataLayer.upsertEntity(
            GraphEntityUpsert(
                id: id,
                entityType: entityType,
                subtype: subtype,
                label: label,
                body: body,
                sourceURL: sourceURL,
                embedding: existing?.embedding
            )
        )
    }

    // MARK: - Apply mutations

    /// Apply one mutation to a document, persist the updated AST,
    /// and append a signed receipt to the chain. The receipt's
    /// `priorReceiptID` is the most recent receipt in the chain
    /// (or `nil` for a new document).
    public func apply(
        mutation: Mutation,
        to documentID: UUID,
        actor: Actor
    ) async throws -> Receipt {
        try await applyBatch(mutations: [mutation], to: documentID, actor: actor)
    }

    /// Apply a batch of mutations as one receipt. The mutations
    /// are validated and applied in order; if any fails, none of
    /// them are persisted and the document is unchanged.
    public func applyBatch(
        mutations: [Mutation],
        to documentID: UUID,
        actor: Actor
    ) async throws -> Receipt {
        guard !mutations.isEmpty else {
            throw DocumentStoreError.emptyMutationBatch
        }

        // 1. Load the current AST.
        var ast = try await loadDocument(id: documentID)

        // 2. Apply mutations in memory, capturing the union of
        //    the per-mutation pre-mutation snapshots. The receipt
        //    needs this to be self-contained for undo.
        var engine = MutationEngine()
        var preSnapshot: [UUID: Block] = [:]
        for mutation in mutations {
            let pre = try engine.apply(mutation, to: &ast)
            for (k, v) in pre { preSnapshot[k] = v }
        }

        // 3. Sign the receipt. The signer composes with the
        //    Keychain-backed volume password (or the injected
        //    key in tests).
        let signer = ReceiptSigner()
        let contentHash = try ast.contentHash()

        // 4. Look up the prior receipt id. The most recent
        //    chain entry's receipt is the prior; nil for a
        //    fresh document.
        let priorReceiptID: UUID? = try await latestReceiptID(documentID: documentID)

        let receipt = try signer.sign(
            documentID: documentID,
            mutations: mutations,
            priorReceiptID: priorReceiptID,
            actor: actor,
            documentContentHash: contentHash,
            preMutationSnapshot: preSnapshot
        )

        // 5. Persist the updated AST and the receipt. We do the
        //    AST save first (it's the load-bearing piece; a
        //    crash here leaves the receipt un-appendable, but
        //    the AST change is durable). The receipt append
        //    comes second.
        try await saveDocument(id: documentID, ast: ast)
        try await persistReceipt(receipt)

        return receipt
    }

    // MARK: - History

    /// Return the receipt chain for a document, oldest first.
    /// Decodes each row's `payload` into a `Receipt` (so the
    /// caller gets the typed shape, not the raw `GraphReceipt`).
    public func history(
        of documentID: UUID,
        limit: Int = 100
    ) async throws -> [Receipt] {
        let chain = try await dataLayer.receiptChain(documentID: documentID, limit: limit)
        return chain.compactMap { entry in
            Self.decodeReceipt(from: entry.receipt)
        }
    }

    // MARK: - Chat queue

    /// Load the chat queue for a document. Returns an empty
    /// queue when no row exists yet.
    public func loadChatQueue(documentID: UUID) async throws -> ChatQueue {
        let json = try await dataLayer.loadChatQueue(documentID: documentID)
        guard let data = json.data(using: .utf8), !data.isEmpty else {
            return ChatQueue.empty
        }
        do {
            return try Self.jsonDecoder.decode(ChatQueue.self, from: data)
        } catch {
            // Corrupt JSON in the queue table shouldn't crash the
            // document open. Return an empty queue; the user can
            // re-add items.
            return ChatQueue.empty
        }
    }

    /// Persist the chat queue. The full queue is rewritten on
    /// every save (the chat panel's drag-to-reorder and
    /// match-and-supersede operations rewrite the whole array).
    public func saveChatQueue(_ queue: ChatQueue, documentID: UUID) async throws {
        let data = try Self.jsonEncoder.encode(queue)
        guard let json = String(data: data, encoding: .utf8) else {
            throw DocumentStoreError.invalidChatQueue(reason: "ChatQueue not UTF-8 encodable")
        }
        try await dataLayer.saveChatQueue(documentID: documentID, itemsJSON: json)
    }

    // MARK: - Internals

    /// Decode a `Receipt` from a `GraphReceipt` row. The receipt
    /// payload stores the full ``Receipt`` shape; the
    /// `graph_receipts.signature` column is the same ed25519
    /// signature. We use the row's `payload` as the source of
    /// truth and ignore the column when both are present (they
    /// are by construction).
    private static func decodeReceipt(from row: GraphReceipt) -> Receipt? {
        guard let data = try? jsonEncoder.encode(row.payload),
              let receipt = try? jsonDecoder.decode(Receipt.self, from: data) else {
            return nil
        }
        return receipt
    }

    private func persistReceipt(_ receipt: Receipt) async throws {
        let payload = try Self.encodeReceiptPayload(receipt)
        _ = try await dataLayer.appendReceiptToChain(
            documentID: receipt.documentID,
            receiptType: "document_mutation",
            payload: payload,
            signature: receipt.signature
        )
    }

    private func latestReceiptID(documentID: UUID) async throws -> UUID? {
        // Get the latest chain index, then look up that row's
        // receipt_id. The chain's receipt_id is the receipt
        // itself; the receipt's own id is the prior pointer.
        let chain = try await dataLayer.receiptChain(documentID: documentID, limit: 1)
        return chain.last?.receipt.id
    }

    private static func encodeReceiptPayload(_ receipt: Receipt) throws -> [String: JSONValue] {
        let data = try jsonEncoder.encode(receipt)
        let raw = try JSONSerialization.jsonObject(with: data, options: [])
        guard let obj = raw as? [String: Any] else {
            throw DocumentStoreError.invalidBody(
                documentID: receipt.documentID,
                reason: "encoded receipt did not round-trip to a dictionary"
            )
        }
        // Convert [String: Any] to [String: JSONValue] recursively.
        return convert(obj) as? [String: JSONValue] ?? [:]
    }

    /// Recursively convert a Foundation JSON value to a
    /// `JSONValue`. Used to round-trip the receipt payload
    /// through the `graph_receipts.payload` JSONB column.
    private static func convert(_ obj: Any) -> JSONValue {
        if obj is NSNull { return .null }
        if let v = obj as? Bool { return .bool(v) }
        if let v = obj as? Double { return .number(v) }
        if let v = obj as? Int { return .number(Double(v)) }
        if let v = obj as? String { return .string(v) }
        if let v = obj as? [Any] { return .array(v.map(convert)) }
        if let v = obj as? [String: Any] {
            var out: [String: JSONValue] = [:]
            for (k, val) in v { out[k] = convert(val) }
            return .object(out)
        }
        // Fallback: encode as a JSON string. Should not happen
        // for the receipt's known fields.
        return .string(String(describing: obj))
    }

    private static let jsonEncoder: JSONEncoder = {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .iso8601
        return e
    }()

    private static let jsonDecoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()
}

// MARK: - DocumentStoreError

public enum DocumentStoreError: Error, Sendable, Equatable {
    case documentNotFound(id: UUID)
    case invalidBody(documentID: UUID, reason: String)
    case emptyMutationBatch
    case invalidChatQueue(reason: String)
}
