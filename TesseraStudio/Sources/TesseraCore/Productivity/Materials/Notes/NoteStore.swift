import Foundation

// MARK: - NoteStore

/// The note-specific seam between importers / the SwiftUI view /
/// the chat-panel agent and the data layer facade
/// (``TesseraDataLayer``). Mirrors ``ContactStore``'s role for the
/// contact material type: own the note ↔ data layer integration,
/// enforce the constitutional-receipt invariant for every mutation,
/// expose a domain-shaped API to the rest of the productivity
/// surface.
///
/// The store is a struct because the data layer actor is the
/// source of concurrency safety. There's no mutable state of our
/// own.
///
/// **Receipt contract:** every mutation that changes note state
/// (create, update body, pin, unpin, archive, unarchive, delete,
/// tag, link, unlink) appends a signed receipt to the
/// `graph_receipts` table with a `receipt_type` from
/// ``NoteReceiptType`` and a payload that names the affected
/// `entity_id` plus a structural summary. The agent's
/// `hybrid_search` over `entity_type = 'note'` returns these
/// receipts when assembling context.
///
/// **Why a separate store?** The note mutation vocabulary
/// (pin, archive, tag, link) is note-specific; the existing
/// ``DocumentStore`` covers the Block AST mutations (insert,
/// delete, set content, set attribute). Phase 5 doesn't need
/// every note mutation to go through ``DocumentStore/apply``
/// (which is a Block-AST layer); a note is also a
/// `graph_entity` row with its own metadata, and that metadata
/// has its own receipt vocabulary.
public struct NoteStore: Sendable {

    private let dataLayer: TesseraDataLayer

    public init(dataLayer: TesseraDataLayer) {
        self.dataLayer = dataLayer
    }

    // MARK: - CRUD

    /// Persist a note. If the note's id already exists in the
    /// `graph_entities` table the row is updated in place;
    /// otherwise a new row is inserted. Always appends a
    /// `note_upsert` receipt so the audit trail captures the
    /// change.
    @discardableResult
    public func upsert(_ note: Note) async throws -> Note {
        let body = try note.jsonDataString()
        let label = note.displayTitle
        _ = try await dataLayer.upsertEntity(
            GraphEntityUpsert(
                id: note.id,
                entityType: Note.entityType,
                subtype: Note.subtype,
                label: label,
                body: body,
                sourceURL: nil,
                embedding: nil
            )
        )
        try await appendReceipt(
            entityID: note.id,
            receiptType: NoteReceiptType.upsert.rawValue,
            payload: [
                "title": .string(label),
                "tagCount": .number(Double(note.tags.count)),
                "pinned": .bool(note.isPinned),
                "archived": .bool(note.isArchived),
                "linkedEntityCount": .number(Double(note.linkedEntityIDs.count)),
            ]
        )
        return note
    }

    /// Fetch one note by id. Returns nil when no
    /// `graph_entity` row with `entity_type = 'note'` matches.
    public func get(id: UUID) async throws -> Note? {
        guard let entity = try await dataLayer.getEntity(id: id) else {
            return nil
        }
        guard entity.entityType == Note.entityType else {
            return nil
        }
        return try Self.noteFromEntity(entity)
    }

    /// Delete a note by id. Returns true when a row was removed.
    /// The deletion cascades to `entity_links` and `graph_receipts`
    /// via the foreign keys; the receipt chain for the note is
    /// preserved (the row count on the receipts table doesn't
    /// decrease — receipts are append-only, and the note row is
    /// the only thing deleted).
    @discardableResult
    public func delete(id: UUID) async throws -> Bool {
        let didDelete = try await dataLayer.deleteEntity(id: id)
        if didDelete {
            try await appendReceipt(
                entityID: id,
                receiptType: NoteReceiptType.delete.rawValue,
                payload: [:]
            )
        }
        return didDelete
    }

    // MARK: - Listing

    /// Every note, newest updated first. Excludes nothing — the
    /// caller (the notes view's "All" filter) shows pinned +
    /// archived mixed in. The data layer's `listByEntityType`
    /// orders by `updated_at DESC` then `label`; the index from
    /// migration `0007_notes.sql` keeps the lookup O(log n) even
    /// at 10k+ notes.
    public func list(limit: Int = 1000) async throws -> [Note] {
        let rows = try await dataLayer.listByEntityType(
            entityType: Note.entityType,
            limit: limit
        )
        return try rows.compactMap { try? Self.noteFromEntity($0) }
    }

    /// Pinned notes only. The view's Pinned tab calls this. The
    /// in-memory filter is cheap (we always fetch the full list
    /// and partition in code); a Postgres `WHERE pinned_at IS
    /// NOT NULL` filter is a v2 optimization if the list grows
    /// past a few thousand notes.
    public func pinned(limit: Int = 1000) async throws -> [Note] {
        let all = try await list(limit: limit)
        return all.filter { $0.isPinned && !$0.isArchived }
            .sorted { (lhs, rhs) in
                // Most recently pinned first; fall back to
                // updated_at for notes that share a pin time.
                let l = lhs.pinnedAt ?? .distantPast
                let r = rhs.pinnedAt ?? .distantPast
                if l != r { return l > r }
                return lhs.updatedAt > rhs.updatedAt
            }
    }

    /// Archived notes only. The view's Archived tab calls this.
    /// Most-recently-archived first.
    public func archived(limit: Int = 1000) async throws -> [Note] {
        let all = try await list(limit: limit)
        return all.filter { $0.isArchived }
            .sorted { (lhs, rhs) in
                let l = lhs.archivedAt ?? .distantPast
                let r = rhs.archivedAt ?? .distantPast
                if l != r { return l > r }
                return lhs.updatedAt > rhs.updatedAt
            }
    }

    // MARK: - Search

    /// Find notes whose display title matches `query`. The
    /// implementation uses the data layer's `searchByLabelPrefix`
    /// against the contact-style index — same pattern as
    /// ``ContactStore/search(matching:)``. Case-insensitive.
    public func search(matching query: String, limit: Int = 20) async throws -> [Note] {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !trimmed.isEmpty else { return [] }
        let rows = try await dataLayer.searchByLabelPrefix(
            entityType: Note.entityType,
            labelPrefix: trimmed,
            limit: limit
        )
        return try rows.compactMap { try? Self.noteFromEntity($0) }
    }

    // MARK: - Mutations (note-level)

    /// Set the title. The `updatedAt` is bumped. The receipt
    /// payload includes the new title and the old title (when
    /// known — `oldNote == nil` means we're just persisting a
    /// brand-new title without a prior snapshot, which is fine
    /// for the chat-panel's "create a note titled X" call).
    public func setTitle(
        _ title: String,
        for noteID: UUID,
        oldNote: Note? = nil
    ) async throws -> Note {
        var note = try await loadOrFail(id: noteID)
        let oldTitle = note.title
        note.title = title
        note.updatedAt = Date()
        _ = try await upsert(note)
        try await appendReceipt(
            entityID: noteID,
            receiptType: NoteReceiptType.titleChanged.rawValue,
            payload: [
                "newTitle": .string(title),
                "oldTitle": .string(oldTitle),
            ]
        )
        return note
    }

    /// Replace the body AST. The editor calls this on every
    /// commit (the AST-level mutations go through
    /// ``DocumentStore/apply``; this is the note-level wrapper
    /// that bumps `updatedAt` and emits the receipt).
    public func setBody(
        _ body: DocumentAST,
        for noteID: UUID
    ) async throws -> Note {
        var note = try await loadOrFail(id: noteID)
        // Auto-derive the title from the first heading if the
        // user hasn't set one explicitly (the field is empty
        // and the body's first heading is non-empty). The
        // editor shows the title in the bar above the body, so
        // keeping it in sync is the editor's job too; this is
        // a safety net.
        if note.title.isEmpty, let first = Note.firstHeadingText(in: body) {
            note.title = first
        }
        note.body = body
        note.updatedAt = Date()
        _ = try await upsert(note)
        try await appendReceipt(
            entityID: noteID,
            receiptType: NoteReceiptType.bodyChanged.rawValue,
            payload: [
                "blockCount": .number(Double(body.blocks.count)),
                "rootChildCount": .number(Double(body.rootChildren.count)),
            ]
        )
        return note
    }

    /// Pin a note. Idempotent — pinning a pinned note is a no-op
    /// for the data layer, but we still append a receipt so the
    /// audit trail captures the intent. The receipt payload
    /// includes whether the pin was a no-op (so the user can
    /// tell which pin calls were "real" actions).
    public func pin(_ noteID: UUID) async throws -> Note {
        var note = try await loadOrFail(id: noteID)
        let wasPinned = note.isPinned
        if !wasPinned {
            note.pinnedAt = Date()
            note.updatedAt = Date()
            _ = try await upsert(note)
        }
        try await appendReceipt(
            entityID: noteID,
            receiptType: NoteReceiptType.pinned.rawValue,
            payload: [
                "wasAlreadyPinned": .bool(wasPinned),
            ]
        )
        return note
    }

    /// Unpin a note. Idempotent — unpinning an unpinned note is
    /// a no-op for the data layer, but we still append a receipt
    /// so the audit trail captures the intent.
    public func unpin(_ noteID: UUID) async throws -> Note {
        var note = try await loadOrFail(id: noteID)
        let wasPinned = note.isPinned
        if wasPinned {
            note.pinnedAt = nil
            note.updatedAt = Date()
            _ = try await upsert(note)
        }
        try await appendReceipt(
            entityID: noteID,
            receiptType: NoteReceiptType.unpinned.rawValue,
            payload: [
                "wasPinned": .bool(wasPinned),
            ]
        )
        return note
    }

    /// Archive a note. Idempotent — archiving an archived note
    /// is a no-op. The Archived list filters on `archivedAt !=
    /// nil`; pinning an archived note is preserved (the pin
    /// state stays across archive/restore, matching Bear's
    /// behavior).
    public func archive(_ noteID: UUID) async throws -> Note {
        var note = try await loadOrFail(id: noteID)
        let wasArchived = note.isArchived
        if !wasArchived {
            note.archivedAt = Date()
            note.updatedAt = Date()
            _ = try await upsert(note)
        }
        try await appendReceipt(
            entityID: noteID,
            receiptType: NoteReceiptType.archived.rawValue,
            payload: [
                "wasAlreadyArchived": .bool(wasArchived),
            ]
        )
        return note
    }

    /// Unarchive a note. Idempotent.
    public func unarchive(_ noteID: UUID) async throws -> Note {
        var note = try await loadOrFail(id: noteID)
        let wasArchived = note.isArchived
        if wasArchived {
            note.archivedAt = nil
            note.updatedAt = Date()
            _ = try await upsert(note)
        }
        try await appendReceipt(
            entityID: noteID,
            receiptType: NoteReceiptType.unarchived.rawValue,
            payload: [
                "wasArchived": .bool(wasArchived),
            ]
        )
        return note
    }

    /// Set the full tag list. The list is normalized (lowercased,
    /// trimmed, de-duplicated) before persisting. The receipt
    /// payload carries the old and new tag sets so the audit
    /// trail can answer "what tags were added/removed".
    public func setTags(
        _ tags: [String],
        for noteID: UUID
    ) async throws -> Note {
        var note = try await loadOrFail(id: noteID)
        let oldTags = note.tags
        let normalized = Note.normalizeTags(tags)
        note.tags = normalized
        note.updatedAt = Date()
        _ = try await upsert(note)
        let added = normalized.filter { !oldTags.contains($0) }
        let removed = oldTags.filter { !normalized.contains($0) }
        try await appendReceipt(
            entityID: noteID,
            receiptType: NoteReceiptType.tagsChanged.rawValue,
            payload: [
                "addedTags": .array(added.map { .string($0) }),
                "removedTags": .array(removed.map { .string($0) }),
            ]
        )
        return note
    }

    /// Add a single tag. Convenience for the chat-panel's "add a
    /// tag 'q3-2026' to this note" call. Idempotent — adding a
    /// tag that's already present is a no-op (the receipt
    /// payload marks it as a no-op so the audit trail is
    /// honest).
    public func addTag(
        _ tag: String,
        to noteID: UUID
    ) async throws -> Note {
        let normalized = Note.normalizeTags([tag])
        guard let first = normalized.first else { return try await loadOrFail(id: noteID) }
        var note = try await loadOrFail(id: noteID)
        if note.tags.contains(first) {
            try await appendReceipt(
                entityID: noteID,
                receiptType: NoteReceiptType.tagAdded.rawValue,
                payload: [
                    "tag": .string(first),
                    "wasAlreadyPresent": .bool(true),
                ]
            )
            return note
        }
        note.tags.append(first)
        note.updatedAt = Date()
        _ = try await upsert(note)
        try await appendReceipt(
            entityID: noteID,
            receiptType: NoteReceiptType.tagAdded.rawValue,
            payload: [
                "tag": .string(first),
                "wasAlreadyPresent": .bool(false),
            ]
        )
        return note
    }

    /// Remove a single tag. Idempotent — removing a tag that
    /// isn't present is a no-op (the receipt payload marks it as
    /// a no-op).
    public func removeTag(
        _ tag: String,
        from noteID: UUID
    ) async throws -> Note {
        let normalized = Note.normalizeTags([tag])
        guard let first = normalized.first else { return try await loadOrFail(id: noteID) }
        var note = try await loadOrFail(id: noteID)
        guard let index = note.tags.firstIndex(of: first) else {
            try await appendReceipt(
                entityID: noteID,
                receiptType: NoteReceiptType.tagRemoved.rawValue,
                payload: [
                    "tag": .string(first),
                    "wasPresent": .bool(false),
                ]
            )
            return note
        }
        note.tags.remove(at: index)
        note.updatedAt = Date()
        _ = try await upsert(note)
        try await appendReceipt(
            entityID: noteID,
            receiptType: NoteReceiptType.tagRemoved.rawValue,
            payload: [
                "tag": .string(first),
                "wasPresent": .bool(true),
            ]
        )
        return note
    }

    // MARK: - Linking

    /// Link a note to another graph entity (a document, contact,
    /// calendar event, task, or other note). Creates an
    /// `entity_link` row AND appends a `note_link_created`
    /// receipt so the audit trail captures the relationship.
    /// The link is a no-op if a row with the same (source,
    /// target, type) tuple already exists (the data layer's
    /// unique constraint handles this). The note's
    /// `linkedEntityIDs` is kept in sync with the link table —
    /// we re-read the note and update the denormalized list
    /// after the link call.
    @discardableResult
    public func link(
        noteID: UUID,
        to otherEntityID: UUID,
        linkType: String = "related_to",
        weight: Float = 1.0
    ) async throws -> EntityLink {
        let link = try await dataLayer.linkEntities(
            sourceID: noteID,
            targetID: otherEntityID,
            linkType: linkType,
            weight: weight
        )
        // Re-read the note and append the linked id if it's
        // not already in the denormalized list.
        if var note = try await get(id: noteID),
           !note.linkedEntityIDs.contains(otherEntityID) {
            note.linkedEntityIDs.append(otherEntityID)
            note.updatedAt = Date()
            _ = try await dataLayer.upsertEntity(
                GraphEntityUpsert(
                    id: note.id,
                    entityType: Note.entityType,
                    subtype: Note.subtype,
                    label: note.displayTitle,
                    body: try note.jsonDataString(),
                    sourceURL: nil,
                    embedding: nil
                )
            )
        }
        try await appendReceipt(
            entityID: noteID,
            receiptType: NoteReceiptType.linkCreated.rawValue,
            payload: [
                "targetEntityID": .string(otherEntityID.uuidString),
                "linkType": .string(linkType),
                "weight": .number(Double(weight)),
            ]
        )
        return link
    }

    // MARK: - Receipts

    /// All constitutional receipts for a note, oldest first. The
    /// receipt chain is the audit trail the user sees in the
    /// receipt drawer.
    public func receipts(forNote noteID: UUID) async throws -> [GraphReceipt] {
        try await dataLayer.receipts(forEntity: noteID)
    }

    // MARK: - Helpers

    private func appendReceipt(
        entityID: UUID,
        receiptType: String,
        payload: [String: JSONValue]
    ) async throws {
        _ = try await dataLayer.appendReceipt(
            entityID: entityID,
            receiptType: receiptType,
            payload: payload
        )
    }

    /// Load a note or throw ``NoteStoreError/noteNotFound(id:)``.
    /// Used by the mutation methods so the public API surfaces a
    /// typed error rather than an optional.
    private func loadOrFail(id: UUID) async throws -> Note {
        guard let note = try await get(id: id) else {
            throw NoteStoreError.noteNotFound(id: id)
        }
        return note
    }

    private static func noteFromEntity(_ entity: GraphEntity) throws -> Note? {
        guard entity.entityType == Note.entityType else { return nil }
        guard let body = entity.body, !body.isEmpty else { return nil }
        return try Note.from(jsonDataString: body)
    }
}

// MARK: - JSON helpers on Note

extension Note {
    /// Encode to a UTF-8 string. Convenience for the note store
    /// (which deals in `String` bodies for the data layer).
    func jsonDataString() throws -> String {
        let data = try jsonData()
        guard let s = String(data: data, encoding: .utf8) else {
            throw NoteStoreError.invalidNoteBody(reason: "encoded JSON is not UTF-8")
        }
        return s
    }

    /// Decode from a UTF-8 string. Inverse of ``jsonDataString()``.
    static func from(jsonDataString: String) throws -> Note {
        guard let data = jsonDataString.data(using: .utf8) else {
            throw NoteStoreError.invalidNoteBody(reason: "body is not valid UTF-8")
        }
        return try from(jsonData: data)
    }
}

// MARK: - Receipt type taxonomy

/// The set of receipt types ``NoteStore`` emits. The string
/// values are what show up in the `graph_receipts.receipt_type`
/// column; the enum keeps the call sites self-documenting and
/// makes the receipt vocabulary searchable.
public enum NoteReceiptType: String, Codable, Sendable, CaseIterable {
    /// A note was created or updated. The payload includes the
    /// title, tag count, and pin/archive state.
    case upsert = "note_upsert"
    /// A note was deleted.
    case delete = "note_delete"
    /// The note's title was changed.
    case titleChanged = "note_title_changed"
    /// The note's body AST was replaced. The payload includes
    /// the block count.
    case bodyChanged = "note_body_changed"
    /// A note was pinned.
    case pinned = "note_pinned"
    /// A note was unpinned.
    case unpinned = "note_unpinned"
    /// A note was archived.
    case archived = "note_archived"
    /// A note was unarchived.
    case unarchived = "note_unarchived"
    /// The note's full tag list was set.
    case tagsChanged = "note_tags_changed"
    /// A single tag was added to a note.
    case tagAdded = "note_tag_added"
    /// A single tag was removed from a note.
    case tagRemoved = "note_tag_removed"
    /// A note was linked to another graph entity.
    case linkCreated = "note_link_created"
}

// MARK: - Errors

public enum NoteStoreError: Error, Sendable, Equatable {
    case noteNotFound(id: UUID)
    case invalidNoteBody(reason: String)
}
