import Foundation

// MARK: - EmailStore

/// The email-specific seam between importers / the
/// SwiftUI view / the chat panel and the data layer
/// facade (``TesseraDataLayer``). Mirrors
/// ``ContactStore``'s role for the contact material
/// type: own the email <-> data layer integration,
/// enforce the constitutional receipt invariant for
/// every mutation, expose a domain-shaped API to the
/// rest of the productivity surface.
///
/// The store is a struct because the data layer actor
/// is the source of concurrency safety. There's no
/// mutable state of our own.
///
/// **Receipt contract.** Every mutation that changes
/// email state (import, read, flag changes, folder
/// moves, delete, draft save, reply/forward send)
/// appends a signed receipt to the `graph_receipts`
/// table with a `receipt_type` from
/// ``EmailReceiptType`` and a payload that names the
/// affected `entity_id` plus a structural summary.
/// The agent's `hybrid_search` over
/// `entity_type = 'email'` returns these receipts
/// when assembling context (the receipt drawer
/// surfaces the full chain).
public struct EmailStore: Sendable {

    private let dataLayer: TesseraDataLayer

    public init(dataLayer: TesseraDataLayer) {
        self.dataLayer = dataLayer
    }

    /// Internal accessor used by the
    /// ``EmailSender`` extension. Marked internal
    /// (not private) so the cross-file extension
    /// in `EmailSender.swift` can append receipts
    /// on behalf of the sender without re-fetching
    /// the email. The store's public surface
    /// (CRUD, listing, flag mutations) is what
    /// callers use; this is the seam between
    /// ``EmailStore`` and ``EmailSender``.
    internal func _dataLayer() -> TesseraDataLayer {
        dataLayer
    }

    // MARK: - CRUD

    /// Persist an email. If the email's id already
    /// exists in `graph_entities` the row is updated
    /// in place; otherwise a new row is inserted.
    /// Always appends an `email_upsert` receipt.
    @discardableResult
    public func upsert(_ email: EmailMessage) async throws -> EmailMessage {
        let body = try email.jsonDataString()
        let label = email.displaySubject
        _ = try await dataLayer.upsertEntity(
            GraphEntityUpsert(
                id: email.id,
                entityType: EmailMessage.entityType,
                subtype: nil,
                label: label,
                body: body,
                sourceURL: nil,
                embedding: nil
            )
        )
        try await appendReceipt(
            entityID: email.id,
            receiptType: EmailReceiptType.upsert.rawValue,
            payload: [
                "folder": .string(email.folder.systemID),
                "subject": .string(email.displaySubject),
                "messageID": .string(email.messageID),
                "from": .string(email.from.email),
                "attachmentCount": .number(Double(email.attachments.count)),
            ]
        )
        return email
    }

    /// Fetch one email by id. Returns nil when no
    /// `graph_entity` row with `entity_type = 'email'`
    /// matches.
    public func get(id: UUID) async throws -> EmailMessage? {
        guard let entity = try await dataLayer.getEntity(id: id) else {
            return nil
        }
        guard entity.entityType == EmailMessage.entityType else {
            return nil
        }
        return try Self.emailFromEntity(entity)
    }

    /// Delete an email by id. Returns true when a row
    /// was removed AND appends an `email_delete`
    /// receipt. The deletion cascades to
    /// `entity_links` via the foreign keys; the
    /// receipt chain for the email is preserved
    /// (the row count on the receipts table doesn't
    /// decrease — receipts are append-only).
    @discardableResult
    public func delete(id: UUID) async throws -> Bool {
        let didDelete = try await dataLayer.deleteEntity(id: id)
        if didDelete {
            try await appendReceipt(
                entityID: id,
                receiptType: EmailReceiptType.delete.rawValue,
                payload: [:]
            )
        }
        return didDelete
    }

    // MARK: - Listing

    /// Every email in the user's store, newest first
    /// (by `receivedAt` DESC). The list is what the
    /// macOS Email view's middle column renders.
    /// Uses the data layer's `listByEntityType` —
    /// for thousands of emails the partial index
    /// `idx_entities_email_received` from migration
    /// 0008 makes the list a cheap index scan.
    public func list(limit: Int = 1000, offset: Int = 0) async throws -> [EmailMessage] {
        let rows = try await dataLayer.listByEntityType(
            entityType: EmailMessage.entityType,
            limit: limit,
            offset: offset
        )
        let emails = try rows.compactMap { try? Self.emailFromEntity($0) }
        return emails.sorted { $0.receivedAt > $1.receivedAt }
    }

    /// Emails in a given folder. The list query
    /// happens in-memory (the folder is part of the
    /// JSON body); for the v1 email volumes (<10k
    /// messages) this is well under a millisecond.
    /// The v2 dedicated `email_folders` table is the
    /// path to scale beyond.
    public func list(
        in folder: Folder,
        limit: Int = 1000
    ) async throws -> [EmailMessage] {
        let all = try await list(limit: limit * 4, offset: 0)
        return all
            .filter { $0.folder == folder }
            .prefix(limit)
            .map { $0 }
    }

    /// Email with the given `messageID` (RFC 5322
    /// bare id). Used by the importer's idempotency
    /// path and by the chat panel's "find this email"
    /// hook. Returns nil when no match.
    public func find(messageID: String) async throws -> EmailMessage? {
        let all = try await list(limit: 1000)
        return all.first { $0.messageID == messageID }
    }

    /// Group emails by thread. The result is a list
    /// of (threadID, emails) pairs; the order is
    /// newest-first (by the latest message in each
    /// thread). Emails with no thread information
    /// are bucketed under their own messageID.
    public func threads(limit: Int = 500) async throws -> [(threadID: String, emails: [EmailMessage])] {
        let all = try await list(limit: limit)
        var groups: [String: [EmailMessage]] = [:]
        for email in all {
            let tid = email.threadID ?? email.messageID
            groups[tid, default: []].append(email)
        }
        return groups
            .map { (threadID: $0.key, emails: $0.value.sorted { $0.receivedAt > $1.receivedAt }) }
            .sorted { $0.emails.first?.receivedAt ?? .distantPast > $1.emails.first?.receivedAt ?? .distantPast }
    }

    // MARK: - Flag mutations

    /// Mark an email as read. Returns the updated
    /// message. The receipt payload carries the
    /// prior read state so the audit trail records
    /// "unread -> read" transitions.
    @discardableResult
    public func markRead(_ id: UUID, read: Bool = true) async throws -> EmailMessage? {
        guard var email = try await get(id: id) else { return nil }
        let prior = email.isRead
        guard prior != read else { return email }
        email.isRead = read
        email.updatedAt = Date()
        try await upsert(email)
        try await appendReceipt(
            entityID: id,
            receiptType: EmailReceiptType.read.rawValue,
            payload: [
                "prior": .bool(prior),
                "next": .bool(read),
            ]
        )
        return email
    }

    /// Star / unstar. The receipt payload records
    /// the prior star state.
    @discardableResult
    public func setStarred(_ id: UUID, starred: Bool) async throws -> EmailMessage? {
        guard var email = try await get(id: id) else { return nil }
        let prior = email.isStarred
        guard prior != starred else { return email }
        email.isStarred = starred
        email.updatedAt = Date()
        try await upsert(email)
        try await appendReceipt(
            entityID: id,
            receiptType: EmailReceiptType.starred.rawValue,
            payload: [
                "prior": .bool(prior),
                "next": .bool(starred),
            ]
        )
        return email
    }

    /// Move an email to a folder (archive, trash,
    /// sent, drafts, ...). The receipt payload
    /// carries the prior folder's system id and the
    /// next one so the audit trail records the
    /// transition.
    @discardableResult
    public func setFolder(_ id: UUID, folder: Folder) async throws -> EmailMessage? {
        guard var email = try await get(id: id) else { return nil }
        let prior = email.folder
        guard prior != folder else { return email }
        email.folder = folder
        // The `isArchived` / `isTrashed` flags stay in
        // sync with the folder for the sidebar's
        // filtering. They were the pre-folder-era way
        // to mark these states; the folder is now the
        // canonical source.
        email.isArchived = (folder == .archive)
        email.isTrashed = (folder == .trash)
        email.updatedAt = Date()
        try await upsert(email)
        let receiptType: String
        switch folder {
        case .archive: receiptType = EmailReceiptType.archived.rawValue
        case .trash: receiptType = EmailReceiptType.trashed.rawValue
        default: receiptType = EmailReceiptType.folderChanged.rawValue
        }
        try await appendReceipt(
            entityID: id,
            receiptType: receiptType,
            payload: [
                "priorFolder": .string(prior.systemID),
                "nextFolder": .string(folder.systemID),
            ]
        )
        return email
    }

    // MARK: - Sending

    /// Record that a reply was sent for an email.
    /// The original message's `isReplied` flips to
    /// `true`; the draft is upserted as a new
    /// `EmailMessage` in the `.sent` folder; the
    /// reply receipt links the two via the
    /// `inReplyTo` message id.
    @discardableResult
    public func recordReply(
        to originalID: UUID,
        draft: EmailMessage
    ) async throws -> EmailMessage? {
        guard let original = try await get(id: originalID) else { return nil }
        var updated = original
        updated.isReplied = true
        updated.updatedAt = Date()
        try await upsert(updated)

        var sent = draft
        sent.folder = .sent
        sent.isReplied = false
        sent.updatedAt = Date()
        try await upsert(sent)

        try await appendReceipt(
            entityID: originalID,
            receiptType: EmailReceiptType.replied.rawValue,
            payload: [
                "draftID": .string(sent.id.uuidString),
                "threadID": .string(sent.threadID ?? ""),
            ]
        )
        return sent
    }

    /// Record that an email was forwarded. The
    /// forward itself is stored as a new
    /// `EmailMessage` in `.sent`; the original
    /// message's `isForwarded` flips to `true`.
    @discardableResult
    public func recordForward(
        of originalID: UUID,
        draft: EmailMessage
    ) async throws -> EmailMessage? {
        guard let original = try await get(id: originalID) else { return nil }
        var updated = original
        updated.isForwarded = true
        updated.updatedAt = Date()
        try await upsert(updated)

        var sent = draft
        sent.folder = .sent
        sent.updatedAt = Date()
        try await upsert(sent)

        try await appendReceipt(
            entityID: originalID,
            receiptType: EmailReceiptType.forwarded.rawValue,
            payload: [
                "draftID": .string(sent.id.uuidString),
                "threadID": .string(sent.threadID ?? ""),
            ]
        )
        return sent
    }

    /// Persist a draft. Used by the composer when
    /// the user types but doesn't send; also used
    /// for the share-sheet cancel path (the draft
    /// stays in `.drafts` with a `pendingSend` flag).
    @discardableResult
    public func saveDraft(_ draft: EmailMessage) async throws -> EmailMessage {
        var stored = draft
        if case .drafts = stored.folder {
            // already drafts
        } else {
            stored.folder = .drafts
        }
        stored.updatedAt = Date()
        return try await upsert(stored)
    }

    // MARK: - Cross-surface linking

    /// Link an email to another graph entity (a
    /// contact, a task, a document, a note). Creates
    /// an `entity_link` row AND appends a receipt
    /// so the audit trail captures the relationship.
    /// The link is a no-op if a row with the same
    /// (source, target, type) tuple already exists.
    @discardableResult
    public func link(
        emailID: UUID,
        to otherEntityID: UUID,
        linkType: String = "related_to",
        weight: Float = 1.0
    ) async throws -> EntityLink {
        let link = try await dataLayer.linkEntities(
            sourceID: emailID,
            targetID: otherEntityID,
            linkType: linkType,
            weight: weight
        )
        try await appendReceipt(
            entityID: emailID,
            receiptType: EmailReceiptType.linkCreated.rawValue,
            payload: [
                "targetEntityID": .string(otherEntityID.uuidString),
                "linkType": .string(linkType),
                "weight": .number(Double(weight)),
            ]
        )
        return link
    }

    // MARK: - Receipts

    /// All constitutional receipts for an email,
    /// oldest first. The receipt chain is the audit
    /// trail the user sees in the receipt drawer.
    public func receipts(forEmail emailID: UUID) async throws -> [GraphReceipt] {
        try await dataLayer.receipts(forEntity: emailID)
    }

    /// The set of receipt types this email has
    /// accumulated. Used by the email detail view's
    /// "history" pane to summarize the activity.
    public func receiptTypes(forEmail emailID: UUID) async throws -> [String] {
        let r = try await receipts(forEmail: emailID)
        return r.map { $0.receiptType }
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

    private static func emailFromEntity(_ entity: GraphEntity) throws -> EmailMessage? {
        guard entity.entityType == EmailMessage.entityType else { return nil }
        guard let body = entity.body, !body.isEmpty else { return nil }
        return try EmailMessage.from(jsonDataString: body)
    }
}

// MARK: - Receipt type taxonomy

/// The set of receipt types ``EmailStore`` emits.
/// The string values are what show up in
/// `graph_receipts.receipt_type`; the enum keeps the
/// call sites self-documenting and makes the receipt
/// vocabulary searchable.
///
/// The taxonomy follows the spec's per-mutation
/// requirement (every email mutation produces a
/// receipt). The pre-existing
/// `ContactReceiptType.upsert` / `delete` /
/// `linkCreated` shapes are the template; we extend
/// the vocabulary to cover the email-specific
/// mutations (read, archive, trash, reply, forward).
public enum EmailReceiptType: String, Codable, Sendable, CaseIterable {
    /// An email was created or updated.
    case upsert = "email_upsert"
    /// An email was deleted.
    case delete = "email_delete"
    /// An email's read state changed.
    case read = "email_read"
    /// An email's starred state changed.
    case starred = "email_starred"
    /// An email was moved to a non-trivial folder
    /// (the `.archive` / `.trash` cases use their
    /// own receipt types for fast audit filtering).
    case folderChanged = "email_folder_changed"
    /// An email was archived.
    case archived = "email_archived"
    /// An email was moved to the trash.
    case trashed = "email_trashed"
    /// A reply was sent for an email.
    case replied = "email_replied"
    /// An email was forwarded.
    case forwarded = "email_forwarded"
    /// An email was imported (.eml / .mbox / Apple
    /// Mail export). The importer is the source of
    /// this receipt; the store re-emits it on
    /// upsert for the "I just imported this" case.
    case imported = "email_imported"
    /// An email was linked to another graph entity.
    case linkCreated = "email_link_created"
    /// An email's link to another graph entity was
    /// removed. (v1 the store doesn't expose an
    /// unlink; the enum case is pinned for the v2
    /// graph view's "remove link" gesture.)
    case linkDeleted = "email_link_deleted"
    /// A draft was saved.
    case draftSaved = "email_draft_saved"
    /// A draft was routed to the system share
    /// sheet. The payload carries the path to the
    /// staged .eml file.
    case routedToShareSheet = "email_routed_to_share_sheet"
}

// MARK: - Errors

public enum EmailStoreError: Error, Sendable, Equatable {
    case emailNotFound(id: UUID)
    case invalidEmailBody(reason: String)
    case invalidFolderTransition(from: String, to: String)
    case importerSubprocessFailed(reason: String)
    case shareSheetUnavailable(reason: String)
}
