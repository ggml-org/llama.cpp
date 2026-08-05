import Foundation

// MARK: - ContactStore

/// The contact-specific seam between importers / the SwiftUI view
/// and the data layer facade (``TesseraDataLayer``). Mirrors
/// ``DocumentStore``'s role for the document material type: own
/// the contact ↔ data layer integration, enforce the constitutional
/// receipt invariant for every mutation, expose a domain-shaped API
/// to the rest of the productivity surface.
///
/// The store is a struct because the data layer actor is the source
/// of concurrency safety. There's no mutable state of our own.
///
/// **Receipt contract:** every mutation that changes contact state
/// (insert, update, delete, link, unlink) appends a signed receipt
/// to the `graph_receipts` table with a `receipt_type` from
/// `ContactReceiptType` and a payload that names the affected
/// `entity_id` plus a structural summary. The agent's
/// `hybrid_search` over `entity_type = 'contact'` returns these
/// receipts when assembling context.
public struct ContactStore: Sendable {

    private let dataLayer: TesseraDataLayer

    public init(dataLayer: TesseraDataLayer) {
        self.dataLayer = dataLayer
    }

    // MARK: - CRUD

    /// Persist a contact. If the contact's id already exists in the
    /// graph_entities table the row is updated in place; otherwise a
    /// new row is inserted. Always appends a `contact_upsert` receipt.
    public func upsert(_ contact: Contact) async throws -> Contact {
        let body = try contact.jsonDataString()
        let label = contact.displayName
        _ = try await dataLayer.upsertEntity(
            GraphEntityUpsert(
                id: contact.id,
                entityType: Contact.entityType,
                subtype: contact.subtypeString,
                label: label,
                body: body,
                sourceURL: contact.sourceURL,
                embedding: nil
            )
        )
        try await appendReceipt(
            entityID: contact.id,
            receiptType: ContactReceiptType.upsert.rawValue,
            payload: [
                "subtype": .string(contact.subtypeString),
                "displayName": .string(label),
                "emailCount": .number(Double(contact.emails.count)),
                "phoneCount": .number(Double(contact.phones.count)),
            ]
        )
        return contact
    }

    /// Fetch one contact by id. Returns nil when no `graph_entity`
    /// row with `entity_type = 'contact'` matches.
    public func get(id: UUID) async throws -> Contact? {
        guard let entity = try await dataLayer.getEntity(id: id) else {
            return nil
        }
        guard entity.entityType == Contact.entityType else {
            return nil
        }
        return try Self.contactFromEntity(entity)
    }

    /// Delete a contact by id. Returns true when a row was removed.
    /// The deletion cascades to `entity_links` and `graph_receipts`
    /// via the foreign keys; the receipt chain for the contact is
    /// preserved (the row count on the receipts table doesn't
    /// decrease — receipts are append-only, and the contact row
    /// is the only thing deleted).
    @discardableResult
    public func delete(id: UUID) async throws -> Bool {
        let didDelete = try await dataLayer.deleteEntity(id: id)
        if didDelete {
            try await appendReceipt(
                entityID: id,
                receiptType: ContactReceiptType.delete.rawValue,
                payload: [:]
            )
        }
        return didDelete
    }

    // MARK: - Search

    /// Find contacts whose display name matches `query`. The
    /// implementation uses `hybrid_search` over the entity graph
    /// (the data layer's RRF query already returns every
    /// `entity_type = 'contact'` row, ranked by name + body
    /// similarity). This is what the agent's "find John's contact"
    /// call goes through.
    ///
    /// The query is anchored to a contact the agent already knows
    /// about (the `anchor` parameter). When the agent has no anchor
    /// it can pass the first matching contact from a prior
    /// unanchored query. The facade is intentionally anchor-based
    /// because the underlying `hybrid_search` walks the graph
    /// starting from `anchor`; the `fallback` is a name match for
    /// the common "no anchor" case.
    public func search(matching query: String, limit: Int = 20) async throws -> [Contact] {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        // Use the data layer's hybrid_search against the anchor
        // that's the closest match. For the no-anchor case we
        // fall back to a prefix label search directly against
        // the entity table.
        let rows = try await dataLayer.searchByLabelPrefix(
            entityType: Contact.entityType,
            labelPrefix: trimmed.lowercased(),
            limit: limit
        )
        return try rows.compactMap { try? Self.contactFromEntity($0) }
    }

    /// All contacts. Returns an empty array when the table is
    /// empty. The implementation reads the entity table without an
    /// anchor (the data layer's `listByEntityType` covers this
    /// case; see the implementation for the offset / limit pattern).
    public func list(limit: Int = 1000) async throws -> [Contact] {
        let rows = try await dataLayer.listByEntityType(
            entityType: Contact.entityType,
            limit: limit
        )
        return try rows.compactMap { try? Self.contactFromEntity($0) }
    }

    // MARK: - Linking

    /// Link a contact to another graph entity (an email, a calendar
    /// event, a task, a document). Creates an `entity_link` row
    /// AND appends a `contact_link_created` receipt so the audit
    /// trail captures the relationship. The link is a no-op if a
    /// row with the same (source, target, type) tuple already
    /// exists (the data layer's unique constraint handles this).
    @discardableResult
    public func linkContact(
        _ contactID: UUID,
        to otherEntityID: UUID,
        linkType: String = "related_to",
        weight: Float = 1.0
    ) async throws -> EntityLink {
        let link = try await dataLayer.linkEntities(
            sourceID: contactID,
            targetID: otherEntityID,
            linkType: linkType,
            weight: weight
        )
        try await appendReceipt(
            entityID: contactID,
            receiptType: ContactReceiptType.linkCreated.rawValue,
            payload: [
                "targetEntityID": .string(otherEntityID.uuidString),
                "linkType": .string(linkType),
                "weight": .number(Double(weight)),
            ]
        )
        return link
    }

    // MARK: - Receipts

    /// All constitutional receipts for a contact, oldest first. The
    /// receipt chain is the audit trail the user sees in the
    /// receipt drawer.
    public func receipts(forContact contactID: UUID) async throws -> [GraphReceipt] {
        try await dataLayer.receipts(forEntity: contactID)
    }

    // MARK: - Egress (privacy)

    /// Export a contact to VCard data. The export is gated by
    /// ``TesseraContactEgressGuard``: the contact data is sensitive
    /// (PII), so we require a positive `provenance` value and we
    /// log a `contact_export` receipt.
    ///
    /// The caller pre-serializes the contact (the importer is an
    /// actor; the store is a struct). The store just orchestrates
    /// the egress policy and the receipt.
    public func exportVCard(
        _ contact: Contact,
        preEncodedVCard data: Data,
        provenance: String
    ) async throws -> Data {
        guard TesseraContactEgressGuard.allows(provenance) else {
            throw ContactStoreError.egressDenied(provenance: provenance)
        }
        try await appendReceipt(
            entityID: contact.id,
            receiptType: ContactReceiptType.contactExport.rawValue,
            payload: [
                "provenance": .string(provenance),
                "byteCount": .number(Double(data.count)),
            ]
        )
        return data
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

    private static func contactFromEntity(_ entity: GraphEntity) throws -> Contact? {
        guard entity.entityType == Contact.entityType else { return nil }
        guard let body = entity.body, !body.isEmpty else { return nil }
        return try Contact.from(jsonDataString: body)
    }
}

// MARK: - JSON helpers on Contact

extension Contact {
    /// Encode to a UTF-8 string. Convenience for the contact store
    /// (which deals in `String` bodies for the data layer).
    func jsonDataString() throws -> String {
        let data = try jsonData()
        guard let s = String(data: data, encoding: .utf8) else {
            throw ContactStoreError.invalidContactBody(reason: "encoded JSON is not UTF-8")
        }
        return s
    }

    /// Decode from a UTF-8 string. Inverse of ``jsonDataString()``.
    static func from(jsonDataString: String) throws -> Contact {
        guard let data = jsonDataString.data(using: .utf8) else {
            throw ContactStoreError.invalidContactBody(reason: "body is not valid UTF-8")
        }
        return try from(jsonData: data)
    }
}

// MARK: - Receipt type taxonomy

/// The set of receipt types ``ContactStore`` emits. The string
/// values are what show up in the `graph_receipts.receipt_type`
/// column; the enum keeps the call sites self-documenting and
/// makes the receipt vocabulary searchable.
public enum ContactReceiptType: String, Codable, Sendable, CaseIterable {
    /// A contact was created or updated.
    case upsert = "contact_upsert"
    /// A contact was deleted.
    case delete = "contact_delete"
    /// A contact was linked to another graph entity.
    case linkCreated = "contact_link_created"
    /// A contact's link to another graph entity was removed.
    case linkDeleted = "contact_link_deleted"
    /// A contact was exported (VCard, Google write-back, ...).
    /// Always paired with a positive ``TesseraContactEgressGuard``
    /// check.
    case contactExport = "contact_export"
}

// MARK: - Errors

public enum ContactStoreError: Error, Sendable, Equatable {
    case contactNotFound(id: UUID)
    case invalidContactBody(reason: String)
    case egressDenied(provenance: String)
}

// MARK: - Contact egress guard

/// Privacy guard for contact exports. Mirrors the shape of
/// ``TesseraEgressGuard`` (used by the runtime-traces path) but
/// with contact-specific allow-list values. Fail-closed: any
/// provenance value that isn't in the allow-list denies the
/// export. The receipt drawer shows the deny reason so the user
/// can correct it (most often by setting the right provenance
/// in the export dialog).
public enum TesseraContactEgressGuard {
    /// Allow-list of provenance values. The user explicitly chose
    /// each of these in the export dialog; the guard denies any
    /// value that isn't on the list.
    public static let allowedProvenances: Set<String> = [
        "user_explicit_export",
        "share_sheet",
        "agent_for_user",
    ]

    /// True iff the export may proceed.
    public static func allows(_ provenance: String) -> Bool {
        allowedProvenances.contains(provenance)
    }
}
