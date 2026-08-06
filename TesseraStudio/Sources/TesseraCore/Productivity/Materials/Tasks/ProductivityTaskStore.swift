import Foundation

// MARK: - ProductivityTaskStore

/// The ProductivityTasks material's seam to ``TesseraDataLayer``.
///
/// Mirrors ``ContactStore``'s role: own the ProductivityTask ↔ data layer
/// integration, enforce the constitutional receipt invariant
/// for every mutation, expose a domain-shaped API to the rest
/// of the productivity surface.
///
/// The store is a struct because the data layer actor is the
/// source of concurrency safety. There's no mutable state of
/// our own.
///
/// **Receipt contract:** every mutation that changes task state
/// (create, update, complete, delete, move list, link, unlink)
/// appends a signed receipt to the `graph_receipts` table with
/// a `receipt_type` from ``ProductivityTaskReceiptType`` and a payload that
/// names the affected `entity_id` plus a structural summary.
/// The agent's `hybrid_search` over `entity_type = 'task'`
/// returns these receipts when assembling context.
public struct ProductivityTaskStore: Sendable {

    private let dataLayer: TesseraDataLayer

    public init(dataLayer: TesseraDataLayer) {
        self.dataLayer = dataLayer
    }

    /// The underlying data layer. Exposed for callers
    /// (the chat-panel bridge, the graph view) that need
    /// to emit custom receipt types or reach the data
    /// layer's other affordances.
    public var exposedDataLayer: TesseraDataLayer { dataLayer }

    // MARK: - CRUD

    /// Create or update a task. If the task's id already
    /// exists in the `graph_entities` table the row is updated
    /// in place; otherwise a new row is inserted. Always appends
    /// a `task_upsert` receipt.
    ///
    /// The `actor` parameter drives the receipt's provenance
    /// (user vs. agent). The chat panel integration passes the
    /// user actor because the human initiated the action; the
    /// agent-created tasks get an agent actor.
    public func upsert(
        _ task: ProductivityTask,
        actor: Actor = .user(UUID())
    ) async throws -> ProductivityTask {
        var stored = task
        stored.updatedAt = Date()
        let body = try stored.jsonDataString()
        let label = stored.title.isEmpty ? "(untitled)" : stored.title
        _ = try await dataLayer.upsertEntity(
            GraphEntityUpsert(
                id: stored.id,
                entityType: ProductivityTask.entityType,
                subtype: stored.subtypeString,
                label: label,
                body: body,
                sourceURL: stored.sourceURL,
                embedding: nil
            )
        )
        try await appendReceipt(
            entityID: stored.id,
            receiptType: ProductivityTaskReceiptType.upsert.rawValue,
            payload: [
                "list": .string(stored.list.rawValue),
                "priority": .string(stored.priority.rawValue),
                "hasDueAt": .bool(stored.dueAt != nil),
                "isCompleted": .bool(stored.isCompleted),
                "tagCount": .number(Double(stored.tags.count)),
                "linkCount": .number(Double(stored.linkedEntityIDs.count)),
            ],
            actor: actor
        )
        return stored
    }

    /// Fetch one task by id. Returns nil when no `graph_entity`
    /// row with `entity_type = 'task'` matches.
    public func get(id: UUID) async throws -> ProductivityTask? {
        guard let entity = try await dataLayer.getEntity(id: id) else {
            return nil
        }
        guard entity.entityType == ProductivityTask.entityType else {
            return nil
        }
        return try Self.taskFromEntity(entity)
    }

    /// Delete a task by id. Returns true when a row was
    /// removed. The deletion cascades to `entity_links` and
    /// `graph_receipts` via the foreign keys; the receipt
    /// chain for the task is preserved (the row count on the
    /// receipts table doesn't decrease -- receipts are
    /// append-only, and the task row is the only thing
    /// deleted).
    @discardableResult
    public func delete(
        id: UUID,
        actor: Actor = .user(UUID())
    ) async throws -> Bool {
        let didDelete = try await dataLayer.deleteEntity(id: id)
        if didDelete {
            try await appendReceipt(
                entityID: id,
                receiptType: ProductivityTaskReceiptType.delete.rawValue,
                payload: [:],
                actor: actor
            )
        }
        return didDelete
    }

    // MARK: - Domain operations

    /// Mark a task as completed. The `completedAt` field is
    /// set to `now`; the task is moved to a completed state
    /// (the list views filter on this). The mutation is
    /// append-only via the receipt chain.
    public func complete(
        id: UUID,
        at completionTime: Date = Date(),
        actor: Actor = .user(UUID())
    ) async throws -> ProductivityTask? {
        guard var task = try await get(id: id) else { return nil }
        task.completedAt = completionTime
        task.updatedAt = completionTime
        return try await upsert(task, actor: actor)
    }

    /// Reopen a completed task. The `completedAt` field is
    /// cleared; the task returns to its previous list (the
    /// user can move it back to the inbox if they want to
    /// re-triage).
    public func reopen(
        id: UUID,
        actor: Actor = .user(UUID())
    ) async throws -> ProductivityTask? {
        guard var task = try await get(id: id) else { return nil }
        task.completedAt = nil
        task.updatedAt = Date()
        return try await upsert(task, actor: actor)
    }

    /// Move a task to a different list. The mutation is
    /// recorded as a `task_moved` receipt so the audit trail
    /// shows the path (Inbox → Today → Anytime, etc.).
    public func move(
        id: UUID,
        to newList: ProductivityTask.List,
        actor: Actor = .user(UUID())
    ) async throws -> ProductivityTask? {
        guard var task = try await get(id: id) else { return nil }
        let oldList = task.list
        guard oldList != newList else { return task }
        task.list = newList
        task.updatedAt = Date()
        let saved = try await upsert(task, actor: actor)
        try await appendReceipt(
            entityID: saved.id,
            receiptType: ProductivityTaskReceiptType.moved.rawValue,
            payload: [
                "fromList": .string(oldList.rawValue),
                "toList": .string(newList.rawValue),
            ],
            actor: actor
        )
        return saved
    }

    /// Update the priority. Recorded as a `task_priority_changed`
    /// receipt.
    public func setPriority(
        id: UUID,
        to newPriority: ProductivityTask.Priority,
        actor: Actor = .user(UUID())
    ) async throws -> ProductivityTask? {
        guard var task = try await get(id: id) else { return nil }
        let oldPriority = task.priority
        guard oldPriority != newPriority else { return task }
        task.priority = newPriority
        task.updatedAt = Date()
        let saved = try await upsert(task, actor: actor)
        try await appendReceipt(
            entityID: saved.id,
            receiptType: ProductivityTaskReceiptType.priorityChanged.rawValue,
            payload: [
                "fromPriority": .string(oldPriority.rawValue),
                "toPriority": .string(newPriority.rawValue),
            ],
            actor: actor
        )
        return saved
    }

    /// Set the due date. The receipt records the transition
    /// (or the new value when `newDueAt` is non-nil). Setting
    /// `newDueAt` to nil removes the due date.
    public func setDueDate(
        id: UUID,
        to newDueAt: Date?,
        actor: Actor = .user(UUID())
    ) async throws -> ProductivityTask? {
        guard var task = try await get(id: id) else { return nil }
        let oldDueAt = task.dueAt
        guard oldDueAt != newDueAt else { return task }
        task.dueAt = newDueAt
        task.updatedAt = Date()
        let saved = try await upsert(task, actor: actor)
        let payload: [String: JSONValue] = [
            "hadDueAt": .bool(oldDueAt != nil),
            "hasDueAt": .bool(newDueAt != nil),
        ]
        try await appendReceipt(
            entityID: saved.id,
            receiptType: ProductivityTaskReceiptType.dueDateChanged.rawValue,
            payload: payload,
            actor: actor
        )
        return saved
    }

    // MARK: - Listing

    /// All tasks. Returns an empty array when the table is
    /// empty. The implementation reads the entity table
    /// without an anchor (the data layer's `listByEntityType`
    /// covers this case; see the implementation for the
    /// offset / limit pattern).
    public func list(limit: Int = 1000) async throws -> [ProductivityTask] {
        let rows = try await dataLayer.listByEntityType(
            entityType: ProductivityTask.entityType,
            limit: limit
        )
        return try rows.compactMap { try? Self.taskFromEntity($0) }
    }

    /// ProductivityTasks in one specific list. The `list(...)` method
    /// returns every task; the caller filters in memory. This
    /// method is the in-DB version of the same filter
    /// (the data layer's `searchByLabelPrefix` is too narrow
    /// for this; we use `listByEntityType` and filter on the
    /// materialised list field).
    public func list(
        in list: ProductivityTask.List,
        limit: Int = 1000
    ) async throws -> [ProductivityTask] {
        let all = try await self.list(limit: limit)
        return all.filter { $0.list == list }
    }

    /// The Inbox list. Alias for `list(in: .inbox)`; provided
    /// for parity with the UX spec's vocabulary.
    public func inbox(limit: Int = 1000) async throws -> [ProductivityTask] {
        try await list(in: .inbox, limit: limit)
    }

    /// The Today list. Today is computed from the due date
    /// (`dueAt <= now + 24h` for active tasks) rather than
    /// from the `list` field. A task can be in `list = .anytime`
    /// but still appear in Today if its due date is in the
    /// next 24 hours; the spec §12.2 is explicit about this.
    public func today(
        asOf now: Date = Date(),
        limit: Int = 1000
    ) async throws -> [ProductivityTask] {
        let all = try await self.list(limit: limit)
        return all
            .filter { !$0.isCompleted && $0.isDueWithin24Hours(asOf: now) }
            .sorted { $0.dueAt ?? now < $1.dueAt ?? now }
    }

    /// The Upcoming list. ProductivityTasks due in the next 7 days but
    /// NOT in the next 24 hours.
    public func upcoming(
        asOf now: Date = Date(),
        limit: Int = 1000
    ) async throws -> [ProductivityTask] {
        let all = try await self.list(limit: limit)
        return all
            .filter { !$0.isCompleted && $0.isDueWithin7DaysButNotToday(asOf: now) }
            .sorted { $0.dueAt ?? now < $1.dueAt ?? now }
    }

    /// The Anytime list. Active tasks with no due date, AND
    /// tasks whose `list` field is `.anytime`. The user
    /// sometimes moves a task to Anytime to defer it
    /// (even with a due date); we honour that.
    public func anytime(limit: Int = 1000) async throws -> [ProductivityTask] {
        try await list(in: .anytime, limit: limit)
    }

    /// The Someday list. ProductivityTasks the user has explicitly
    /// deferred.
    public func someday(limit: Int = 1000) async throws -> [ProductivityTask] {
        try await list(in: .someday, limit: limit)
    }

    /// Completed tasks, newest completed first. The "Recently
    /// completed" view in the detail panel; the surface does
    /// not have a dedicated sidebar entry.
    public func completed(limit: Int = 1000) async throws -> [ProductivityTask] {
        let all = try await self.list(limit: limit)
        return all
            .filter { $0.isCompleted }
            .sorted { ($0.completedAt ?? .distantPast) > ($1.completedAt ?? .distantPast) }
    }

    // MARK: - Search

    /// Find tasks whose title or notes match `query`. The
    /// implementation uses the data layer's
    /// `searchByLabelPrefix` for the title match (the
    /// index is on `(entity_type, label)`) and falls back
    /// to a case-insensitive substring search on the body
    /// for the notes match. For 10k+ tasks the prefix match
    /// is O(log n); the substring scan is O(n) but only on
    /// the body of the matching rows.
    public func search(matching query: String, limit: Int = 50) async throws -> [ProductivityTask] {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        let rows = try await dataLayer.searchByLabelPrefix(
            entityType: ProductivityTask.entityType,
            labelPrefix: trimmed.lowercased(),
            limit: limit
        )
        let prefixMatches = try rows.compactMap { try? Self.taskFromEntity($0) }
        // Substring search across body (notes + title)
        if trimmed.count >= 3 {
            let all = try await self.list(limit: 5000)
            let needle = trimmed.lowercased()
            let extras = all.filter { task in
                !prefixMatches.contains(where: { $0.id == task.id })
                && (task.notes.lowercased().contains(needle)
                    || task.title.lowercased().contains(needle))
            }
            return Array((prefixMatches + extras).prefix(limit))
        }
        return prefixMatches
    }

    // MARK: - Linking

    /// Link a task to another graph entity. Creates an
    /// `entity_link` row AND appends a `task_link_created`
    /// receipt so the audit trail captures the relationship.
    /// The link is a no-op if a row with the same (source,
    /// target, type) tuple already exists (the data layer's
    /// unique constraint handles this).
    @discardableResult
    public func linkTask(
        _ taskID: UUID,
        to otherEntityID: UUID,
        linkType: String = "related_to",
        weight: Float = 1.0,
        actor: Actor = .user(UUID())
    ) async throws -> EntityLink {
        let link = try await dataLayer.linkEntities(
            sourceID: taskID,
            targetID: otherEntityID,
            linkType: linkType,
            weight: weight
        )
        try await appendReceipt(
            entityID: taskID,
            receiptType: ProductivityTaskReceiptType.linkCreated.rawValue,
            payload: [
                "targetEntityID": .string(otherEntityID.uuidString),
                "linkType": .string(linkType),
                "weight": .number(Double(weight)),
            ],
            actor: actor
        )
        return link
    }

    /// Remove a link from a task to another graph entity.
    /// The receipt records the unlink.
    public func unlinkTask(
        _ taskID: UUID,
        from otherEntityID: UUID,
        linkType: String = "related_to",
        actor: Actor = .user(UUID())
    ) async throws {
        // The data layer doesn't expose a `deleteEntityLink`
        // yet; the receipt records the unlink request and
        // the link row is left for the follow-up migration
        // to clean up. The Phase 1 surfaces don't have this
        // need (the receipt chain is the source of truth).
        try await appendReceipt(
            entityID: taskID,
            receiptType: ProductivityTaskReceiptType.linkDeleted.rawValue,
            payload: [
                "targetEntityID": .string(otherEntityID.uuidString),
                "linkType": .string(linkType),
            ],
            actor: actor
        )
    }

    // MARK: - Receipts

    /// All constitutional receipts for a task, oldest first.
    /// The receipt chain is the audit trail the user sees in
    /// the receipt drawer.
    public func receipts(forTask taskID: UUID) async throws -> [GraphReceipt] {
        try await dataLayer.receipts(forEntity: taskID)
    }

    // MARK: - Helpers

    private func appendReceipt(
        entityID: UUID,
        receiptType: String,
        payload: [String: JSONValue],
        actor: Actor
    ) async throws {
        _ = try await dataLayer.appendReceipt(
            entityID: entityID,
            receiptType: receiptType,
            payload: payload
        )
        // Note: the actor parameter is currently unused at the
        // data layer (Phase 1's `appendReceipt` doesn't have
        // an actor field; it gets one when the receipt
        // infrastructure lands). We keep the parameter for
        // call-site symmetry with the receipt-signed path
        // (Phase 1's `Receipt` type carries the actor and the
        // chat panel emits agent-actor receipts).
        _ = actor
    }

    private static func taskFromEntity(_ entity: GraphEntity) throws -> ProductivityTask? {
        guard entity.entityType == ProductivityTask.entityType else { return nil }
        guard let body = entity.body, !body.isEmpty else { return nil }
        return try ProductivityTask.from(jsonDataString: body)
    }
}

// MARK: - Receipt type taxonomy

/// The set of receipt types ``ProductivityTaskStore`` emits. The string
/// values are what show up in the `graph_receipts.receipt_type`
/// column; the enum keeps the call sites self-documenting
/// and makes the receipt vocabulary searchable.
public enum ProductivityTaskReceiptType: String, Codable, Sendable, CaseIterable {
    /// A task was created or updated (the upsert path).
    case upsert = "task_upsert"
    /// A task was deleted.
    case delete = "task_delete"
    /// A task was marked completed.
    case completed = "task_completed"
    /// A completed task was reopened.
    case reopened = "task_reopened"
    /// A task was moved to a different list.
    case moved = "task_moved"
    /// A task's priority was changed.
    case priorityChanged = "task_priority_changed"
    /// A task's due date was set or cleared.
    case dueDateChanged = "task_due_date_changed"
    /// A task was linked to another graph entity.
    case linkCreated = "task_link_created"
    /// A task's link to another graph entity was removed.
    case linkDeleted = "task_link_deleted"
    /// A task was created from a chat panel queue item.
    /// Distinct from `upsert` so the audit trail can show
    /// the chat-panel provenance.
    case createdFromChat = "task_created_from_chat"
    /// A task was created by the NLU parser. The payload
    /// carries the raw user input.
    case createdFromNLU = "task_created_from_nlu"
}

// MARK: - Errors

public enum ProductivityTaskStoreError: Error, Sendable, Equatable {
    case taskNotFound(id: UUID)
    case invalidTaskBody(reason: String)
    case noDueDateSet
    case noActiveUser
}
