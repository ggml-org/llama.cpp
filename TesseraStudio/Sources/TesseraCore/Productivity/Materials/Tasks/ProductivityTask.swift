import Foundation

// MARK: - ProductivityTask

/// A first-class material in the Tessera productivity surface.
///
/// ProductivityTasks are stored as `graph_entity` rows with
/// `entity_type = 'task'`, matching the universal "one row per
/// thing" pattern documented in `docs/tessera-data-layer-design.md`
/// §3. The `body` column carries the task fields as JSON
/// (title, notes, dueAt, completedAt, priority, tags,
/// linkedEntityIDs, sourceURL); `label` is the title (for the
/// graph view / search prefix); `subtype` is the current
/// ``List`` raw value (so the list can be filtered in SQL
/// without a JSONB extract).
///
/// The struct is intentionally self-contained: no references
/// to the data layer, no framework imports beyond `Foundation`.
/// The ``ProductivityTaskStore`` is the seam to ``TesseraDataLayer``; the
/// NLU parser (``ProductivityTaskNLUParser``) is the seam from natural
/// language input.
public struct ProductivityTask: Codable, Sendable, Identifiable, Hashable {

    /// Stable identifier. New tasks get a fresh UUID at
    /// creation; importer round-trips preserve the existing id
    /// so re-imports upsert in place rather than duplicating.
    public let id: UUID

    /// One-line description. The user sees this in every list;
    /// the graph view uses it as the node label.
    public var title: String

    /// Free-form multi-line notes. The detail view shows them;
    /// the NLU parser can also extract structured data from
    /// them.
    public var notes: String

    /// When the task is due. `nil` for tasks that don't have a
    /// commitment (the Anytime list). Drives the Today /
    /// Upcoming / Overdue filters.
    public var dueAt: Date?

    /// When the task was completed. `nil` for active tasks. The
    /// receipt chain records the transition.
    public var completedAt: Date?

    /// The task's priority. Drives the badge in the list view;
    /// the NLU parser can extract a priority from the input
    /// text ("high priority: review the Q3 report").
    public var priority: Priority

    /// Free-form tags. Used for filtering and search. Not the
    /// same as the LinkedMaterial `list` -- `tags` is for
    /// user-defined categorization, `list` is the Things 3-style
    /// bucket.
    public var tags: [String]

    /// The current Things 3-style list. Drives the row's home
    /// in the sidebar. The user can move a task between lists
    /// (the receipt chain records the move).
    public var list: List

    /// Other graph entities this task is linked to. The set
    /// can include contacts (call John), documents (review the
    /// Q3 report), calendar events (prep for the 3pm standup),
    /// or other tasks (parent / subtask). The detail view shows
    /// them as quick links.
    public var linkedEntityIDs: [UUID]

    /// Where the task came from. For tasks the user typed
    /// directly in the UI this is nil; for tasks created from
    /// an email subject or a chat queue item this is the
    /// source URL or the chat item id.
    public var sourceURL: String?

    public var createdAt: Date
    public var updatedAt: Date

    public enum Priority: String, Codable, Sendable, Hashable, CaseIterable, Comparable {
        case none
        case low
        case medium
        case high

        /// Ordinal comparison: `none < low < medium < high`.
        /// Used by the list view's sort.
        public static func < (lhs: Priority, rhs: Priority) -> Bool {
            lhs.rank < rhs.rank
        }

        private var rank: Int {
            switch self {
            case .none: return 0
            case .low: return 1
            case .medium: return 2
            case .high: return 3
            }
        }

        /// Display label. Lowercase to match the rest of the
        /// productivity surface's badge style.
        public var displayName: String {
            switch self {
            case .none: return "no priority"
            case .low: return "low"
            case .medium: return "medium"
            case .high: return "high"
            }
        }
    }

    /// The Things 3-style bucket. The list a task currently
    /// lives in. "Anytime" is the default for new tasks with
    /// no due date; "Inbox" is the default for new tasks before
    /// the user has triaged them.
    public enum List: String, Codable, Sendable, Hashable, CaseIterable {
        case inbox
        case today
        case upcoming
        case anytime
        case someday

        /// Display label. The sidebar shows these in
        /// order.
        public var displayName: String {
            switch self {
            case .inbox: return "Inbox"
            case .today: return "Today"
            case .upcoming: return "Upcoming"
            case .anytime: return "Anytime"
            case .someday: return "Someday"
            }
        }

        /// SF Symbol used in the sidebar. Matches Apple's
        /// "things 3-style" vocabulary; we don't ship the
        /// actual third-party app's icon set.
        public var systemImageName: String {
            switch self {
            case .inbox: return "tray"
            case .today: return "sun.max"
            case .upcoming: return "calendar"
            case .anytime: return "tray.full"
            case .someday: return "moon.zzz"
            }
        }
    }

    public init(
        id: UUID = UUID(),
        title: String,
        notes: String = "",
        dueAt: Date? = nil,
        completedAt: Date? = nil,
        priority: Priority = .none,
        tags: [String] = [],
        list: List = .inbox,
        linkedEntityIDs: [UUID] = [],
        sourceURL: String? = nil,
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.title = title
        self.notes = notes
        self.dueAt = dueAt
        self.completedAt = completedAt
        self.priority = priority
        self.tags = tags
        self.list = list
        self.linkedEntityIDs = linkedEntityIDs
        self.sourceURL = sourceURL
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    /// True iff the task has been completed. The list views
    /// filter on this; the Inbox sort pushes completed
    /// tasks to the bottom.
    public var isCompleted: Bool { completedAt != nil }

    /// True iff the task is overdue (due date is in the past
    /// and the task is not completed). The Today list shows
    /// these with a small red badge.
    public func isOverdue(asOf now: Date = Date()) -> Bool {
        guard let dueAt, completedAt == nil else { return false }
        return dueAt < now
    }

    /// True iff the task is due in the next 24 hours
    /// (inclusive of overdue). Drives the Today list.
    public func isDueWithin24Hours(asOf now: Date = Date()) -> Bool {
        guard let dueAt, completedAt == nil else { return false }
        let cutoff = now.addingTimeInterval(24 * 60 * 60)
        return dueAt <= cutoff
    }

    /// True iff the task is due in the next 7 days but NOT
    /// in the next 24 hours. Drives the Upcoming list. The
    /// "but not today" check is the explicit rule from the
    /// spec's §12.2.
    public func isDueWithin7DaysButNotToday(asOf now: Date = Date()) -> Bool {
        guard let dueAt, completedAt == nil else { return false }
        let tomorrow = Calendar.current.date(
            byAdding: .day, value: 1, to: Calendar.current.startOfDay(for: now)
        ) ?? now.addingTimeInterval(24 * 60 * 60)
        let weekFromNow = now.addingTimeInterval(7 * 24 * 60 * 60)
        return dueAt >= tomorrow && dueAt <= weekFromNow
    }

    /// The list this task SHOULD be in based on its due
    /// date. The user can override (e.g. an upcoming task
    /// parked in Someday), but this is the "auto-classify"
    /// answer the Inbox triage uses.
    public func autoClassifiedList(asOf now: Date = Date()) -> List {
        if completedAt != nil { return list }
        guard let dueAt else { return .anytime }
        if dueAt <= now.addingTimeInterval(24 * 60 * 60) { return .today }
        if dueAt <= now.addingTimeInterval(7 * 24 * 60 * 60) { return .upcoming }
        return .anytime
    }

    /// SF Symbol name for the priority. Drives the badge in
    /// the list view.
    public var prioritySystemImageName: String {
        switch priority {
        case .none: return "circle"
        case .low: return "arrow.down.circle"
        case .medium: return "equal.circle"
        case .high: return "exclamationmark.circle"
        }
    }

    /// Short single-line summary used in the list view. The
    /// notes are clipped at 60 chars so the row stays
    /// single-line.
    public var notesPreview: String {
        let trimmed = notes.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return "" }
        if trimmed.count <= 60 { return trimmed }
        return String(trimmed.prefix(60)) + "…"
    }

    // MARK: - Constants

    /// The `entity_type` value persisted in
    /// `graph_entities.entity_type`. Pinned so the
    /// migration's `WHERE entity_type = 'task'` predicate,
    /// the graph view's "by type" grouping, and the list
    /// filters all agree.
    public static let entityType = "task"

    /// The `subtype` value persisted in
    /// `graph_entities.subtype`. Mirrors ``List``.
    public var subtypeString: String { list.rawValue }
}

// MARK: - JSON helpers

extension ProductivityTask {
    /// Encode the task to JSON. The encoded form is what the
    /// `body` column of `graph_entities` stores. Uses sorted
    /// keys + ISO-8601 dates so the bytes are deterministic
    /// (this lets the receipt chain compare task content).
    public func jsonData() throws -> Data {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(self)
    }

    /// Decode a task from JSON. The inverse of ``jsonData()``.
    public static func from(jsonData: Data) throws -> ProductivityTask {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(ProductivityTask.self, from: jsonData)
    }
}

// MARK: - String helpers

extension ProductivityTask {
    /// Encode to a UTF-8 string. Convenience for the task
    /// store (which deals in `String` bodies for the data
    /// layer).
    func jsonDataString() throws -> String {
        let data = try jsonData()
        guard let s = String(data: data, encoding: .utf8) else {
            throw ProductivityTaskStoreError.invalidTaskBody(reason: "encoded JSON is not UTF-8")
        }
        return s
    }

    /// Decode from a UTF-8 string. Inverse of ``jsonDataString()``.
    static func from(jsonDataString: String) throws -> ProductivityTask {
        guard let data = jsonDataString.data(using: .utf8) else {
            throw ProductivityTaskStoreError.invalidTaskBody(reason: "body is not valid UTF-8")
        }
        return try from(jsonData: data)
    }
}
