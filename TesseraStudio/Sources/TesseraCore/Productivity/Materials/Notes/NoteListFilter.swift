import Foundation

// MARK: - NoteListFilter

/// Which list the user is currently looking at in the notes
/// surface. Mirrors the three tabs in the Bear UI: All, Pinned,
/// Archived. The filter is a small enum so the macOS sidebar and
/// the iOS tab bar can drive the same `NotesViewModel` without
/// duplicating the filter vocabulary.
public enum NoteListFilter: String, Codable, Sendable, Hashable, CaseIterable, Identifiable {
    /// Every note (excluding the ones that are filtered by
    /// the active tag chip, when set). Sorted by `updated_at
    /// DESC`.
    case all
    /// Pinned + non-archived notes, sorted by `pinned_at DESC`.
    case pinned
    /// Archived notes, sorted by `archived_at DESC`.
    case archived

    public var id: String { rawValue }

    /// Display label for the list header / tab.
    public var displayName: String {
        switch self {
        case .all: return "All"
        case .pinned: return "Pinned"
        case .archived: return "Archived"
        }
    }

    /// SF Symbol for the tab / sidebar row.
    public var systemImage: String {
        switch self {
        case .all: return "note.text"
        case .pinned: return "pin.fill"
        case .archived: return "archivebox"
        }
    }

    /// Apply this filter to a list of notes, returning the rows
    /// the view should render, in the right sort order. The
    /// sort order is per-filter:
    ///   * `.all` — `updated_at DESC`
    ///   * `.pinned` — `pinned_at DESC`, then `updated_at DESC`
    ///   * `.archived` — `archived_at DESC`, then `updated_at DESC`
    public func apply(to notes: [Note]) -> [Note] {
        switch self {
        case .all:
            return notes
                .filter { !$0.isArchived }
                .sorted { $0.updatedAt > $1.updatedAt }
        case .pinned:
            return notes
                .filter { $0.isPinned && !$0.isArchived }
                .sorted { (lhs, rhs) in
                    let l = lhs.pinnedAt ?? .distantPast
                    let r = rhs.pinnedAt ?? .distantPast
                    if l != r { return l > r }
                    return lhs.updatedAt > rhs.updatedAt
                }
        case .archived:
            return notes
                .filter { $0.isArchived }
                .sorted { (lhs, rhs) in
                    let l = lhs.archivedAt ?? .distantPast
                    let r = rhs.archivedAt ?? .distantPast
                    if l != r { return l > r }
                    return lhs.updatedAt > rhs.updatedAt
                }
        }
    }
}

// MARK: - NoteRow

/// A flattened view-model row for the notes list. Computed
/// from a `Note` so the view layer doesn't have to know about
/// date formatting / relative time strings. The `relativeTime`
/// string is computed at construction time (the view doesn't
/// refresh its labels automatically — a refresh on `onAppear`
/// is enough for v1; v2 will move to a TimelineView for live
/// relative time).
public struct NoteRow: Identifiable, Sendable, Hashable {
    public let id: UUID
    public let title: String
    public let snippet: String
    public let tags: [String]
    public let updatedAt: Date
    public let relativeTime: String
    public let isPinned: Bool
    public let isArchived: Bool
    public let wordCount: Int

    public init(note: Note, now: Date = Date()) {
        self.id = note.id
        self.title = note.displayTitle
        self.snippet = note.snippet(maxLength: 200)
        self.tags = note.tags
        self.updatedAt = note.updatedAt
        self.relativeTime = NoteRow.relativeTimeString(for: note.updatedAt, now: now)
        self.isPinned = note.isPinned
        self.isArchived = note.isArchived
        self.wordCount = note.wordCount
    }

    /// Human-readable relative time. Picks the most natural
    /// unit (just now, N min ago, N hr ago, yesterday, N days
    /// ago, N weeks ago, formatted date). The formatter
    /// matches the spec's "edited N days ago" requirement.
    public static func relativeTimeString(for date: Date, now: Date) -> String {
        let delta = now.timeIntervalSince(date)
        if delta < 0 { return "just now" }
        if delta < 60 { return "just now" }
        let minutes = Int(delta / 60)
        if minutes < 60 { return "\(minutes) min ago" }
        let hours = Int(delta / 3600)
        if hours < 24 { return "\(hours) hr ago" }
        let calendar = Calendar.current
        if calendar.isDateInYesterday(date) { return "yesterday" }
        let days = Int(delta / 86400)
        if days < 14 { return "\(days) days ago" }
        let weeks = days / 7
        if weeks < 5 { return "\(weeks) weeks ago" }
        // Older: format as "Jan 12" (current year) or "Jan 12, 2024".
        let formatter = DateFormatter()
        let isCurrentYear = calendar.component(.year, from: date) == calendar.component(.year, from: now)
        formatter.dateFormat = isCurrentYear ? "MMM d" : "MMM d, yyyy"
        return formatter.string(from: date)
    }
}
