import Foundation

// MARK: - ProductivityTaskFilter

/// In-memory filter + sort logic for the Tasks surface.
/// The store reads all tasks from the data layer and
/// filters them client-side; the predicates here are
/// the source of truth for what "Today" / "Upcoming" /
/// etc. mean.
///
/// **Why client-side:** the data layer's hybrid_search
/// is the cross-cutting search; the per-list filters
/// are simple (due date within X hours for Today,
/// list field == .anytime for Anytime, etc.) and don't
/// justify a server round-trip for 1k+ rows. The data
/// layer's `idx_entities_task_list` and
/// `idx_entities_task_due` partial indexes (migration
/// 0004_tasks.sql) make the SQL scan cheap, but the
/// client-side filter is what the UI binds to.
public enum ProductivityTaskFilter {

    /// `true` iff the task belongs in the given list,
    /// evaluated at `now`.
    ///
    /// - `inbox` / `anytime` / `someday`: match by
    ///   `list` field.
    /// - `today`: due in the next 24h, not completed.
    ///   Overdue tasks are also in Today (per the
    ///   spec's "Today auto-populates from due date,
    ///   including overdue" rule).
    /// - `upcoming`: due in the next 7 days but NOT in
    ///   the next 24h, not completed.
    public static func isIn(
        _ task: ProductivityTask,
        list: ProductivityTask.List,
        asOf now: Date = Date()
    ) -> Bool {
        switch list {
        case .inbox, .anytime, .someday:
            return task.list == list
        case .today:
            return task.isDueWithin24Hours(asOf: now)
        case .upcoming:
            return task.isDueWithin7DaysButNotToday(asOf: now)
        }
    }

    /// Sort a list of tasks for the given list view.
    /// - `today` / `upcoming`: by due date ascending.
    /// - `inbox` / `anytime` / `someday`: by priority
    ///   desc, then title asc.
    public static func sortForList(
        _ tasks: [ProductivityTask],
        list: ProductivityTask.List,
        asOf now: Date = Date()
    ) -> [ProductivityTask] {
        switch list {
        case .today, .upcoming:
            return tasks.sorted { $0.dueAt ?? now < $1.dueAt ?? now }
        case .inbox, .anytime, .someday:
            return tasks.sorted { a, b in
                if a.priority != b.priority { return a.priority > b.priority }
                return a.title.localizedCaseInsensitiveCompare(b.title) == .orderedAscending
            }
        }
    }
}
