import Foundation

// MARK: - ReminderFilter

/// The set of buckets the Reminders list view's sidebar /
/// tab bar offers. The list view's data flow is:
///   1. Pull every reminder via ``ReminderStore/list()``
///   2. Apply the chosen filter in memory
///   3. Sort by `triggerAt` ascending
///   4. Render the rows
///
/// The filter is in-memory because the data layer's
/// `listByEntityType` is a single query — running four
/// separate queries (one per filter) is wasteful, and the
/// reminders table is small (hundreds, not millions).
public enum ReminderFilter: String, CaseIterable, Identifiable, Sendable, Hashable {
    /// Reminders that haven't fired yet (`triggerAt` in the
    /// future) AND aren't snoozed AND haven't been
    /// acknowledged. The default filter when the surface
    /// opens.
    case upcoming
    /// Reminders the user has dismissed / acknowledged.
    case acknowledged
    /// Reminders the user has snoozed AND the snooze is
    /// still in the future.
    case snoozed
    /// Every reminder, regardless of state. Useful for the
    /// "where did that one go" sweep.
    case all

    public var id: String { rawValue }

    /// Human-readable name for the sidebar / tab label.
    public var displayName: String {
        switch self {
        case .upcoming: return "Upcoming"
        case .acknowledged: return "Acknowledged"
        case .snoozed: return "Snoozed"
        case .all: return "All"
        }
    }

    /// SF Symbol for the sidebar / tab icon.
    public var systemImage: String {
        switch self {
        case .upcoming: return "bell"
        case .acknowledged: return "checkmark.circle"
        case .snoozed: return "moon.zzz"
        case .all: return "tray"
        }
    }

    /// Apply this filter to a list of reminders. The
    /// `referenceDate` is injectable for tests (the default
    /// is `Date()`). The reminder's own `isUpcoming`,
    /// `isSnoozed`, and `isAcknowledged` accessors are
    /// `referenceDate`-aware so this is just bucketing.
    public func apply(
        to reminders: [Reminder],
        referenceDate: Date = Date()
    ) -> [Reminder] {
        switch self {
        case .upcoming:
            return reminders
                .filter { $0.isUpcoming(now: referenceDate) }
                .sorted { $0.triggerAt < $1.triggerAt }
        case .acknowledged:
            return reminders
                .filter { $0.isAcknowledged() }
                .sorted { ($0.acknowledgedAt ?? .distantPast) > ($1.acknowledgedAt ?? .distantPast) }
        case .snoozed:
            return reminders
                .filter { $0.isSnoozed(now: referenceDate) }
                .sorted { ($0.snoozedUntil ?? .distantPast) < ($1.snoozedUntil ?? .distantPast) }
        case .all:
            return reminders
                .sorted { $0.triggerAt < $1.triggerAt }
        }
    }
}

// MARK: - ReminderListViewModel

/// The view-model for the Reminders list view. The model is
/// a value type that owns the filter, the cached list, and
/// the loading state. The SwiftUI view binds to it and
/// re-renders when the published properties change.
///
/// The model is `Sendable` so the macOS / iOS view can hold
/// it on the main actor without isolation gymnastics; the
/// store calls happen in a `Task` and the result is assigned
/// back on the main actor.
public final class ReminderListViewModel: ObservableObject, @unchecked Sendable {

    /// The active filter. Bound to the sidebar's selection
    /// (macOS) or the tab bar (iOS).
    @Published public var filter: ReminderFilter = .upcoming

    /// Every reminder, sorted by `triggerAt` ascending. The
    /// filter buckets this in memory.
    @Published public private(set) var reminders: [Reminder] = []

    /// The set of reminders currently loading. The view
    /// shows a progress indicator when true.
    @Published public private(set) var isLoading: Bool = false

    /// The most recent load error, if any. The view shows
    /// a `ContentUnavailableView` with the message.
    @Published public private(set) var loadError: String?

    /// The id of the currently selected reminder. The
    /// detail view binds to this for NavigationSplitView /
    /// NavigationStack navigation.
    @Published public var selectedID: UUID?

    private let store: any ReminderStoring
    private let now: @Sendable () -> Date

    public init(
        store: any ReminderStoring,
        now: @escaping @Sendable () -> Date = { Date() }
    ) {
        self.store = store
        self.now = now
    }

    // MARK: - Loading

    /// Fetch the full reminder list from the store. Safe to
    /// call repeatedly; the result is sorted by `triggerAt`
    /// ascending so the list view can render without a
    /// second pass.
    public func load() async {
        isLoading = true
        loadError = nil
        defer { isLoading = false }
        do {
            reminders = try await store.list(limit: 1000)
        } catch {
            loadError = String(describing: error)
            reminders = []
        }
    }

    // MARK: - Derived

    /// The reminders that pass the current filter, in the
    /// filter's preferred order. The view binds to this
    /// rather than to `reminders` directly.
    public var filtered: [Reminder] {
        filter.apply(to: reminders, referenceDate: now())
    }

    // MARK: - Mutations (UI actions)

    /// Acknowledge a reminder. The store updates the row
    /// AND writes the receipt; the scheduler cancels the
    /// pending notification. The view reloads the list to
    /// pick up the new state.
    public func acknowledge(_ reminder: Reminder) async {
        do {
            _ = try await store.acknowledge(id: reminder.id, at: now())
            await load()
        } catch {
            loadError = String(describing: error)
        }
    }

    /// Snooze a reminder. The store updates `snoozedUntil`
    /// AND writes the receipt; the scheduler cancels + re-
    /// schedules.
    public func snooze(_ reminder: Reminder, until: Date) async {
        do {
            _ = try await store.snooze(id: reminder.id, until: until, at: now())
            await load()
        } catch {
            loadError = String(describing: error)
        }
    }

    /// Delete a reminder. The store removes the row AND
    /// writes the receipt; the scheduler cancels the
    /// notification.
    public func delete(_ reminder: Reminder) async {
        do {
            _ = try await store.delete(id: reminder.id)
            await load()
        } catch {
            loadError = String(describing: error)
        }
    }

    // MARK: - Relative time formatting

    /// Human-readable relative time, e.g. "in 15 min",
    /// "2 days ago", "now". Used by the list row. Pass a
    /// custom `now` for tests.
    public func relativeTime(
        for date: Date,
        relativeTo reference: Date? = nil
    ) -> String {
        let r = reference ?? now()
        let f = RelativeDateTimeFormatter()
        f.unitsStyle = .abbreviated
        return f.localizedString(for: date, relativeTo: r)
    }
}
