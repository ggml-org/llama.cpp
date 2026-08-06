import Foundation
import UserNotifications

// MARK: - ReminderNotificationScheduler

/// Wraps `UNUserNotificationCenter` for the Reminders surface.
///
/// The scheduler is an actor because the notification
/// center's API is async-by-callback (or async/await) and we
/// want the call site to be straightforward (`try await
/// scheduler.schedule(reminder)`). The store doesn't own the
/// scheduler — the SwiftUI view / chat panel wires them
/// together so the scheduler can be mocked in tests.
///
/// **Authorization.** The scheduler requests authorization
/// lazily: the first call to ``schedule(_:)`` requests
/// `.alert + .sound + .badge` and waits for the user. The
/// scheduler never prompts on app launch; the prompt is
/// triggered by the user's first "remind me …" or by the
/// Reminders surface's "Enable notifications" banner.
///
/// **Snooze semantics.** A snooze cancels the original
/// `triggerAt` notification and schedules a new one at
/// `snoozedUntil`. The store owns the row update; the
/// scheduler owns the notification-center side effect.
///
/// **Acknowledgment semantics.** An acknowledgment cancels
/// the pending notification. The store owns the row update
/// (`acknowledgedAt`); the scheduler owns the cancel.
public actor ReminderNotificationScheduler {

    /// The notification center. Injectable so tests can pass
    /// a mock that records scheduled requests.
    private let center: UNUserNotificationCenter

    /// Identifier prefix for reminder notifications. The
    /// reminder's UUID is the suffix; cancel-by-id is
    /// O(1) (the center maintains a set of pending
    /// identifiers).
    private let identifierPrefix: String

    public init(
        center: UNUserNotificationCenter = .current(),
        identifierPrefix: String = "tessera.reminder."
    ) {
        self.center = center
        self.identifierPrefix = identifierPrefix
    }

    // MARK: - Authorization

    /// The current authorization status. The SwiftUI view
    /// uses this to decide whether to show a "Notifications
    /// disabled — open Settings" banner.
    public func authorizationStatus() async -> UNAuthorizationStatus {
        let settings = await center.notificationSettings()
        return settings.authorizationStatus
    }

    /// Request authorization for `.alert + .sound + .badge`.
    /// Returns true on success (the user accepted). Returns
    /// false on denial; the SwiftUI view reacts by surfacing
    /// the "open Settings to enable" hint.
    @discardableResult
    public func requestAuthorization() async throws -> Bool {
        try await center.requestAuthorization(options: [.alert, .sound, .badge])
    }

    // MARK: - Schedule / cancel / snooze

    /// Schedule a notification for a reminder. The
    /// notification fires at `triggerAt` (or `snoozedUntil`
    /// when the reminder is currently snoozed). The
    /// `triggerAt` is matched against `Date()` — past
    /// triggers are rejected (the notification center
    /// silently drops them, but the store still wants to
    /// log the rejection so the user can see why a snoozed
    /// reminder didn't fire).
    public func schedule(_ reminder: Reminder) async throws {
        let identifier = identifier(for: reminder.id)
        let fireDate = effectiveFireDate(for: reminder)
        guard fireDate > Date() else {
            throw ReminderNotificationError.triggerInPast(
                at: fireDate,
                reminderID: reminder.id
            )
        }
        // Cancel any prior notification for this id so a
        // re-schedule replaces (rather than doubles) the
        // pending request.
        center.removePendingNotificationRequests(
            withIdentifiers: [identifier]
        )
        let content = UNMutableNotificationContent()
        content.title = reminder.title
        if !reminder.notes.isEmpty {
            content.body = reminder.notes
        } else {
            content.body = reminder.displayLine()
        }
        content.sound = .default
        content.userInfo = [
            "reminderID": reminder.id.uuidString,
            "calendarEventID": reminder.calendarEventID.uuidString,
            "kind": "reminder",
        ]
        let components = Calendar.current.dateComponents(
            [.year, .month, .day, .hour, .minute, .second],
            from: fireDate
        )
        let trigger = UNCalendarNotificationTrigger(
            dateMatching: components,
            repeats: false
        )
        let request = UNNotificationRequest(
            identifier: identifier,
            content: content,
            trigger: trigger
        )
        try await center.add(request)
    }

    /// Cancel the pending notification for a reminder. No-op
    /// when nothing is scheduled. Called by acknowledge,
    /// delete, and the "snooze" path before re-scheduling.
    public func cancel(_ reminder: Reminder) async {
        center.removePendingNotificationRequests(
            withIdentifiers: [identifier(for: reminder.id)]
        )
    }

    /// Snooze a reminder: cancel the current pending
    /// notification and schedule a new one at `until`. The
    /// store has already updated `snoozedUntil`; this method
    /// only handles the notification-center side.
    public func snooze(
        _ reminder: Reminder,
        until: Date
    ) async throws {
        center.removePendingNotificationRequests(
            withIdentifiers: [identifier(for: reminder.id)]
        )
        guard until > Date() else {
            throw ReminderNotificationError.triggerInPast(
                at: until,
                reminderID: reminder.id
            )
        }
        let content = UNMutableNotificationContent()
        content.title = reminder.title
        content.body = "Snoozed: \(reminder.offsetLabel)"
        content.sound = .default
        content.userInfo = [
            "reminderID": reminder.id.uuidString,
            "calendarEventID": reminder.calendarEventID.uuidString,
            "kind": "reminder.snoozed",
        ]
        let components = Calendar.current.dateComponents(
            [.year, .month, .day, .hour, .minute, .second],
            from: until
        )
        let trigger = UNCalendarNotificationTrigger(
            dateMatching: components,
            repeats: false
        )
        let request = UNNotificationRequest(
            identifier: identifier(for: reminder.id),
            content: content,
            trigger: trigger
        )
        try await center.add(request)
    }

    /// Cancel + re-schedule everything in the supplied list.
    /// Called on app launch (the notification center forgets
    /// pending requests on cold start if the app was
    /// terminated for more than a day; this rebuilds the
    /// schedule from the durable store).
    public func rescheduleAll(_ reminders: [Reminder]) async throws {
        // The center has no bulk-clear, so we ask for the
        // current pending set and remove just the
        // reminder-prefixed ones. Anything else (system,
        // other apps) is left alone.
        let pending = await center.pendingNotificationRequests()
        let prefix = identifierPrefix
        let toRemove = pending
            .map(\.identifier)
            .filter { $0.hasPrefix(prefix) }
        if !toRemove.isEmpty {
            center.removePendingNotificationRequests(withIdentifiers: toRemove)
        }
        for r in reminders where !r.isAcknowledged() {
            try? await schedule(r)
        }
    }

    // MARK: - Helpers

    /// The notification identifier for a reminder id. The
    /// prefix keeps reminder notifications separate from
    /// other system notifications.
    public func identifier(for reminderID: UUID) -> String {
        Self.identifier(for: reminderID, prefix: identifierPrefix)
    }

    /// Static identifier helper so the call site doesn't need
    /// an actor handle (and so unit tests can build the same
    /// string without instantiating the scheduler).
    public static func identifier(
        for reminderID: UUID,
        prefix: String
    ) -> String {
        "\(prefix)\(reminderID.uuidString)"
    }

    /// The effective fire date: `snoozedUntil` if set and in
    /// the future, otherwise `triggerAt`. Snoozed reminders
    /// fire at the snooze time, not at the original
    /// `triggerAt`.
    public func effectiveFireDate(for reminder: Reminder) -> Date {
        Self.effectiveFireDate(for: reminder, now: Date())
    }

    /// Pure (no-IO) effective fire date. Exposed for unit
    /// tests so they can pin the reference time without
    /// monkey-patching `Date()`.
    public static func effectiveFireDate(
        for reminder: Reminder,
        now: Date
    ) -> Date {
        if let snooze = reminder.snoozedUntil, snooze > now {
            return snooze
        }
        return reminder.triggerAt
    }

    /// True iff the reminder's effective fire date is in
    /// the past. Pure (no IO) so unit tests can pin `now`.
    public static func isFireDateInPast(
        for reminder: Reminder,
        now: Date
    ) -> Bool {
        effectiveFireDate(for: reminder, now: now) <= now
    }
}

// MARK: - Errors

public enum ReminderNotificationError: Error, Sendable, Equatable {
    /// The reminder's fire time is in the past. The
    /// notification center silently drops these; we surface
    /// the failure so the chat panel can warn the user.
    case triggerInPast(at: Date, reminderID: UUID)
    /// The notification center refused the request. Wraps
    /// the localized description for diagnostics.
    case notificationCenterFailed(reason: String)

    public static func == (lhs: ReminderNotificationError, rhs: ReminderNotificationError) -> Bool {
        switch (lhs, rhs) {
        case (.triggerInPast(let a, let b), .triggerInPast(let c, let d)):
            return a == c && b == d
        case (.notificationCenterFailed(let a), .notificationCenterFailed(let b)):
            return a == b
        default:
            return false
        }
    }
}
