import Foundation

// MARK: - HoldMode

/// The pause state of a document's chat queue (per spec §6.8).
/// The "Hold your horses" button is always present in the chat
/// panel footer; clicking it cycles the queue through these
/// states.
///
/// The state transitions are:
///
///   running → holdRequested   (user clicked "Hold your horses")
///   holdRequested → hold      (the in-flight item finished or
///                              was cancelled, and the dialog is up)
///   hold → resuming           (user clicked "Resume")
///   resuming → running        (agent picked up the new front)
///
/// `holdRequested` and `resuming` are transient states: they
/// exist to give the UI a moment to animate the transition and
/// to give the agent a chance to drain its in-flight work
/// before the hold takes effect. The state machine handles
/// these transitions atomically; the UI just observes the
/// current value.
public enum HoldMode: String, Codable, Sendable, Hashable, CaseIterable {
    /// Normal operation. The agent picks up pending items when
    /// idle.
    case running
    /// The user clicked "Hold your horses". The queue is still
    /// picking up the current in-flight item; new pending items
    /// are NOT picked up.
    case holdRequested
    /// The queue is paused. No new items will be picked up.
    case hold
    /// The user clicked "Resume". The agent is about to pick up
    /// the new front item.
    case resuming

    /// True iff the queue is in a state where the agent will not
    /// pick up a new pending item. Equivalent to "is paused".
    public var isPaused: Bool {
        switch self {
        case .running: return false
        case .holdRequested, .hold, .resuming: return true
        }
    }

    /// True iff the user has explicitly paused the queue. The
    /// transient states (holdRequested, resuming) are not
    /// "paused" for the user-facing UI — the user is in the
    /// middle of a transition, not a hold.
    public var isUserPaused: Bool {
        switch self {
        case .hold: return true
        case .running, .holdRequested, .resuming: return false
        }
    }

    /// Human-readable label for the chat panel footer button.
    public var footerButtonLabel: String {
        switch self {
        case .running: return "Hold your horses"
        case .holdRequested, .hold: return "Resume"
        case .resuming: return "Resume"
        }
    }
}
