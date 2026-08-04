import Foundation

/// The terminal result of a workflow run. Parsed exactly once
/// at the run boundary (the executor's final `.finished`
/// event, or the editor's Cancel action) so every downstream
/// consumer (progress sheet, notifications, telemetry)
/// switches on a structured outcome instead of re-interpreting
/// raw event flags.
public enum WorkflowRunOutcome: Sendable, Equatable {
    case succeeded(summary: String?)
    case failed(message: String?)
    case cancelled(completedNodes: Int)

    /// Parse the executor's terminal event. Returns nil for
    /// non-terminal events: the run is still going and the
    /// caller must keep waiting, not guess an outcome.
    public init?(finishedEvent: WorkflowEvent) {
        guard case .finished(let success, let message) = finishedEvent else {
            return nil
        }
        self = success ? .succeeded(summary: message) : .failed(message: message)
    }

    /// The outcome for a run the user cancelled mid-flight.
    /// `events` is everything the executor surfaced before the
    /// cancel; `completedNodes` counts the nodes that finished
    /// successfully so the UI can report what actually got done.
    public static func cancelled(events: [WorkflowEvent]) -> WorkflowRunOutcome {
        let done = events.reduce(0) { acc, event in
            if case .nodeFinished(_, let success, _) = event, success {
                return acc + 1
            }
            return acc
        }
        return .cancelled(completedNodes: done)
    }

    public var isSucceeded: Bool {
        if case .succeeded = self { return true }
        return false
    }
}

/// Notification text for a terminal workflow run. A pure
/// function of the outcome + workflow name, so the notifier
/// and any future surface (banner, telemetry) cannot disagree
/// about what a run's result says.
public struct WorkflowRunNotificationContent: Sendable, Equatable {
    public let title: String
    public let body: String

    public init(outcome: WorkflowRunOutcome, workflowName: String) {
        switch outcome {
        case .succeeded:
            self.title = "Workflow finished"
            self.body = "\"\(workflowName)\" completed successfully."
        case .failed(let message):
            self.title = "Workflow failed"
            if let message {
                self.body = "\"\(workflowName)\": \(message)"
            } else {
                self.body = "\"\(workflowName)\" did not complete."
            }
        case .cancelled(let completedNodes):
            self.title = "Workflow cancelled"
            self.body = "\"\(workflowName)\" stopped after \(completedNodes) node(s)."
        }
    }
}
