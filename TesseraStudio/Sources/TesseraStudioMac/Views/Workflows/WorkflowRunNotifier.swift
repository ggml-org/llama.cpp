import AppKit
import UserNotifications
import TesseraCore

/// Posts a local notification when a workflow run reaches a
/// terminal outcome the user might not be looking at.
///
/// Two HIG rules shape this:
/// - The permission prompt is deferred to the FIRST completed
///   run, not app launch: the request is only justified once
///   the user has actually started a long-running workflow.
///   A denial is remembered by the system and never re-asked.
/// - No notification while the run surface is visible to the
///   user (app frontmost AND the run's own window showing the
///   Workflows destination): the sheet already carries the
///   outcome, and notifying for something on screen is noise.
///   Runs are scene-lived, so a run can finish while its window
///   has swapped to another destination - that case DOES ping.
@MainActor
enum WorkflowRunNotifier {
    /// Notify about a terminal run outcome. Cancelled runs are
    /// not posted: a cancel is always a deliberate foreground
    /// act, so there is nothing to pull the user back to.
    /// `runSurfaceVisible` says whether the run's own surface
    /// (the Workflows destination of its window) is on screen;
    /// the ping fires only when the outcome is NOT visible.
    static func post(
        outcome: WorkflowRunOutcome,
        workflowName: String,
        runSurfaceVisible: Bool
    ) {
        switch outcome {
        case .succeeded, .failed:
            break
        case .cancelled:
            return
        }
        guard !(NSApplication.shared.isActive && runSurfaceVisible) else { return }
        let content = WorkflowRunNotificationContent(
            outcome: outcome, workflowName: workflowName
        )
        Task {
            let center = UNUserNotificationCenter.current()
            let settings = await center.notificationSettings()
            switch settings.authorizationStatus {
            case .notDetermined:
                // First completed run: the moment the permission
                // request becomes justified.
                let granted = (try? await center.requestAuthorization(
                    options: [.alert, .sound]
                )) ?? false
                guard granted else { return }
                await deliver(content, via: center)
            case .authorized, .provisional, .ephemeral:
                await deliver(content, via: center)
            case .denied:
                return
            @unknown default:
                return
            }
        }
    }

    private static func deliver(
        _ content: WorkflowRunNotificationContent,
        via center: UNUserNotificationCenter
    ) async {
        let notification = UNMutableNotificationContent()
        notification.title = content.title
        notification.body = content.body
        notification.sound = .default
        // Routing (HIG 14.12): suppress only when frontmost + run surface visible, ping otherwise, cancels never ping - a post that reaches here always interrupts (explicit, not defaulted).
        notification.interruptionLevel = .active
        let request = UNNotificationRequest(
            identifier: "tessera.workflow.run.\(UUID().uuidString)",
            content: notification,
            trigger: nil
        )
        // A missing bundle identifier (bare SwiftPM run) or a
        // system-level refusal surfaces as a thrown error; the
        // run sheet already carries the result, so swallow it.
        _ = try? await center.add(request)
    }
}
