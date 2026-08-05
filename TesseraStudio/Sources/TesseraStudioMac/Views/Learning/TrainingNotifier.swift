import AppKit
import SwiftUI
import UserNotifications
import TesseraCore

/// Cross-window tally of whether the Learning destination is on screen.
/// Training runs are global (one orchestrator per app, not per window),
/// so visibility is an app-wide tally instead of a per-window flag: a
/// completion ping is suppressed while ANY window shows the surface.
@MainActor
final class LearningSurfaceTracker {
    static let shared = LearningSurfaceTracker()

    private var visibility: [UUID: Bool] = [:]

    var isVisible: Bool { visibility.values.contains(true) }

    func setVisible(_ visible: Bool, for window: UUID) {
        visibility[window] = visible
    }
}

/// Posts a local notification when a drafter-training cycle reaches a
/// terminal outcome the user might not be looking at.
///
/// Same routing rules as WorkflowRunNotifier:
/// - no ping while the Learning surface is visible in any window of the
///   frontmost app - the dashboard already carries the outcome
/// - skips are routine gate states (not enough traces, no model) and
///   never ping; only completed / validated / failed cycles do
/// - a user-initiated cancel is a deliberate foreground act and never
///   pings, exactly like a cancelled workflow run
/// - the notification permission prompt is deferred to the first post:
///   it is only justified once a training cycle has actually finished
@MainActor
enum TrainingNotifier {
    static func post(record: TesseraTrainingOrchestrator.TrainingRecord) {
        switch record.outcome {
        case .trainingCompleted, .dryRun, .trainingFailed:
            break
        case .skippedInsufficientTraces, .skippedNoModel, .guardPassed, .guardFailed:
            return
        }
        // The orchestrator reports a user cancellation as a failed record
        // with this note; treat it like a cancelled run: never ping.
        if record.note.hasPrefix("training cancelled") { return }
        guard !(NSApplication.shared.isActive && LearningSurfaceTracker.shared.isVisible) else { return }

        let content = content(for: record)
        Task {
            let center = UNUserNotificationCenter.current()
            let settings = await center.notificationSettings()
            switch settings.authorizationStatus {
            case .notDetermined:
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

    private static func content(
        for record: TesseraTrainingOrchestrator.TrainingRecord
    ) -> (title: String, body: String) {
        switch record.outcome {
        case .trainingCompleted:
            return ("Drafter training complete", record.note)
        case .dryRun:
            return ("Training setup validated", record.note)
        case .trainingFailed:
            return ("Drafter training failed", record.note)
        default:
            return ("Drafter training", record.note)
        }
    }

    private static func deliver(
        _ content: (title: String, body: String),
        via center: UNUserNotificationCenter
    ) async {
        let notification = UNMutableNotificationContent()
        notification.title = content.title
        notification.body = content.body
        notification.sound = .default
        // A post that reaches here always interrupts (explicit, not
        // defaulted): the outcome is long-running and not on screen.
        notification.interruptionLevel = .active
        let request = UNNotificationRequest(
            identifier: "tessera.training.cycle.\(UUID().uuidString)",
            content: notification,
            trigger: nil
        )
        // A missing bundle identifier (bare SwiftPM run) or a system-level
        // refusal surfaces as a thrown error; the dashboard already carries
        // the result, so swallow it.
        _ = try? await center.add(request)
    }
}
