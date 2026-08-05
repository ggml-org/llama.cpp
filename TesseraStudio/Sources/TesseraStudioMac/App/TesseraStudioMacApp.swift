import SwiftUI
import SwiftData
import TesseraCore

@main
struct TesseraStudioMacApp: App {
    let container: ModelContainer

    init() {
        TesseraSettings.registerDefaults()
        TesseraLearningServices.installDefaults()
        // "Plea the Fifth" composition root (phase 3). Loads
        // the user's covert trigger phrase from the Keychain
        // and installs the AppKit text-input swizzle so every
        // text input in the app feeds `CovertTriggerMonitor`.
        // The actual wipe executor (phase 2) is wired in here
        // once it lands; for now the `onFire` callback is left
        // unset so the trigger is silent.
        Task { await CovertTriggerMonitor.shared.loadFromKeychain() }
        TextInputInterceptor.install()
        // Idle training sweeps are global (one orchestrator per app), so
        // their completion ping is posted from here, not from a window.
        // The notifier suppresses the ping while the Learning surface is
        // visible and never pings for routine skips.
        TesseraLearningServices.trainingScheduler?.onFinished = { record in
            Task { @MainActor in
                TrainingNotifier.post(record: record)
            }
        }
        // If the encrypted volume is configured (Keychain has a
        // password + a bundle exists), mount it on launch and repoint
        // the data root redirector. This is the "subsequent launch"
        // path from the design spec (Section 6.5).
        TesseraAppLifecycle.bootstrapEncryptedVolume()
        do {
            let schema = Schema([ChatMessage.self, RunRecord.self, Conversation.self])
            let storeURL = TesseraDataRoot.appSupportStoreFile()
            let config = ModelConfiguration("TesseraStudio", schema: schema, url: storeURL)
            container = try ModelContainer(for: schema, configurations: [config])
        } catch {
            fatalError("Failed to create ModelContainer: \(error)")
        }
    }

    var body: some Scene {
        WindowGroup {
            // SwiftUI's `EnvironmentValues.undoManager` setter is
            // not public — the framework populates a per-window
            // UndoManager for us. Views that need custom undo
            // (e.g. WorkflowsView) read it via
            // `@Environment(\.undoManager)` and call
            // `registerUndo(withTarget:handler:)`. The system
            // Edit menu's Undo/Redo items auto-dispatch to it.
            ContentView()
        }
        .modelContainer(container)
        .defaultSize(width: 1200, height: 800)
        .commands {
            // Add a Workflows-aware File > New Workflow AFTER the
            // system's New Window item. Replacing `.newItem` would
            // delete File > New Window (Cmd-Shift-N), and each
            // WindowGroup window is an independent studio surface
            // (own sidebar selection, own agent loop, own
            // workflow editor store), so multi-window is supported.
            CommandGroup(after: .newItem) {
                NewWorkflowMenuItem()
            }

            // Insert Open / Save / Save As after the system Save
            // group. Cmd-O opens a workflow JSON, Cmd-S saves the
            // current workflow, Shift-Cmd-S presents a separate
            // Save As panel and re-baselines at the chosen URL.
            CommandGroup(after: .saveItem) {
                Divider()
                OpenWorkflowMenuItem()
                SaveWorkflowMenuItem()
                SaveAsWorkflowMenuItem()
            }

            // Help menu (Cmd-? auto-binds). Three items: docs,
            // release notes, sample workflows.
            CommandGroup(replacing: .help) {
                HelpMenuItems()
            }

            // View menu: node palette + inspector toggles (live
            // only when a Workflows surface is focused) and the
            // telemetry drawer toggle. Inserted after the system
            // sidebar group so the standard items stay put.
            CommandGroup(after: .sidebar) {
                ViewMenuItems()
            }

            // Edit menu's Undo/Redo are auto-bound by the system
            // to the first responder's UndoManager. Views that
            // need undoable mutations read `@Environment(\.undoManager)`
            // and call `registerUndo(withTarget:handler:)` +
            // `setActionName(_:)`. No custom command group needed.
        }

        Settings {
            SettingsView()
        }
    }
}
