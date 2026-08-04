import SwiftUI
import SwiftData
import TesseraCore

@main
struct TesseraStudioMacApp: App {
    let container: ModelContainer

    init() {
        TesseraSettings.registerDefaults()
        TesseraLearningServices.installDefaults()
        do {
            let schema = Schema([ChatMessage.self, RunRecord.self, Conversation.self])
            let config = ModelConfiguration("TesseraStudio", schema: schema)
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
            // Replace File > New with a Workflows-aware version
            // (Cmd-N creates a new workflow; no document launcher
            // because Tessera Studio is single-window for now).
            CommandGroup(replacing: .newItem) {
                NewWorkflowMenuItem()
            }

            // Insert Open / Save / Save As after the system Save
            // group. Cmd-O opens a workflow JSON, Cmd-S saves the
            // current workflow, Shift-Cmd-S does a Save As (same
            // code path today; FileDocument handles the rename).
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
