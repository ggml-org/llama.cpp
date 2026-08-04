import SwiftUI

/// Actions that the Workflows view publishes to the focused scene
/// so the App-level menu commands (File > New / Open / Save) can
/// dispatch into the live view. Without this plumbing, menu
/// commands can't reach the view's mutable state.
struct WorkflowMenuActions {
    var new: () -> Void
    var open: () -> Void
    var save: () -> Void
    var canSave: () -> Bool

    static let unavailable = WorkflowMenuActions(
        new: {},
        open: {},
        save: {},
        canSave: { false }
    )
}

private struct WorkflowMenuActionsKey: FocusedValueKey {
    typealias Value = WorkflowMenuActions
}

extension FocusedValues {
    var workflowMenuActions: WorkflowMenuActions? {
        get { self[WorkflowMenuActionsKey.self] }
        set { self[WorkflowMenuActionsKey.self] = newValue }
    }
}

/// File > New Workflow. Disabled when no WorkflowsView is in the
/// focused scene. `Cmd-N` auto-binds via `.keyboardShortcut`.
struct NewWorkflowMenuItem: View {
    @FocusedValue(\.workflowMenuActions) private var actions

    var body: some View {
        Button("New Workflow") { actions?.new() }
            .keyboardShortcut("n", modifiers: .command)
            .disabled(actions == nil)
    }
}

/// File > Open Workflow. `Cmd-O`.
struct OpenWorkflowMenuItem: View {
    @FocusedValue(\.workflowMenuActions) private var actions

    var body: some View {
        Button("Open Workflow…") { actions?.open() }
            .keyboardShortcut("o", modifiers: .command)
            .disabled(actions == nil)
    }
}

/// File > Save. `Cmd-S`. Disabled when there's nothing to save
/// (no nodes added).
struct SaveWorkflowMenuItem: View {
    @FocusedValue(\.workflowMenuActions) private var actions

    var body: some View {
        Button("Save") { actions?.save() }
            .keyboardShortcut("s", modifiers: .command)
            .disabled(actions == nil || (actions?.canSave() == false))
    }
}

/// File > Save As. `Shift-Cmd-S`.
struct SaveAsWorkflowMenuItem: View {
    @FocusedValue(\.workflowMenuActions) private var actions

    var body: some View {
        Button("Save As…") { actions?.save() }
            .keyboardShortcut("s", modifiers: [.command, .shift])
            .disabled(actions == nil)
    }
}

/// Help > Tessera Studio Help. `Cmd-?` (auto-bound by the
/// CommandGroup(replacing: .help)).
struct HelpMenuItems: View {
    private let docsURL = URL(string: "https://github.com/tessera/tessera/blob/main/docs/tessera-studio-design.md")
    private let releaseNotesURL = URL(string: "https://github.com/tessera/tessera/releases")
    private let samplesURL = URL(string: "https://github.com/tessera/tessera/tree/main/TesseraStudio/Examples")

    var body: some View {
        Button("Tessera Studio Help") {
            if let url = docsURL { NSWorkspace.shared.open(url) }
        }
        .disabled(docsURL == nil)

        Divider()

        Button("Release Notes") {
            if let url = releaseNotesURL { NSWorkspace.shared.open(url) }
        }
        .disabled(releaseNotesURL == nil)

        Button("Open Sample Workflows") {
            if let url = samplesURL { NSWorkspace.shared.open(url) }
        }
        .disabled(samplesURL == nil)
    }
}
