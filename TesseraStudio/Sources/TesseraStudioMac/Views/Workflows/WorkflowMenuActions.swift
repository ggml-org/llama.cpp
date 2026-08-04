import SwiftUI

/// Actions that the Workflows view publishes to the focused scene
/// so the App-level menu commands (File > New / Open / Save and
/// the View menu toggles) can dispatch into the live view.
/// Without this plumbing, menu commands can't reach the view's
/// mutable state.
struct WorkflowMenuActions {
    var new: () -> Void
    var open: () -> Void
    var save: () -> Void
    var saveAs: () -> Void
    var canSave: () -> Bool
    var togglePalette: () -> Void
    var paletteVisible: () -> Bool
    var toggleInspector: () -> Void
    var inspectorVisible: () -> Bool

    static let unavailable = WorkflowMenuActions(
        new: {},
        open: {},
        save: {},
        saveAs: {},
        canSave: { false },
        togglePalette: {},
        paletteVisible: { false },
        toggleInspector: {},
        inspectorVisible: { false }
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

/// Telemetry drawer controls, published by the shell (ContentView)
/// so the View menu can toggle the drawer from any destination.
struct TelemetryMenuActions {
    var toggle: () -> Void
    var isExpanded: () -> Bool
}

private struct TelemetryMenuActionsKey: FocusedValueKey {
    typealias Value = TelemetryMenuActions
}

extension FocusedValues {
    var telemetryMenuActions: TelemetryMenuActions? {
        get { self[TelemetryMenuActionsKey.self] }
        set { self[TelemetryMenuActionsKey.self] = newValue }
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

/// File > Save As. `Shift-Cmd-S`. Presents its own save panel
/// (distinct from Save's) and records the chosen URL as the new
/// saved baseline.
struct SaveAsWorkflowMenuItem: View {
    @FocusedValue(\.workflowMenuActions) private var actions

    var body: some View {
        Button("Save As…") { actions?.saveAs() }
            .keyboardShortcut("s", modifiers: [.command, .shift])
            .disabled(actions == nil || (actions?.canSave() == false))
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

/// View menu items. The Workflows-surface toggles (node palette,
/// parameter inspector) need a focused WorkflowsView and stay
/// disabled otherwise; the telemetry drawer toggle is always
/// available. Titles flip Show/Hide off the live state, so the
/// menu reads as a statement of what will happen.
struct ViewMenuItems: View {
    @FocusedValue(\.workflowMenuActions) private var workflowActions
    @FocusedValue(\.telemetryMenuActions) private var telemetryActions

    var body: some View {
        Button(workflowActions?.paletteVisible() == true
               ? "Hide Node Palette" : "Show Node Palette") {
            workflowActions?.togglePalette()
        }
        .disabled(workflowActions == nil)

        Button(workflowActions?.inspectorVisible() == true
               ? "Hide Inspector" : "Show Inspector") {
            workflowActions?.toggleInspector()
        }
        .keyboardShortcut("i", modifiers: [.command, .option])
        .disabled(workflowActions == nil)

        Divider()

        Button(telemetryActions?.isExpanded() == true
               ? "Hide Telemetry" : "Show Telemetry") {
            telemetryActions?.toggle()
        }
        .disabled(telemetryActions == nil)
    }
}
