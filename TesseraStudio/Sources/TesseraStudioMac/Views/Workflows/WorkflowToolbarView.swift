import SwiftUI

/// The toolbar across the top of the Workflows surface.
/// Phase 2.1 ships the four buttons with stubs (disabled where
/// the behaviour lands in a later sub-step):
///
/// - **New**: clears the document, generates a new id.
/// - **Open**: presents an NSOpenPanel for `.tessera-workflow`
///   (added in 2.4 alongside the document type).
/// - **Save**: serialises the current workflow + positions to
///   JSON and presents an NSSavePanel (2.4).
/// - **Run**: validates and runs the workflow through
///   ``WorkflowExecutor.run`` (2.5).
///
/// The toolbar's action closures are owned by the parent
/// `WorkflowsView`; this view is a pure layout primitive so
/// unit tests can render it without spinning up a
/// `WorkflowExecutor`.
struct WorkflowToolbarView: View {
    let onNew: () -> Void
    let onOpen: () -> Void
    let onSave: () -> Void
    let onRun: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            Button(action: onNew) {
                Label("New", systemImage: "doc.badge.plus")
            }
            .help("New workflow")

            Button(action: onOpen) {
                Label("Open", systemImage: "folder")
            }
            .help("Open workflow from disk")
            .disabled(true) // Phase 2.4

            Button(action: onSave) {
                Label("Save", systemImage: "square.and.arrow.down")
            }
            .help("Save workflow to disk")
            .disabled(true) // Phase 2.4

            Spacer()

            Button(action: onRun) {
                Label("Run", systemImage: "play.fill")
            }
            .help("Run workflow")
            .disabled(true) // Phase 2.5
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.bar)
    }
}
