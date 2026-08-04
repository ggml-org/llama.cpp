import SwiftUI

/// The toolbar across the top of the Workflows surface.
/// Phase 2.5 enables all four buttons: New / Open / Save
/// (through ``WorkflowDocument``) and Run (through
/// ``WorkflowExecutor``). The toolbar's action closures are
/// owned by the parent `WorkflowsView`; this view is a pure
/// layout primitive so unit tests can render it without
/// spinning up an executor.
struct WorkflowToolbarView: View {
    let onNew: () -> Void
    let onOpen: () -> Void
    let onSave: () -> Void
    let onRun: () -> Void
    let isRunning: Bool

    var body: some View {
        HStack(spacing: 8) {
            Button(action: onNew) {
                Label("New", systemImage: "doc.badge.plus")
            }
            .help("New workflow")
            .disabled(isRunning)

            Button(action: onOpen) {
                Label("Open", systemImage: "folder")
            }
            .help("Open workflow from disk")
            .disabled(isRunning)

            Button(action: onSave) {
                Label("Save", systemImage: "square.and.arrow.down")
            }
            .help("Save workflow to disk")
            .disabled(isRunning)

            Spacer()

            if isRunning {
                ProgressView()
                    .controlSize(.small)
                    .padding(.trailing, 4)
            }

            Button(action: onRun) {
                Label("Run", systemImage: "play.fill")
            }
            .help("Run workflow")
            .disabled(isRunning)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.bar)
    }
}
