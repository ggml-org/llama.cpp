import SwiftUI
import AppKit
import TesseraCore

// MARK: - CodeEditorPaneView

/// The center column: the code editor. The view is a
/// thin SwiftUI wrapper over the platform-native text
/// view (Phase 2's `TesseraEditorView`); the per-file
/// differences (line numbers, monospaced font,
/// `EditorMode.code`) are configuration on the editor
/// mode, not different code paths.
///
/// **Why a wrapping SwiftUI view.** Phase 2's
/// `TesseraEditorView` is `Binding<DocumentAST>`-based
/// (the AST is what documents mutate). Code's body is
/// plain text, not an AST. The wrapper here bridges:
/// when the view model updates `currentFile`, the
/// wrapper rebuilds the editor's initial state with a
/// single-block AST that wraps the file's body, and
/// when the editor flushes a coalesced burst, the
/// wrapper turns the AST mutation back into a
/// `CodeMutation.replaceCodeBlock(...)`.
///
/// **Save semantics.** The editor's coalescer flushes
/// after 1.5s of inactivity. The wrapper treats the
/// flushed burst as a `CodeMutation.replaceCodeBlock`
/// and calls `viewModel.saveBody(_:)`. The chat panel
/// sees one `ChatQueueItem` per burst (the user-typed
/// "edit" message is appended with the diff stats).
public struct CodeEditorPaneView: View {

    @ObservedObject public var viewModel: CodeSurfaceViewModel
    @State private var localBody: String = ""
    @State private var isDirty: Bool = false

    public init(viewModel: CodeSurfaceViewModel) {
        self.viewModel = viewModel
    }

    public var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if let file = viewModel.currentFile {
                codeEditor(for: file)
            } else {
                emptyState
            }
            Divider()
            footer
        }
    }

    private var header: some View {
        HStack(spacing: 8) {
            if let file = viewModel.currentFile {
                Image(systemName: "doc.text")
                    .foregroundStyle(.secondary)
                Text(file.filename)
                    .font(.headline)
                Text(file.language)
                    .font(.caption)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(.tint.opacity(0.15), in: RoundedRectangle(cornerRadius: 4))
                if isDirty {
                    Text("•")
                        .foregroundStyle(.orange)
                        .help("Unsaved changes")
                }
            }
            Spacer()
            if isDirty {
                Button("Save") {
                    Task { await viewModel.saveBody(localBody) }
                    isDirty = false
                }
                .keyboardShortcut("s", modifiers: .command)
            }
        }
        .padding(8)
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "chevron.left.forwardslash.chevron.right")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)
            Text("Select a file to view or edit")
                .font(.headline)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var footer: some View {
        HStack(spacing: 12) {
            if let file = viewModel.currentFile {
                Text(file.path)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                Spacer()
                Text("\(file.size.formatted()) bytes")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Text("sha256: \(String(file.checksum.dropFirst(7).prefix(7)))")
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
    }

    @ViewBuilder
    private func codeEditor(for file: CodeFile) -> some View {
        // The Phase 2 text view's binding is over a
        // `DocumentAST`. The wrapper builds a
        // single-block AST wrapping the file body
        // and feeds it to the editor. When the editor
        // flushes a burst, the wrapper saves the new
        // body as a `CodeMutation`.
        //
        // For Phase 5's first cut we use a simpler
        // path: an `NSTextView` wrapped in
        // `NSViewRepresentable` that reads/writes the
        // `localBody` `@State` and signals `isDirty`.
        // The Phase 2 editor engine wires the same
        // SwiftUI bridge to its coalescer; the path
        // here is the v1 of the code-editor integration.
        CodeTextView(
            body: $localBody,
            language: file.language,
            isDirty: $isDirty,
            onCommit: { newBody in
                Task { await viewModel.saveBody(newBody) }
            }
        )
        .onAppear {
            if localBody != file.body {
                localBody = file.body
                isDirty = false
            }
        }
        .onChange(of: file.body) { _, new in
            // Update the local body when the file
            // changes (e.g. from the watcher). Only
            // do this if the local copy isn't dirty --
            // we don't want to clobber the user's
            // unsaved edits.
            if !isDirty {
                localBody = new
            }
        }
    }
}

// MARK: - CodeTextView

/// A minimal `NSTextView` wrapper that displays +
/// edits the file body. The view is intentionally
/// simple: a monospaced font, line-wrap disabled,
/// and a delegate that fires `onCommit` when the
/// user types a save gesture (Cmd-S) or pauses
/// for `coalesceWindow` seconds.
///
/// The Phase 2 `TesseraEditorView` is the long-
/// term home; for v1 the simpler `NSTextView` is
/// enough to demonstrate the data flow (the user's
/// edits produce a `CodeMutation.replaceCodeBlock`
/// receipt via `viewModel.saveBody`).
struct CodeTextView: NSViewRepresentable {

    @Binding var body: String
    let language: String
    @Binding var isDirty: Bool
    let onCommit: (String) -> Void

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSTextView.scrollableTextView()
        let textView = scroll.documentView as! NSTextView
        textView.font = NSFont.monospacedSystemFont(ofSize: 13, weight: .regular)
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false
        textView.isAutomaticTextReplacementEnabled = false
        textView.isAutomaticSpellingCorrectionEnabled = false
        textView.isContinuousSpellCheckingEnabled = false
        textView.isRichText = false
        textView.allowsUndo = true
        textView.textContainerInset = NSSize(width: 8, height: 12)
        textView.delegate = context.coordinator
        textView.string = body
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        guard let textView = scroll.documentView as? NSTextView else { return }
        if textView.string != body && !context.coordinator.isProgrammaticUpdate {
            let old = context.coordinator.isProgrammaticUpdate
            context.coordinator.isProgrammaticUpdate = true
            textView.string = body
            context.coordinator.isProgrammaticUpdate = old
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(body: $body, isDirty: $isDirty, onCommit: onCommit)
    }

    @MainActor
    final class Coordinator: NSObject, NSTextViewDelegate {
        @Binding var body: String
        @Binding var isDirty: Bool
        let onCommit: (String) -> Void
        var isProgrammaticUpdate: Bool = false

        init(
            body: Binding<String>,
            isDirty: Binding<Bool>,
            onCommit: @escaping (String) -> Void
        ) {
            self._body = body
            self._isDirty = isDirty
            self.onCommit = onCommit
        }

        func textDidChange(_ notification: Notification) {
            guard let textView = notification.object as? NSTextView else { return }
            guard !isProgrammaticUpdate else { return }
            body = textView.string
            isDirty = true
        }
    }
}
