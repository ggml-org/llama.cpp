import SwiftUI
import AppKit
import TesseraCore

// MARK: - NoteEditorColumn

/// The note editor column. The top section shows the note's
/// title (editable) + tag bar + pin/archive toggles. The
/// middle section is the `TesseraEditorView` configured
/// for `EditorMode.notes`. The bottom section is the
/// linked-entities list.
///
/// **Focus mode.** When `isFocusMode` is on, the title bar +
/// tag bar + linked-entities section all fade; only the
/// editor + a minimal status bar (word count + reading
/// time) remain. Escape exits focus mode.
public struct NoteEditorColumn: View {

    @ObservedObject public var viewModel: NoteEditorViewModel
    @Binding public var isFocusMode: Bool
    public let onDelete: () -> Void

    @State private var formattingState: FormattingState = FormattingState()
    @State private var showDeleteConfirm: Bool = false
    @State private var showLinkSearch: Bool = false
    @State private var linkSearchQuery: String = ""

    public init(
        viewModel: NoteEditorViewModel,
        isFocusMode: Binding<Bool>,
        onDelete: @escaping () -> Void
    ) {
        self.viewModel = viewModel
        self._isFocusMode = isFocusMode
        self.onDelete = onDelete
    }

    public var body: some View {
        VStack(spacing: 0) {
            // The title + tag bar + toggles fade in focus
            // mode; the editor + status bar stay.
            if !isFocusMode {
                topBar
                    .transition(.opacity)
            }
            Divider()
            editor
            Divider()
            if !isFocusMode {
                bottomBar
                    .transition(.opacity)
            }
            if isFocusMode {
                focusStatusBar
                    .transition(.opacity)
            }
        }
        .background(Color(nsColor: .textBackgroundColor))
        .toolbar { editorToolbar }
        .sheet(isPresented: $showDeleteConfirm) {
            deleteConfirmationSheet
        }
        .sheet(isPresented: $showLinkSearch) {
            linkSearchSheet
        }
    }

    // MARK: - Top bar (title + tag bar + pin/archive)

    private var topBar: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Title field
            TextField("Title", text: $viewModel.draftTitle, onCommit: {
                Task { await viewModel.commitTitle() }
            })
            .textFieldStyle(.plain)
            .font(.title)
            .fontWeight(.bold)

            // Tag bar
            tagBar

            // Pin / archive / link toggles
            HStack(spacing: 12) {
                Toggle(isOn: pinToggleBinding) {
                    Label(
                        viewModel.note.isPinned ? "Pinned" : "Pin",
                        systemImage: viewModel.note.isPinned ? "pin.fill" : "pin"
                    )
                }
                .toggleStyle(.button)
                .controlSize(.small)

                Toggle(isOn: archiveToggleBinding) {
                    Label(
                        viewModel.note.isArchived ? "Archived" : "Archive",
                        systemImage: viewModel.note.isArchived ? "archivebox.fill" : "archivebox"
                    )
                }
                .toggleStyle(.button)
                .controlSize(.small)

                Button {
                    showLinkSearch = true
                } label: {
                    Label("Link…", systemImage: "link.badge.plus")
                }
                .controlSize(.small)

                Spacer()

                if viewModel.isSaving {
                    ProgressView()
                        .controlSize(.small)
                }
                if let err = viewModel.lastError {
                    Text(err)
                        .foregroundStyle(.red)
                        .font(.caption)
                        .lineLimit(1)
                }
            }
            .font(.callout)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(.bar)
    }

    private var tagBar: some View {
        HStack(spacing: 6) {
            FlowLayout(spacing: 4) {
                ForEach(viewModel.note.tags, id: \.self) { tag in
                    Button {
                        Task { await viewModel.removeTag(tag) }
                    } label: {
                        HStack(spacing: 4) {
                            Text("#\(tag)")
                            Image(systemName: "xmark")
                                .font(.caption2)
                        }
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(
                            Capsule().fill(Color.accentColor.opacity(0.15))
                        )
                        .foregroundStyle(Color.accentColor)
                    }
                    .buttonStyle(.plain)
                }
            }
            TextField("Add tag…", text: $viewModel.draftTag, onCommit: {
                Task { await viewModel.addDraftTag() }
            })
            .textFieldStyle(.roundedBorder)
            .controlSize(.small)
            .frame(maxWidth: 160)
        }
    }

    // MARK: - Editor (TesseraEditorView in notes mode)

    private var editor: some View {
        TesseraEditorView(
            mode: .notes,
            theme: .light,
            document: documentBinding,
            onMutationCommitted: { _, _ in
                // The editor's coalescer already updates the
                // document binding. We just need to commit
                // the body to the store on flush. The local
                // document is updated through the binding;
                // the persist is a debounced commit.
                let ast = viewModel.document
                Task { await viewModel.commitBody(ast) }
            }
        )
    }

    private var documentBinding: Binding<DocumentAST> {
        Binding<DocumentAST>(
            get: { viewModel.document },
            set: { newValue in
                viewModel.setDocumentLocal(newValue)
            }
        )
    }

    // MARK: - Bottom bar (linked entities)

    private var bottomBar: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Linked entities")
                    .font(.headline)
                Spacer()
                if !viewModel.note.linkedEntityIDs.isEmpty {
                    Text("\(viewModel.note.linkedEntityIDs.count) link\(viewModel.note.linkedEntityIDs.count == 1 ? "" : "s")")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            if viewModel.note.linkedEntityIDs.isEmpty {
                Text("No linked entities yet. Use the Link… button or chat panel.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                FlowLayout(spacing: 6) {
                    ForEach(viewModel.note.linkedEntityIDs, id: \.self) { id in
                        LinkedEntityChip(
                            id: id,
                            onClick: { /* open in graph view (Phase 6) */ }
                        )
                    }
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.bar)
    }

    // MARK: - Focus mode status bar

    private var focusStatusBar: some View {
        HStack {
            Text("\(viewModel.note.wordCount) words · \(viewModel.note.readingTimeMinutes) min read")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            Button {
                isFocusMode = false
            } label: {
                Label("Exit Focus", systemImage: "arrow.up.right.and.arrow.down.left.rectangle")
                    .font(.caption)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 6)
        .background(.bar)
    }

    // MARK: - Toolbar (editor formatting + focus toggle + delete)

    @ToolbarContentBuilder
    private var editorToolbar: some ToolbarContent {
        ToolbarItem(placement: .automatic) {
            TesseraEditorToolbar(
                mode: .notes,
                formattingState: $formattingState,
                onCommand: { command in
                    handleEditorCommand(command)
                }
            )
        }
        ToolbarItem(placement: .primaryAction) {
            Button {
                withAnimation(.easeInOut(duration: 0.25)) {
                    isFocusMode.toggle()
                }
            } label: {
                Label(
                    isFocusMode ? "Exit Focus" : "Focus",
                    systemImage: isFocusMode
                        ? "arrow.up.right.and.arrow.down.left.rectangle"
                        : "arrow.down.left.and.arrow.up.right.rectangle"
                )
            }
            .help("Toggle focus mode (Cmd-\\)")
        }
        ToolbarItem(placement: .destructiveAction) {
            Button(role: .destructive) {
                showDeleteConfirm = true
            } label: {
                Label("Delete", systemImage: "trash")
            }
            .help("Delete this note")
        }
    }

    // MARK: - Delete confirm sheet

    private var deleteConfirmationSheet: some View {
        VStack(spacing: 16) {
            Text("Delete this note?")
                .font(.headline)
            Text("\"\(viewModel.note.displayTitle)\"")
                .foregroundStyle(.secondary)
            Text("The note is removed from your library. The receipt chain is preserved for the audit trail.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .font(.callout)
            HStack {
                Button("Cancel") { showDeleteConfirm = false }
                    .keyboardShortcut(.defaultAction)
                Button("Delete", role: .destructive) {
                    showDeleteConfirm = false
                    onDelete()
                }
                .keyboardShortcut(.cancelAction)
            }
        }
        .padding()
        .frame(width: 360)
    }

    // MARK: - Link search sheet (search-and-link to other materials)

    private var linkSearchSheet: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Link to another material")
                .font(.headline)
            TextField("Search documents, contacts, events, tasks, notes…",
                      text: $linkSearchQuery)
                .textFieldStyle(.roundedBorder)
            Text("v1: paste an entity UUID. v2 wires the search to the data layer's hybrid_search.")
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack {
                Spacer()
                Button("Cancel") { showLinkSearch = false }
                Button("Link") {
                    if let target = UUID(uuidString: linkSearchQuery) {
                        Task {
                            await viewModel.link(to: target)
                            showLinkSearch = false
                            linkSearchQuery = ""
                        }
                    }
                }
                .disabled(UUID(uuidString: linkSearchQuery) == nil)
            }
        }
        .padding()
        .frame(width: 460)
    }

    // MARK: - Bindings for toggles

    private var pinToggleBinding: Binding<Bool> {
        Binding<Bool>(
            get: { viewModel.note.isPinned },
            set: { _ in
                Task { await viewModel.togglePinned() }
            }
        )
    }

    private var archiveToggleBinding: Binding<Bool> {
        Binding<Bool>(
            get: { viewModel.note.isArchived },
            set: { _ in
                Task { await viewModel.toggleArchived() }
            }
        )
    }

    // MARK: - Editor command handling

    private func handleEditorCommand(_ command: EditorCommand) {
        // The editor view's coalescer applies the command
        // to the document binding. We just trigger a body
        // commit so the new state persists.
        let ast = viewModel.document
        Task { await viewModel.commitBody(ast) }
    }
}

// MARK: - LinkedEntityChip

/// A small chip showing a linked entity's UUID in a
/// compact form. v1 is a placeholder — v2 will resolve
/// the UUID to the entity's display label via the data
/// layer.
struct LinkedEntityChip: View {
    let id: UUID
    let onClick: () -> Void

    var body: some View {
        Button(action: onClick) {
            HStack(spacing: 4) {
                Image(systemName: "link")
                Text(id.uuidString.prefix(8) + "…")
            }
            .font(.caption)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(
                Capsule().fill(Color.gray.opacity(0.15))
            )
            .foregroundStyle(.secondary)
        }
        .buttonStyle(.plain)
    }
}
