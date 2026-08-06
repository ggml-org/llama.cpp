import SwiftUI
import AppKit
import TesseraCore

// MARK: - NotesView

/// The macOS Notes surface. Bear-style Markdown focus mode.
///
/// **Layout.** A `NavigationSplitView` with three columns:
///   * Sidebar — the three list filters (All / Pinned /
///     Archived) + the tag chip strip + the "New Note" button.
///   * Middle — the note rows for the active filter.
///   * Detail — the note editor column (toolbar + tag bar +
///     TesseraEditorView + linked-entities section + focus
///     mode status bar).
///
/// **Editor.** The editor uses the same `TesseraEditorView`
/// (Phase 2) as the Documents surface, configured with
/// `EditorMode.notes` so the toolbar promotes callouts /
/// quotes and drops tables / code blocks.
///
/// **Focus mode.** Click into the editor and press the focus
/// toggle (Cmd-\) — the sidebar, toolbar, and status bar
/// fade; the note text fills the window; a subtle status bar
/// at the bottom shows the word count + reading time. Press
/// Escape (or click the unfocus button) to exit.
public struct NotesView: View {

    @ObservedObject public var viewModel: NotesViewModel

    public init(viewModel: NotesViewModel) {
        self.viewModel = viewModel
    }

    public var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 200, ideal: 240)
        } content: {
            notesListColumn
                .navigationSplitViewColumnWidth(min: 280, ideal: 320)
        } detail: {
            detailColumn
        }
        .navigationTitle("Notes")
        .toolbar { toolbarContent }
        .task {
            await viewModel.refresh()
            if viewModel.selectedNoteID == nil, let first = viewModel.rows.first {
                viewModel.select(first.id)
            }
        }
        .onChange(of: viewModel.filter) { _, _ in
            viewModel.applyFilter()
        }
        .onChange(of: viewModel.activeTag) { _, _ in
            viewModel.applyFilter()
        }
        .onChange(of: viewModel.selectedNoteID) { _, new in
            viewModel.select(new)
        }
        .onExitCommand {
            // Escape: exit focus mode first, then close
            // the editor (Apple's standard escape ladder).
            if viewModel.isFocusMode {
                viewModel.exitFocusMode()
            }
        }
    }

    // MARK: - Sidebar

    private var sidebar: some View {
        List(selection: filterSelection) {
            Section("Library") {
                ForEach(NoteListFilter.allCases) { f in
                    Label(f.displayName, systemImage: f.systemImage)
                        .tag(f)
                        .badge(rowCount(for: f))
                }
            }
            if !viewModel.allTags.isEmpty {
                Section("Tags") {
                    let chipBinding = Binding<String?>(
                        get: { viewModel.activeTag },
                        set: { viewModel.setActiveTag($0) }
                    )
                    TagChipsView(
                        tags: viewModel.allTags,
                        activeTag: chipBinding
                    )
                    .listRowInsets(EdgeInsets(top: 4, leading: 0, bottom: 4, trailing: 0))
                }
            }
        }
        .listStyle(.sidebar)
    }

    private var filterSelection: Binding<NoteListFilter?> {
        Binding<NoteListFilter?>(
            get: { viewModel.filter },
            set: { newValue in
                if let v = newValue { viewModel.filter = v }
            }
        )
    }

    private func rowCount(for filter: NoteListFilter) -> Int {
        // The counts come from the unfiltered list, so the
        // user always sees the canonical row count for each
        // filter (independent of the active tag chip).
        filter.apply(to: viewModel.allNotes).count
    }

    // MARK: - Notes list column

    private var notesListColumn: some View {
        Group {
            if viewModel.isLoading && viewModel.rows.isEmpty {
                ProgressView("Loading notes…")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let error = viewModel.loadError, viewModel.rows.isEmpty {
                errorState(error)
            } else if viewModel.rows.isEmpty {
                emptyState
            } else {
                notesList
            }
        }
        .searchable(text: searchTextBinding, prompt: "Search notes")
        .toolbar {
            ToolbarItem(placement: .automatic) {
                Button {
                    Task { await createBlankNote() }
                } label: {
                    Label("New Note", systemImage: "square.and.pencil")
                }
                .help("Create a new note (Cmd-N)")
                .keyboardShortcut("n", modifiers: .command)
            }
        }
    }

    private var notesList: some View {
        List(selection: selectionBinding) {
            ForEach(viewModel.rows) { row in
                NoteRowView(row: row)
                    .tag(row.id)
            }
        }
        .listStyle(.inset)
    }

    private var selectionBinding: Binding<UUID?> {
        Binding<UUID?>(
            get: { viewModel.selectedNoteID },
            set: { viewModel.select($0) }
        )
    }

    private var searchTextBinding: Binding<String> {
        // v1 keeps a local search; v2 will push the search
        // to the data layer's hybrid_search. The local
        // search filters the current list by title + body
        // text + tags.
        Binding<String>(
            get: { _localSearch },
            set: { newValue in
                _localSearch = newValue
                viewModel.applyLocalSearch(newValue)
            }
        )
    }

    @State private var _localSearch: String = ""

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "note.text")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)
            Text(emptyStateTitle)
                .font(.headline)
            Text(emptyStateSubtitle)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            Button {
                Task { await createBlankNote() }
            } label: {
                Label("New Note", systemImage: "square.and.pencil")
            }
            .controlSize(.large)
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var emptyStateTitle: String {
        switch viewModel.filter {
        case .all: return "No notes yet"
        case .pinned: return "Nothing pinned"
        case .archived: return "Nothing archived"
        }
    }

    private var emptyStateSubtitle: String {
        switch viewModel.filter {
        case .all:
            return "Create a note to get started. Press Cmd-N or click the button below."
        case .pinned:
            return "Pin a note from the editor to see it here."
        case .archived:
            return "Archive a note from the editor to see it here."
        }
    }

    private func errorState(_ message: String) -> some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 48))
                .foregroundStyle(.orange)
            Text("Couldn't load notes")
                .font(.headline)
            Text(message)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .font(.callout)
            Button("Retry") {
                Task { await viewModel.refresh() }
            }
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Detail column

    @ViewBuilder
    private var detailColumn: some View {
        if let editor = viewModel.editor {
            NoteEditorColumn(
                viewModel: editor,
                isFocusMode: $viewModel.isFocusMode,
                onDelete: { Task { await viewModel.deleteSelected() } }
            )
            .id(editor.note.id)  // Rebuild on note change
        } else {
            VStack(spacing: 8) {
                Image(systemName: "note.text")
                    .font(.system(size: 56))
                    .foregroundStyle(.secondary)
                Text("Select or create a note")
                    .font(.title3)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    // MARK: - Toolbar

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItem(placement: .primaryAction) {
            Button {
                viewModel.toggleFocusMode()
            } label: {
                Label(
                    viewModel.isFocusMode ? "Exit Focus" : "Focus",
                    systemImage: viewModel.isFocusMode ? "arrow.up.right.and.arrow.down.left.rectangle" : "arrow.down.left.and.arrow.up.right.rectangle"
                )
            }
            .help("Toggle focus mode (Cmd-\\)")
            .keyboardShortcut("\\", modifiers: .command)
        }
        ToolbarItem(placement: .automatic) {
            Button {
                Task { await viewModel.refresh() }
            } label: {
                Label("Refresh", systemImage: "arrow.clockwise")
            }
            .help("Reload notes")
        }
    }

    // MARK: - Actions

    private func createBlankNote() async {
        do {
            let note = try await viewModel.createNote(title: "Untitled", body: .empty, tags: [])
            viewModel.select(note.id)
        } catch {
            // The error is logged to the view model; a real
            // production wiring would surface a non-modal
            // banner. For v1, the editor column shows the
            // error string in its `lastError` state.
        }
    }
}

// MARK: - NoteRowView

/// One row in the notes list. Shows the title, snippet, tags
/// as pills, relative time, and a pin/archive icon.
struct NoteRowView: View {
    let row: NoteRow

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                if row.isPinned {
                    Image(systemName: "pin.fill")
                        .foregroundStyle(.orange)
                        .font(.caption)
                }
                if row.isArchived {
                    Image(systemName: "archivebox.fill")
                        .foregroundStyle(.secondary)
                        .font(.caption)
                }
                Text(row.title)
                    .font(.headline)
                    .lineLimit(1)
            }
            if !row.snippet.isEmpty {
                Text(row.snippet)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
            HStack(spacing: 6) {
                Text(row.relativeTime)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if !row.tags.isEmpty {
                    Text("·")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                    ForEach(row.tags.prefix(3), id: \.self) { tag in
                        TagPill(text: tag, isCompact: true)
                    }
                    if row.tags.count > 3 {
                        Text("+\(row.tags.count - 3)")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .padding(.vertical, 2)
    }
}

// MARK: - TagPill

/// A small rounded tag pill. The compact variant is used in
/// the list rows; the regular variant is used in the
/// editor's tag bar.
struct TagPill: View {
    let text: String
    var isCompact: Bool = false

    var body: some View {
        Text("#\(text)")
            .font(isCompact ? .caption2 : .caption)
            .padding(.horizontal, isCompact ? 6 : 8)
            .padding(.vertical, isCompact ? 2 : 3)
            .background(
                Capsule().fill(Color.accentColor.opacity(0.15))
            )
            .foregroundStyle(Color.accentColor)
    }
}

// MARK: - TagChipsView

/// A wrapping horizontal list of tag chips. Used in the
/// sidebar's "Tags" section. Selecting a chip toggles the
/// active tag filter.
struct TagChipsView: View {
    let tags: [String]
    @Binding var activeTag: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            FlowLayout(spacing: 4) {
                ForEach(tags, id: \.self) { tag in
                    Button {
                        activeTag = (activeTag == tag) ? nil : tag
                    } label: {
                        TagPill(text: tag, isCompact: true)
                            .opacity(activeTag == nil || activeTag == tag ? 1.0 : 0.4)
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }
}
