#if os(iOS)
import SwiftUI
import UIKit
import TesseraCore

// MARK: - NotesView_iOS

/// The iOS Notes surface. Bear-style Markdown focus mode,
/// adapted for the touch / phone form factor.
///
/// **Layout.** A `NavigationStack` with a `TabView` at the
/// top for the three list filters (All / Pinned /
/// Archived) and the list below. Tapping a row pushes the
/// note editor (`NoteEditorView_iOS`) onto the stack.
/// The chat panel is a tab in the bottom bar (per the
/// spec §6.1).
///
/// **Editor.** Same `TesseraEditorView` as the macOS view,
/// wrapped in an iOS `UIViewRepresentable` (the Phase 2
/// editor's `TesseraEditorView` already has the iOS path
/// — Phase 5 just reuses it).
public struct NotesView_iOS: View {

    @ObservedObject public var viewModel: NotesViewModel
    @State private var searchText: String = ""
    @State private var newNotePending: Bool = false

    public init(viewModel: NotesViewModel) {
        self.viewModel = viewModel
    }

    public var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                filterTabs
                if viewModel.isLoading && viewModel.rows.isEmpty {
                    ProgressView()
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if viewModel.rows.isEmpty {
                    emptyState
                } else {
                    notesList
                }
            }
            .navigationTitle("Notes")
            .navigationBarTitleDisplayMode(.inline)
            .searchable(text: $searchText)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        Task { await createBlankNote() }
                    } label: {
                        Image(systemName: "square.and.pencil")
                    }
                }
            }
            .navigationDestination(for: UUID.self) { id in
                if let editor = viewModel.editor, editor.note.id == id {
                    NoteEditorView_iOS(viewModel: editor)
                } else if let note = viewModel.allNotes.first(where: { $0.id == id }) {
                    let editor = NoteEditorViewModel(note: note, store: viewModel.store, userID: viewModel.userID)
                    NoteEditorView_iOS(viewModel: editor)
                } else {
                    Text("Note not found")
                }
            }
            .task {
                await viewModel.refresh()
            }
        }
    }

    // MARK: - Filter tabs

    private var filterTabs: some View {
        Picker("Filter", selection: $viewModel.filter) {
            ForEach(NoteListFilter.allCases) { f in
                Text(f.displayName).tag(f)
            }
        }
        .pickerStyle(.segmented)
        .padding(.horizontal)
        .padding(.vertical, 8)
        .onChange(of: viewModel.filter) { _, _ in
            viewModel.applyFilter()
        }
    }

    // MARK: - List

    private var notesList: some View {
        List {
            ForEach(viewModel.rows) { row in
                NavigationLink(value: row.id) {
                    NoteRowView_iOS(row: row)
                }
            }
        }
        .listStyle(.plain)
    }

    // MARK: - Empty state

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "note.text")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)
            Text(emptyStateTitle)
                .font(.headline)
            Button {
                Task { await createBlankNote() }
            } label: {
                Label("New Note", systemImage: "square.and.pencil")
            }
            .controlSize(.large)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var emptyStateTitle: String {
        switch viewModel.filter {
        case .all: return "No notes yet"
        case .pinned: return "Nothing pinned"
        case .archived: return "Nothing archived"
        }
    }

    // MARK: - Actions

    private func createBlankNote() async {
        do {
            let note = try await viewModel.createNote(title: "Untitled", body: .empty, tags: [])
            viewModel.select(note.id)
        } catch {
            // Error logged to the view model.
        }
    }
}

// MARK: - NoteRowView_iOS

/// The iOS row variant. Same content as the macOS row,
/// with iOS-appropriate spacing.
struct NoteRowView_iOS: View {
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
            HStack(spacing: 4) {
                Text(row.relativeTime)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if !row.tags.isEmpty {
                    Text("·")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                    Text(row.tags.prefix(3).map { "#\($0)" }.joined(separator: " "))
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                }
            }
        }
        .padding(.vertical, 2)
    }
}

// MARK: - NoteEditorView_iOS

/// The iOS note editor. Same data flow as the macOS
/// column, but presented in a `NavigationStack` push rather
/// than as a split-view detail column. The toolbar has the
/// pin / archive / delete actions. The editor uses
/// `TesseraEditorView` configured for `EditorMode.notes`.
struct NoteEditorView_iOS: View {

    @ObservedObject public var viewModel: NoteEditorViewModel
    @State private var showDeleteConfirm: Bool = false
    @State private var isFocusMode: Bool = false

    public init(viewModel: NoteEditorViewModel) {
        self.viewModel = viewModel
    }

    var body: some View {
        VStack(spacing: 0) {
            if !isFocusMode {
                titleBar
            }
            TesseraEditorView(
                mode: .notes,
                theme: .light,
                document: documentBinding,
                onMutationCommitted: { _, _ in
                    let ast = viewModel.document
                    Task { await viewModel.commitBody(ast) }
                }
            )
            if isFocusMode {
                focusStatusBar
            }
        }
        .navigationTitle(isFocusMode ? "" : viewModel.note.displayTitle)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    isFocusMode.toggle()
                } label: {
                    Image(systemName: isFocusMode
                          ? "arrow.up.right.and.arrow.down.left.rectangle"
                          : "arrow.down.left.and.arrow.up.right.rectangle")
                }
            }
            ToolbarItem(placement: .topBarTrailing) {
                Menu {
                    Button {
                        Task { await viewModel.togglePinned() }
                    } label: {
                        Label(
                            viewModel.note.isPinned ? "Unpin" : "Pin",
                            systemImage: viewModel.note.isPinned ? "pin.slash" : "pin"
                        )
                    }
                    Button {
                        Task { await viewModel.toggleArchived() }
                    } label: {
                        Label(
                            viewModel.note.isArchived ? "Unarchive" : "Archive",
                            systemImage: viewModel.note.isArchived ? "archivebox.fill" : "archivebox"
                        )
                    }
                    Divider()
                    Button(role: .destructive) {
                        showDeleteConfirm = true
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
            }
        }
        .confirmationDialog(
            "Delete this note?",
            isPresented: $showDeleteConfirm,
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) {
                showDeleteConfirm = false
                // The view model is on the editor; we don't
                // have direct access to the parent
                // NotesViewModel here, so the deletion just
                // marks the note via the store. The
                // NotesViewModel's `deleteSelected` is the
                // path that also clears the selection.
                Task {
                    _ = try? await viewModel.store.delete(id: viewModel.note.id)
                }
            }
            Button("Cancel", role: .cancel) { showDeleteConfirm = false }
        }
    }

    private var titleBar: some View {
        VStack(alignment: .leading, spacing: 6) {
            TextField("Title", text: $viewModel.draftTitle, onCommit: {
                Task { await viewModel.commitTitle() }
            })
            .textFieldStyle(.plain)
            .font(.title2)
            .fontWeight(.bold)

            if !viewModel.note.tags.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 4) {
                        ForEach(viewModel.note.tags, id: \.self) { tag in
                            TagPill_iOS(text: tag)
                        }
                    }
                }
            }
        }
        .padding()
    }

    private var focusStatusBar: some View {
        HStack {
            Text("\(viewModel.note.wordCount) words · \(viewModel.note.readingTimeMinutes) min read")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            Button("Exit Focus") { isFocusMode = false }
                .font(.caption)
        }
        .padding(.horizontal)
        .padding(.vertical, 6)
        .background(.bar)
    }

    private var documentBinding: Binding<DocumentAST> {
        Binding<DocumentAST>(
            get: { viewModel.document },
            set: { newValue in viewModel.setDocumentLocal(newValue) }
        )
    }
}

// MARK: - TagPill_iOS

/// iOS tag pill. Mirrors the macOS ``TagPill``.
struct TagPill_iOS: View {
    let text: String

    var body: some View {
        Text("#\(text)")
            .font(.caption)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(
                Capsule().fill(Color.accentColor.opacity(0.15))
            )
            .foregroundStyle(Color.accentColor)
    }
}

#endif
