import Foundation
import SwiftUI

// MARK: - NotesViewModel

/// The view-model for the Notes surface (the macOS + iOS view
/// that drives the Bear-style Markdown notes). The model owns
/// the list of notes for the active filter, the selected note,
/// the focus-mode state, the active tag chip, and the chat-panel
/// integration.
///
/// The model is `@MainActor` so all `@Published` updates happen
/// on the main thread (SwiftUI's expectation). The data layer
/// facade is an actor; we hop to it for the read/write calls.
///
/// **Loading.** The view calls `refresh()` when it appears, when
/// the filter changes, when the active tag chip changes, and
/// after every mutation. The model reads the full note list
/// from the data layer, applies the filter, and re-projects
/// the rows. This is a v1 simplification — for 10k+ notes we'd
/// push the filter to the data layer — but the note table is
/// small in practice.
///
/// **Chat-panel integration.** The model exposes
/// `createNote(title:body:tags:)` and the tag / pin / archive
/// helpers, so the chat panel's command queue can call them
/// directly. Every chat-driven mutation produces the same
/// constitutional receipt as a user-driven one.
@MainActor
public final class NotesViewModel: ObservableObject {

    // MARK: - Published state

    @Published public private(set) var allNotes: [Note] = []
    @Published public private(set) var rows: [NoteRow] = []
    @Published public var filter: NoteListFilter = .all
    @Published public var selectedNoteID: UUID?
    @Published public var activeTag: String?
    @Published public var isLoading: Bool = false
    @Published public private(set) var loadError: String?
    @Published public var isFocusMode: Bool = false
    /// True while a chat-panel-driven command is in flight.
    /// The view shows a "working in background" chip in the
    /// chat panel header.
    @Published public private(set) var isChatDriven: Bool = false

    /// View-model for the note editor. nil when no note is
    /// selected. The view binds to it for the editor column.
    @Published public private(set) var editor: NoteEditorViewModel?

    // MARK: - Dependencies

    public let store: NoteStore
    public let dataLayer: TesseraDataLayer
    public let userID: UserID

    // MARK: - Init

    public init(
        store: NoteStore,
        dataLayer: TesseraDataLayer,
        userID: UserID = UUID()
    ) {
        self.store = store
        self.dataLayer = dataLayer
        self.userID = userID
    }

    // MARK: - Lifecycle

    /// Load every note + project the rows for the active filter.
    /// Called on view appear and after every mutation. The data
    /// layer call is async; the model is `@MainActor` so the
    /// `@Published` writes happen on the main thread.
    public func refresh() async {
        isLoading = true
        loadError = nil
        do {
            let notes = try await store.list(limit: 1000)
            allNotes = notes
            applyFilter()
        } catch {
            loadError = String(describing: error)
        }
        isLoading = false
    }

    /// Apply the active filter to `allNotes` and re-project the
    /// rows. The selected note (if any) is preserved across the
    /// refresh by id. Pure function over published state; called
    /// when `filter` or `activeTag` change.
    public func applyFilter() {
        let filtered = filter.apply(to: allNotes)
        let tagFiltered: [Note]
        if let activeTag {
            tagFiltered = filtered.filter { $0.tags.contains(activeTag) }
        } else {
            tagFiltered = filtered
        }
        rows = tagFiltered.map { NoteRow(note: $0) }
    }

    /// Replace the local `allNotes` cache without re-reading the
    /// data layer. Used by tests to inject fixture notes; the
    /// production path is `refresh()`.
    public func setAllNotesForTesting(_ notes: [Note]) {
        allNotes = notes
        applyFilter()
    }

    /// Apply a local search query. The local search filters the
    /// active list by title + body + tags (case-insensitive). The
    /// view's search field calls this on every keystroke. The
    /// search is in-memory (v1) — for 10k+ notes we'd push the
    /// search to the data layer's `hybrid_search`.
    public func applyLocalSearch(_ needle: String) {
        let trimmed = needle.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let base = filter.apply(to: allNotes)
        let tagFiltered: [Note]
        if let activeTag {
            tagFiltered = base.filter { $0.tags.contains(activeTag) }
        } else {
            tagFiltered = base
        }
        if trimmed.isEmpty {
            rows = tagFiltered.map { NoteRow(note: $0) }
        } else {
            rows = tagFiltered.compactMap { note in
                let hay = "\(note.title.lowercased()) \(note.tags.joined(separator: " ")) \(note.snippet(maxLength: 1000).lowercased())"
                return hay.contains(trimmed) ? NoteRow(note: note) : nil
            }
        }
    }

    /// The set of distinct tags across all notes (the sidebar's
    /// tag chip list). Sorted alphabetically.
    public var allTags: [String] {
        var seen: Set<String> = []
        for note in allNotes {
            for tag in note.tags { seen.insert(tag) }
        }
        return seen.sorted()
    }

    // MARK: - Selection

    /// Select a note by id. The editor's view-model is replaced
    /// with one for the new note. Selecting nil tears down the
    /// editor.
    public func select(_ noteID: UUID?) {
        selectedNoteID = noteID
        if let noteID, let note = allNotes.first(where: { $0.id == noteID }) {
            editor = NoteEditorViewModel(note: note, store: store, userID: userID)
        } else {
            editor = nil
        }
    }

    /// Create a new note with a title + optional body + tags.
    /// Used by the "New Note" toolbar action and the chat panel's
    /// "create a note titled X" command. The new note is added
    /// to the local cache AND persisted via the store (which
    /// emits a `note_upsert` receipt). Returns the created note.
    @discardableResult
    public func createNote(
        title: String,
        body: DocumentAST = .empty,
        tags: [String] = []
    ) async throws -> Note {
        let now = Date()
        let note = Note(
            id: UUID(),
            title: title,
            body: body,
            tags: Note.normalizeTags(tags),
            createdAt: now,
            updatedAt: now
        )
        _ = try await store.upsert(note)
        await refresh()
        return note
    }

    /// Delete the currently-selected note. The selection is
    /// cleared after the delete so the editor column shows the
    /// empty state.
    public func deleteSelected() async {
        guard let id = selectedNoteID else { return }
        do {
            _ = try await store.delete(id: id)
            selectedNoteID = nil
            editor = nil
            await refresh()
        } catch {
            loadError = String(describing: error)
        }
    }

    // MARK: - Chat-panel integration

    /// Create a note from a chat-panel command. The method is
    /// flagged as chat-driven so the view can show the
    /// "working in background" chip; the chip clears when the
    /// create completes. Always emits the constitutional
    /// `note_upsert` receipt via the store.
    @discardableResult
    public func chatCreateNote(
        title: String,
        body: DocumentAST = .empty,
        tags: [String] = []
    ) async throws -> Note {
        isChatDriven = true
        defer { isChatDriven = false }
        return try await createNote(title: title, body: body, tags: tags)
    }

    /// Edit a note from a chat-panel command (e.g. "add a tag
    /// 'q3-2026' to this note"). The store handles the mutation
    /// + receipt.
    @discardableResult
    public func chatEditNote(
        noteID: UUID,
        apply: (NoteStore) async throws -> Note
    ) async throws -> Note {
        isChatDriven = true
        defer { isChatDriven = false }
        let updated = try await apply(store)
        await refresh()
        if selectedNoteID == noteID, let editor {
            editor.refresh(with: updated)
        }
        return updated
    }

    // MARK: - Focus mode

    /// Toggle focus mode. The view fades the chrome (toolbar,
    /// sidebar, status bar) when focus mode is on and shows a
    /// minimal status bar at the bottom with the word count
    /// and reading time. Escape exits focus mode — the host
    /// view wires the key event to this method.
    public func toggleFocusMode() {
        withAnimation(.easeInOut(duration: 0.25)) {
            isFocusMode.toggle()
        }
    }

    /// Exit focus mode. Called from the Escape key handler.
    public func exitFocusMode() {
        guard isFocusMode else { return }
        withAnimation(.easeInOut(duration: 0.25)) {
            isFocusMode = false
        }
    }

    // MARK: - Tag chip

    /// Set the active tag chip. The filter is re-applied so
    /// the rows show only notes with the given tag. Setting
    /// `nil` clears the chip.
    public func setActiveTag(_ tag: String?) {
        activeTag = tag
        applyFilter()
    }
}

// MARK: - NoteEditorViewModel

/// The view-model for the note editor column. Owns the
/// in-memory copy of the note being edited and routes user
/// edits through the ``NoteStore`` (which appends the
/// constitutional receipt).
///
/// **Editor binding.** The view binds to `document` (the
/// in-memory `DocumentAST`) and renders the same
/// `TesseraEditorView` as the Documents surface, configured
/// with `EditorMode.notes`. The platform text view posts its
/// changes through the `TesseraEditorView`'s coalescer, which
/// fires an `onMutationCommitted` callback; that callback
/// updates the binding, which we then push to the store via
/// `commitBody`.
///
/// **Title / pin / archive / tags / links.** These are
/// note-level concerns (not block-level), so they bypass the
/// editor and call the store directly. The view shows the
/// note's title, tag input, pin/archive toggles, and linked
/// entities around the editor.
@MainActor
public final class NoteEditorViewModel: ObservableObject {

    @Published public private(set) var note: Note
    @Published public private(set) var document: DocumentAST
    @Published public var draftTitle: String
    @Published public var draftTag: String
    @Published public var isSaving: Bool = false
    @Published public private(set) var lastError: String?

    public let store: NoteStore
    public let userID: UserID

    public init(note: Note, store: NoteStore, userID: UserID) {
        self.note = note
        self.document = note.body
        self.draftTitle = note.title
        self.draftTag = ""
        self.store = store
        self.userID = userID
    }

    /// Replace the local copy with a fresh `Note` (e.g. after
    /// a store-level refresh). The drafts reset to the new
    /// note's values.
    public func refresh(with note: Note) {
        self.note = note
        self.document = note.body
        self.draftTitle = note.title
    }

    // MARK: - Title

    /// Persist the title. Called from the title field's
    /// `.onSubmit` / debounced change. The store emits a
    /// `note_title_changed` receipt.
    public func commitTitle() async {
        let trimmed = draftTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed != note.title else { return }
        isSaving = true
        defer { isSaving = false }
        do {
            let updated = try await store.setTitle(trimmed, for: note.id, oldNote: note)
            self.note = updated
        } catch {
            lastError = String(describing: error)
        }
    }

    // MARK: - Body

    /// Persist the body AST. Called by the editor's
    /// `onMutationCommitted` callback after a coalesced edit
    /// burst. The store emits a `note_body_changed` receipt.
    public func commitBody(_ ast: DocumentAST) async {
        guard ast != note.body else { return }
        isSaving = true
        defer { isSaving = false }
        do {
            let updated = try await store.setBody(ast, for: note.id)
            self.note = updated
        } catch {
            lastError = String(describing: error)
        }
    }

    /// Update the local document without persisting (the
    /// editor binding uses this so the typing-to-render path
    /// is instant). The persist happens on coalesce-flush via
    /// `commitBody`.
    public func setDocumentLocal(_ ast: DocumentAST) {
        self.document = ast
    }

    // MARK: - Pin / archive

    public func togglePinned() async {
        isSaving = true
        defer { isSaving = false }
        do {
            let updated = note.isPinned
                ? try await store.unpin(note.id)
                : try await store.pin(note.id)
            self.note = updated
        } catch {
            lastError = String(describing: error)
        }
    }

    public func toggleArchived() async {
        isSaving = true
        defer { isSaving = false }
        do {
            let updated = note.isArchived
                ? try await store.unarchive(note.id)
                : try await store.archive(note.id)
            self.note = updated
        } catch {
            lastError = String(describing: error)
        }
    }

    // MARK: - Tags

    /// Add the current draft tag. The draft is cleared on
    /// success.
    public func addDraftTag() async {
        let normalized = Note.normalizeTags([draftTag])
        guard let first = normalized.first else { return }
        guard !note.tags.contains(first) else {
            draftTag = ""
            return
        }
        isSaving = true
        defer { isSaving = false }
        do {
            let updated = try await store.addTag(first, to: note.id)
            self.note = updated
            self.draftTag = ""
        } catch {
            lastError = String(describing: error)
        }
    }

    /// Remove an existing tag.
    public func removeTag(_ tag: String) async {
        isSaving = true
        defer { isSaving = false }
        do {
            let updated = try await store.removeTag(tag, from: note.id)
            self.note = updated
        } catch {
            lastError = String(describing: error)
        }
    }

    // MARK: - Linking

    /// Link this note to another graph entity.
    public func link(to otherEntityID: UUID, linkType: String = "related_to") async {
        isSaving = true
        defer { isSaving = false }
        do {
            let updated = try await store.link(noteID: note.id, to: otherEntityID, linkType: linkType)
            // The link call doesn't return the updated note; we
            // re-read it to refresh the local copy.
            if let fresh = try await store.get(id: note.id) {
                self.note = fresh
            } else {
                _ = updated
            }
        } catch {
            lastError = String(describing: error)
        }
    }
}
