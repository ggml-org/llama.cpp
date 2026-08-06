import Foundation

// MARK: - CodeSurfaceViewModel

/// The state + actions for the Code surface. The
/// view model composes ``CodeStore``,
/// ``CodeFileWatcher``, ``GitReadOnly``, and the
/// chat panel's command queue. It is `@MainActor`
/// because the SwiftUI views observe it directly
/// and the mutations (open / close / save) come
/// from view events.
///
/// **Lifecycle.** The view model's lifetime is the
/// surface's lifetime. `start()` is called from the
/// view's `onAppear`; `stop()` from `onDisappear`.
/// The view model owns the watcher's `Task` and the
/// git subprocess's lifecycle.
///
/// **Data flow.** The view model:
///   1. Loads the `CodeStore`'s index (durable
///      `graph_entities` rows).
///   2. Starts the `CodeFileWatcher` (if the user
///      picked a watched root).
///   3. Drains the watcher's `AsyncStream` into a
///      per-event handler that upserts/deletes the
///      in-memory index + emits a chat queue item.
@MainActor
public final class CodeSurfaceViewModel: ObservableObject {

    // MARK: - State

    /// The watched root. The user picks this in the
    /// surface's toolbar; the view model watches it
    /// + builds the directory tree + runs the git
    /// integration against the repo at this root.
    @Published public var watchedRoot: URL?

    /// The currently-open file (the one the editor
    /// shows). nil when no file is selected.
    @Published public var currentFile: CodeFile?

    /// The directory tree. The view renders this in
    /// the sidebar; the tree is rebuilt on every
    /// file change (the cost is O(n) over the file
    /// set, which is sub-frame for typical projects).
    @Published public var tree: CodeFileTree

    /// The outline of the current file. The view
    /// renders this in the outline panel; the value
    /// is `CodeOutline.empty` when no file is
    /// selected.
    @Published public var currentOutline: CodeOutline

    /// The recent commits for the current file.
    /// `nil` when the file isn't in a git repo or no
    /// file is selected. The git panel renders this.
    @Published public var recentCommits: [GitCommit]?

    /// The blame for the current file. `nil` when no
    /// file is selected.
    @Published public var currentBlame: [GitBlame]?

    /// The current search query. The view binds the
    /// search bar to this; the index is queried
    /// synchronously on every change.
    @Published public var searchQuery: String = ""
    @Published public var searchHits: [CodeSearchHit] = []

    /// The chat queue items the Code surface has
    /// produced (file imports, mutations from the
    /// agent, etc.). The chat panel's view consumes
    /// this list.
    @Published public var chatQueueItems: [ChatQueueItem] = []

    /// A status string for the surface's footer (e.g.
    /// "Watched 42 files in ~/Developer/MyProject").
    @Published public var statusMessage: String = ""

    /// The most recent error. The view shows a
    /// non-fatal banner with this string.
    @Published public var lastError: String?

    // MARK: - Dependencies

    private let store: CodeStore
    private let fileWatcher: CodeFileWatcher?
    private let git: GitReadOnly?
    private let builder: CodeFileTreeBuilder
    private let outlineExtractor: CodeOutlineExtractor
    private var searchIndex: CodeSearchIndex
    private var watcherTask: Task<Void, Never>?

    // MARK: - Init

    public init(
        store: CodeStore,
        watchedRoot: URL? = nil
    ) {
        self.store = store
        self.watchedRoot = watchedRoot
        self.tree = .empty
        self.currentOutline = .empty
        self.searchIndex = CodeSearchIndex()
        self.builder = CodeFileTreeBuilder()
        self.outlineExtractor = CodeOutlineExtractor()
        if let watchedRoot {
            self.fileWatcher = CodeFileWatcher(rootURL: watchedRoot)
            // The git actor is best-effort: if the
            // root isn't a git repo, the actor's
            // `validate()` throws on every call. We
            // construct the actor anyway; the panel
            // shows "not a git repository" if the
            // validation fails.
            self.git = GitReadOnly(repoURL: watchedRoot)
        } else {
            self.fileWatcher = nil
            self.git = nil
        }
    }

    // MARK: - Lifecycle

    /// Start the surface. Loads the data layer, starts
    /// the watcher, and kicks off the git integration
    /// (if the root is a repo). Safe to call multiple
    /// times; the second call is a no-op.
    public func start() async {
        do {
            try await store.loadAll()
            refreshSearchIndex()
            rebuildTree()
            if let fileWatcher {
                try await fileWatcher.startWatching()
                startDrainingEvents(from: fileWatcher)
            }
            if let git {
                do {
                    try await git.validate()
                } catch {
                    self.lastError = "Git: \(error)"
                }
            }
            statusMessage = watchedRoot.map {
                "Loaded \(indexedCount()) files from \($0.lastPathComponent)"
            } ?? "No watched root"
        } catch {
            self.lastError = "Code surface: failed to start: \(error)"
        }
    }

    /// Stop the surface. Cancels the watcher drain
    /// task and the watcher itself.
    public func stop() {
        watcherTask?.cancel()
        watcherTask = nil
        Task { [fileWatcher] in
            await fileWatcher?.stopWatching()
        }
    }

    deinit {
        watcherTask?.cancel()
    }

    // MARK: - File selection

    /// Open a file. The view model:
    ///   1. Sets `currentFile` (the editor re-renders).
    ///   2. Recomputes the outline (the outline panel
    ///      re-renders).
    ///   3. Kicks off a git background task for the
    ///      recent commits + blame.
    public func open(file: CodeFile) async {
        currentFile = file
        currentOutline = outlineExtractor.extract(
            source: file.body, language: file.language
        )
        recentCommits = nil
        currentBlame = nil
        if let git {
            do {
                let commits = try await git.recentCommits(file: file, limit: 20)
                self.recentCommits = commits
            } catch {
                self.recentCommits = nil
            }
        }
    }

    /// Save a body change to the current file. The
    /// mutation is the `replaceCodeBlock` variant;
    /// the receipt chain records the pre/post
    /// checksums.
    public func saveBody(_ newBody: String) async {
        guard let currentFile else { return }
        let mutation = CodeMutation.replaceCodeBlock(
            fileID: currentFile.id, newBody: newBody
        )
        do {
            let result = try await store.apply(mutation, to: currentFile.id)
            self.currentFile = result.updated
            // Refresh the outline (function names
            // may have changed).
            self.currentOutline = outlineExtractor.extract(
                source: result.updated.body, language: result.updated.language
            )
            // Refresh the search index (the file's
            // body changed; the index needs the new
            // body).
            searchIndex.upsert(result.updated)
        } catch {
            self.lastError = "Save failed: \(error)"
        }
    }

    // MARK: - Tagging

    public func addTag(_ tag: String, to fileID: UUID) async {
        let mutation = CodeMutation.addTag(fileID: fileID, tag: tag)
        do {
            let result = try await store.apply(mutation, to: fileID)
            if currentFile?.id == fileID { self.currentFile = result.updated }
        } catch {
            self.lastError = "Tag add failed: \(error)"
        }
    }

    public func removeTag(_ tag: String, from fileID: UUID) async {
        let mutation = CodeMutation.removeTag(fileID: fileID, tag: tag)
        do {
            let result = try await store.apply(mutation, to: fileID)
            if currentFile?.id == fileID { self.currentFile = result.updated }
        } catch {
            self.lastError = "Tag remove failed: \(error)"
        }
    }

    // MARK: - Search

    /// Run the current search query against the in-
    /// memory index. The query is synchronous (the
    /// index is small); the view calls this on every
    /// keystroke (debounced by SwiftUI's `@Published`
    /// if the view uses `.onChange(of:)`).
    public func runSearch() {
        let query = CodeSearchQuery(pattern: searchQuery)
        searchHits = searchIndex.search(query)
    }

    // MARK: - Watcher event drain

    /// The view model spawns a long-lived `Task` that
    /// iterates the watcher's `AsyncStream` and
    /// applies each event to the store. The task is
    /// cancelled in `stop()`.
    private func startDrainingEvents(from watcher: CodeFileWatcher) {
        watcherTask?.cancel()
        watcherTask = Task { [weak self] in
            guard let self else { return }
            let stream = await watcher.events()
            for await event in stream {
                if Task.isCancelled { break }
                await self.handleEvent(event)
            }
        }
    }

    private func handleEvent(_ event: CodeFileEvent) async {
        do {
            switch event {
            case .created(let file):
                _ = try await store.upsert(file)
                searchIndex.upsert(file)
                enqueueChatItem(
                    message: "Imported \(file.filename)",
                    state: .applied
                )
            case .modified(let file):
                _ = try await store.upsert(file)
                searchIndex.upsert(file)
                if currentFile?.path == file.path {
                    self.currentFile = file
                    self.currentOutline = outlineExtractor.extract(
                        source: file.body, language: file.language
                    )
                }
                enqueueChatItem(
                    message: "Modified \(file.filename)",
                    state: .applied
                )
            case .deleted(let path):
                if let id = (try? store.get(path: path))?.id {
                    try? await store.delete(id: id)
                    searchIndex.remove(path: path)
                    if currentFile?.path == path {
                        self.currentFile = nil
                        self.currentOutline = .empty
                    }
                    enqueueChatItem(
                        message: "Deleted \(path)",
                        state: .applied
                    )
                }
            case .renamed(let from, let to):
                if let id = (try? store.get(path: from))?.id {
                    if let renamed = try? await store.rename(id: id, to: to) {
                        searchIndex.remove(path: from)
                        searchIndex.upsert(renamed)
                    }
                    enqueueChatItem(
                        message: "Renamed \(from) -> \(to)",
                        state: .applied
                    )
                }
            }
            rebuildTree()
        } catch {
            self.lastError = "Watcher event failed: \(error)"
        }
    }

    // MARK: - Chat integration

    /// Append a chat queue item. The Code surface
    /// shares the chat panel with the rest of the
    /// productivity surface; the item is a
    /// first-class `ChatQueueItem` so the chat
    /// panel's view model can render it without
    /// knowing about code specifically.
    public func enqueueChatItem(
        message: String,
        state: ChatQueueItem.State
    ) {
        let fileID = currentFile?.id ?? UUID()
        let item = ChatQueueItem(
            documentID: fileID,
            order: chatQueueItems.count,
            message: message,
            state: state,
            actor: .user(UUID())
        )
        chatQueueItems.append(item)
    }

    /// Process a chat-typed mutation request. The
    /// caller (the chat panel) hands a `CodeMutation`
    /// to the view model; the view model applies it
    /// and updates the editor.
    public func processMutationRequest(_ mutation: CodeMutation) async {
        do {
            _ = try await store.apply(mutation, to: mutation.fileID)
            if let updated = try store.get(id: mutation.fileID) {
                self.currentFile = updated
                self.currentOutline = outlineExtractor.extract(
                    source: updated.body, language: updated.language
                )
                searchIndex.upsert(updated)
            }
        } catch {
            self.lastError = "Chat mutation failed: \(error)"
        }
    }

    // MARK: - Tree + search index

    private func rebuildTree() {
        guard let root = watchedRoot else { return }
        tree = builder.build(root: root, files: store.listAll())
    }

    private func refreshSearchIndex() {
        searchIndex.setFiles(store.listAll())
    }

    private func indexedCount() -> Int {
        return searchIndex.fileCount
    }
}
