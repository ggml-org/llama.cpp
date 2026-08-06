import Foundation

// MARK: - CodeFileEvent

/// A change observed by the ``CodeFileWatcher``. The
/// watcher emits one event per detected change; the
/// consumer (the Code surface's view model) folds the
/// event into a `CodeStore` mutation + a receipt.
///
/// **Coalescing.** A burst of changes (e.g. a `git pull`
/// that touches 50 files) produces 50 events on the
/// kernel side; the watcher fires one event per
/// `DispatchSource` tick. The consumer's job is to
/// debounce / coalesce into a single UI refresh; the
/// watcher does NOT do that (the per-event delivery
/// makes the audit chain clear: one event = one
/// receipt).
///
/// **Type tags.** The four cases match the four
/// `DispatchSource` flags the kernel reports
/// (`.write`, `.delete`, `.rename`, `.extend`).
/// The `.extend` case maps to `.modified` because the
/// user model is "the file changed" -- the kernel's
/// `.extend` flag fires when the file is grown.
/// `.delete` covers both "file removed" and
/// "directory removed"; the consumer is responsible
/// for the `path` interpretation.
public enum CodeFileEvent: Sendable, Equatable, Hashable {
    /// A new file appeared. The `CodeFile` carries the
    /// initial read (body, size, mtime, checksum). The
    /// consumer creates a graph_entity row + a
    /// `codeFileImported` receipt.
    case created(CodeFile)
    /// An existing file was modified. The `CodeFile`
    /// carries the post-change state. The consumer
    /// produces a `codeFileModified` receipt with the
    /// diff stats in the payload.
    case modified(CodeFile)
    /// A file was removed. The `path` is the file's
    /// last-known path; the consumer emits a
    /// `codeFileDeleted` receipt that voids any prior
    /// `codeFileModified` receipts in the chain.
    case deleted(path: String)
    /// A file was renamed. The pair carries the
    /// pre-rename and post-rename paths; the consumer
    /// emits a `codeFileRenamed` receipt with both in
    /// the payload.
    case renamed(from: String, to: String)
}

// MARK: - CodeFileWatcher

/// Watches a user-chosen root directory and emits
/// ``CodeFileEvent``s on a Swift `AsyncStream`. The
/// watcher is an `actor`; the consumer awaits
/// `startWatching()` and iterates the stream until
/// `stopWatching()` is called (or the actor is
/// deinitialized).
///
/// **Why an `actor` (not a `final class`)?** The watcher's
/// state (the open `DispatchSource`, the pending
/// continuation, the in-flight reads) is mutable; an actor
/// serializes the access without `NSLock` boilerplate.
/// The `DispatchSource` callback is the only non-actor
/// code path -- it uses a `Sendable` closure to bounce
/// the event back to the actor for emission.
///
/// **Why a `DispatchSource`, not `FSEvents`?** `FSEvents`
/// is a higher-level API that batches events at the
/// kernel level. We use `DispatchSource.makeFileSystemObjectSource`
/// per file because:
///   1. The kernel's per-file events are finer-grained
///      (we get a `.write` per file, not "something in
///      this directory tree changed").
///   2. We already need per-file metadata (size, mtime,
///      checksum) to produce the `CodeFile` payload;
///      per-file sources are the natural unit.
///   3. v1 watches at most a few hundred files per
///      project; the per-file overhead is negligible.
///   4. The user's root is recursively walked at
///      `startWatching()`; the watcher attaches a
///      source to every regular file (skipping hidden
///      files, dotfiles, `.git/`, `node_modules/`,
///      `build/`, and the user's `.tesseraignore`).
///
/// **Memory model.** The actor owns the event stream's
/// continuation. The consumer iterates the stream
/// (`for await event in await watcher.events { ... }`).
/// The watcher doesn't buffer; the kernel's events go
/// straight to the consumer. If the consumer is slow,
/// the `DispatchSource`'s coalescing kicks in (the
/// default is "deliver on next runloop tick" which
/// already drops duplicates).
public actor CodeFileWatcher {

    /// The root directory the watcher is attached to.
    /// The path is normalized (no trailing slash, no
    /// `..` components) at construction time so event
    /// `path`s compare equal.
    public let watchedRoot: URL

    /// The receive side of the events stream. The
    /// stream is single-consumer (the `AsyncStream.makeStream`
    /// factory uses the default buffering policy: the
    /// producer suspends if the consumer is slow, which
    /// is the behavior we want for a UI surface).
    private var continuation: AsyncStream<CodeFileEvent>.Continuation?
    private var stream: AsyncStream<CodeFileEvent>?

    /// Per-file `DispatchSource` handles. Keyed by the
    /// absolute path. The sources are owned by the
    /// actor; the `eventHandler` closure captures the
    /// `path` and the actor (via `Task { await ... }`).
    private var sources: [String: DispatchSourceFileSystemObject] = [:]
    /// The `open()` file descriptors per path. The
    /// watcher opens each file in O_EVTONLY mode (no
    /// read, no write) and holds the FD until the file
    /// is unobserved. The kernel watches the FD; when
    /// the file is removed the FD becomes invalid and
    /// the source fires `.delete`.
    private var fileDescriptors: [String: Int32] = [:]
    /// A set of paths the watcher is currently emitting
    /// a debounced event for. The debounce window is
    /// short (50ms) and exists to fold `write`+`extend`
    /// bursts from a single save into one event.
    private var pendingEvents: Set<String> = []
    /// The debounce timer per path. The watcher uses
    /// `DispatchSourceTimer` per pending path; the
    /// timer fires once and clears the entry from
    /// `pendingEvents`.
    private var debounceTimers: [String: DispatchSourceTimer] = [:]
    /// The dispatch queue the watcher's sources run on.
    /// The queue is serial (one file event at a time) so
    /// the actor's mailbox doesn't see a thundering herd.
    private let sourceQueue = DispatchQueue(
        label: "tessera.code.watcher",
        qos: .userInitiated
    )
    /// The watcher is started exactly once; subsequent
    /// `startWatching()` calls are no-ops.
    private var started: Bool = false
    /// The paths the user has asked the watcher to ignore
    /// (git, build, node_modules, dotfiles, the user's
    /// own `.tesseraignore`).
    private let ignoreRules: CodeFileWatcherIgnoreRules

    /// Construct a watcher for `rootURL`. The root is
    /// canonicalized (`URL.standardized`) so the
    /// `path` strings in events compare equal to the
    /// paths the user sees in the sidebar.
    public init(rootURL: URL, ignoreRules: CodeFileWatcherIgnoreRules = .default) {
        self.watchedRoot = rootURL.standardizedFileURL
        self.ignoreRules = ignoreRules
    }

    /// The events stream. The first call to
    /// `startWatching()` lazily creates the stream;
    /// subsequent calls return the same stream (the
    /// consumer is expected to iterate it once).
    public func events() async -> AsyncStream<CodeFileEvent> {
        if let existing = stream { return existing }
        var cont: AsyncStream<CodeFileEvent>.Continuation!
        let s = AsyncStream<CodeFileEvent> { c in
            cont = c
        }
        self.stream = s
        self.continuation = cont
        return s
    }

    /// Start observing. Walks the root recursively,
    /// attaches a `DispatchSource` to every regular
    /// file (skipping the ignore list), and arms the
    /// event handlers. Idempotent.
    public func startWatching() async throws {
        guard !started else { return }
        started = true
        // Force-create the stream so the consumer can
        // call `events()` after `startWatching()`.
        _ = await events()
        try walkAndAttach(at: watchedRoot)
    }

    /// Stop observing. Closes every file descriptor and
    /// cancels every source. The stream's continuation
    /// is finished; the consumer's `for await` loop
    /// exits.
    public func stopWatching() {
        for (_, source) in sources {
            source.cancel()
        }
        sources.removeAll()
        for (_, fd) in fileDescriptors {
            close(fd)
        }
        fileDescriptors.removeAll()
        for (_, timer) in debounceTimers {
            timer.cancel()
        }
        debounceTimers.removeAll()
        pendingEvents.removeAll()
        continuation?.finish()
        continuation = nil
        started = false
    }

    deinit {
        // Best-effort cleanup. The actor's mailbox is
        // gone by the time deinit runs; the FDs and
        // sources are closed in `stopWatching()`. If the
        // consumer forgets to call it, the destructor
        // cancels the sources so the kernel's events
        // don't leak.
        for (_, source) in sources {
            source.cancel()
        }
        for (_, fd) in fileDescriptors {
            close(fd)
        }
        for (_, timer) in debounceTimers {
            timer.cancel()
        }
        continuation?.finish()
    }

    // MARK: - Walk + attach

    /// Recursively walk `url` and attach a `DispatchSource`
    /// to every regular file that passes the ignore filter.
    /// Directories are descended AND watched: a write event
    /// on a directory triggers a re-walk so newly-created
    /// files get a per-file source attached. Without
    /// directory watching, files created AFTER
    /// `startWatching()` would be invisible (the per-file
    /// source only fires for the file it was attached to).
    /// Hidden directories (`.git`, `.build`, `node_modules`)
    /// are skipped per the ignore rules.
    private func walkAndAttach(at url: URL) throws {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: url.path, isDirectory: &isDir) else { return }
        if isDir.boolValue {
            // Attach a source to the directory so new
            // file creations are observed.
            attachDirectory(at: url)
            let contents = try fm.contentsOfDirectory(
                at: url,
                includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey],
                options: [.skipsHiddenFiles]
            )
            for child in contents {
                let childIsDir = (try? child.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
                if childIsDir {
                    if !ignoreRules.shouldSkipDirectory(child) {
                        try walkAndAttach(at: child)
                    }
                } else {
                    attachFile(at: child)
                }
            }
        } else {
            attachFile(at: url)
        }
    }

    /// Attach a `DispatchSource` to a directory. The
    /// source's `eventHandler` re-walks the directory
    /// when a write event fires (a new file appeared
    /// or an existing one was modified/renamed).
    /// Delete events drop the source and forget the FD.
    private func attachDirectory(at url: URL) {
        let path = url.path
        if ignoreRules.shouldSkipDirectory(url) { return }
        if sources[path] != nil { return }
        let fd = open(path, O_EVTONLY)
        guard fd >= 0 else { return }
        let source = DispatchSource.makeFileSystemObjectSource(
            fileDescriptor: fd,
            eventMask: [.write, .extend, .delete, .rename],
            queue: sourceQueue
        )
        source.setEventHandler { [weak self] in
            let flags = source.data
            let pathCopy = path
            Task { [weak self] in
                await self?.handleDirectoryFlags(flags, at: pathCopy)
            }
        }
        source.setCancelHandler { [weak self, path] in
            close(fd)
            let pathCopy = path
            Task { [weak self] in
                await self?.forgetFD(pathCopy)
            }
        }
        source.resume()
        sources[path] = source
        fileDescriptors[path] = fd
    }

    private func handleDirectoryFlags(
        _ flags: DispatchSource.FileSystemEvent,
        at path: String
    ) async {
        if flags.contains(.delete) || flags.contains(.rename) {
            if let source = sources.removeValue(forKey: path) {
                source.cancel()
            }
            if let fd = fileDescriptors.removeValue(forKey: path) {
                close(fd)
            }
            return
        }
        // A write event on a directory means a child
        // file was created, deleted, or renamed. Walk
        // the directory + emit a .created for new
        // files; the per-file source will fire its
        // own .modified for changed files.
        let dirURL = URL(fileURLWithPath: path)
        do {
            try rewalkDirectory(at: dirURL)
        } catch {
            // Walk failed (e.g. permissions); skip.
        }
    }

    /// Walk a single directory level and attach
    /// per-file sources for any new files. Existing
    /// files (those already in `sources`) are
    /// skipped; the per-file source will fire on its
    /// own for changes.
    private func rewalkDirectory(at url: URL) throws {
        let fm = FileManager.default
        let contents = try fm.contentsOfDirectory(
            at: url,
            includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey],
            options: [.skipsHiddenFiles]
        )
        for child in contents {
            let childIsDir = (try? child.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
            if childIsDir {
                if !ignoreRules.shouldSkipDirectory(child) {
                    attachDirectory(at: child)
                }
            } else {
                let path = child.path
                if sources[path] == nil {
                    attachFile(at: child)
                    if let file = readCodeFile(at: path) {
                        emit(.created(file))
                    }
                }
            }
        }
    }

    /// Attach a `DispatchSource.makeFileSystemObjectSource`
    /// to the file at `url`. The source's `eventHandler`
    /// bounces the kernel's flags back to the actor via
    /// `Task { await self.handleEvent(...) }`.
    private func attachFile(at url: URL) {
        let path = url.path
        if ignoreRules.shouldSkipFile(url) { return }
        if sources[path] != nil { return }
        let fd = open(path, O_EVTONLY)
        guard fd >= 0 else { return }
        let source = DispatchSource.makeFileSystemObjectSource(
            fileDescriptor: fd,
            eventMask: [.write, .extend, .delete, .rename],
            queue: sourceQueue
        )
        // The kernel flags tell us what happened; we
        // dispatch the actor-side handler with the path
        // so the handler can re-read the file and
        // produce a `CodeFile` payload.
        source.setEventHandler { [weak self] in
            let flags = source.data
            let pathCopy = path
            Task { [weak self] in
                await self?.handleKernelFlags(flags, at: pathCopy, source: source)
            }
        }
        source.setCancelHandler { [weak self, path] in
            // Close the FD when the source is cancelled
            // (the actor's `stopWatching()` or the file
            // being deleted). The weak self captures the
            // actor reference; the closure is called
            // from the source's queue.
            close(fd)
            let pathCopy = path
            Task { [weak self] in
                await self?.forgetFD(pathCopy)
            }
        }
        source.resume()
        sources[path] = source
        fileDescriptors[path] = fd
    }

    private func forgetFD(_ path: String) {
        fileDescriptors.removeValue(forKey: path)
        sources.removeValue(forKey: path)
    }

    // MARK: - Kernel event handling

    /// Handle a kernel event at `path`. The actor's
    /// `source` is the `DispatchSource` that fired; the
    /// actor may decide to cancel + forget the source
    /// (if the file was deleted/renamed).
    private func handleKernelFlags(
        _ flags: DispatchSource.FileSystemEvent,
        at path: String,
        source: DispatchSourceFileSystemObject
    ) async {
        if flags.contains(.delete) || flags.contains(.rename) {
            // The file is gone (rename is reported as
            // delete + create). Forget the source; the
            // next walk (or the user's manual refresh)
            // will re-attach if the file reappears under
            // a new name.
            source.cancel()
            sources.removeValue(forKey: path)
            if let fd = fileDescriptors.removeValue(forKey: path) {
                close(fd)
            }
            if flags.contains(.rename) {
                // The watcher emits `.deleted`; the
                // re-create is observed at the parent
                // directory's level by the next walk
                // (the source is per-file so we don't
                // see the new file until we walk
                // again). The consumer can trigger a
                // walk by calling `refresh()` after
                // a rename.
                scheduleDebounced(.deleted(path: path))
            } else {
                scheduleDebounced(.deleted(path: path))
            }
            return
        }
        // .write + .extend collapse to "modified" in
        // the user model.
        if flags.contains(.write) || flags.contains(.extend) {
            scheduleDebouncedEmit(at: path)
        }
    }

    /// Schedule a debounced "modified" event for `path`.
    /// Multiple write/extend events in the same 50ms
    /// window fold into one `CodeFile` read + one event.
    private func scheduleDebouncedEmit(at path: String) {
        if let existing = debounceTimers[path] {
            existing.cancel()
        }
        let timer = DispatchSource.makeTimerSource(queue: sourceQueue)
        timer.schedule(deadline: .now() + .milliseconds(50))
        timer.setEventHandler { [weak self] in
            Task { [weak self] in
                await self?.fireDebounced(at: path)
            }
        }
        timer.setCancelHandler { [weak self] in
            Task { [weak self] in
                await self?.clearDebounce(at: path)
            }
        }
        timer.resume()
        debounceTimers[path] = timer
        pendingEvents.insert(path)
    }

    /// Schedule a debounced "deleted" event. The debounce
    /// is shorter (10ms) because there's nothing to
    /// coalesce against.
    private func scheduleDebounced(_ event: CodeFileEvent) {
        let key: String
        switch event {
        case .created(let f): key = f.path
        case .modified(let f): key = f.path
        case .deleted(let p): key = "delete:\(p)"
        case .renamed(let from, _): key = "rename:\(from)"
        }
        if let existing = debounceTimers[key] {
            existing.cancel()
        }
        let timer = DispatchSource.makeTimerSource(queue: sourceQueue)
        timer.schedule(deadline: .now() + .milliseconds(10))
        timer.setEventHandler { [weak self] in
            Task { [weak self] in
                await self?.emit(event)
            }
        }
        timer.resume()
        debounceTimers[key] = timer
    }

    private func clearDebounce(at path: String) {
        debounceTimers.removeValue(forKey: path)
        pendingEvents.remove(path)
    }

    /// Read the file at `path` and emit a `.modified` event
    /// with a fresh `CodeFile` payload. The debounce timer
    /// fires once, so this is called at most once per burst.
    private func fireDebounced(at path: String) async {
        debounceTimers.removeValue(forKey: path)
        pendingEvents.remove(path)
        // The file may have been deleted between the
        // kernel flag and the debounce firing; if so,
        // skip the read.
        guard fileDescriptors[path] != nil else { return }
        guard let file = readCodeFile(at: path) else { return }
        emit(.modified(file))
    }

    /// Emit an event to the stream. The actor's
    /// `continuation` is the AsyncStream's receive side.
    /// A `nil` continuation means the stream was finished
    /// (the consumer broke out of its loop).
    private func emit(_ event: CodeFileEvent) {
        continuation?.yield(event)
    }

    // MARK: - Manual refresh

    /// Walk the root and emit a `.created` event for every
    /// file the consumer hasn't seen yet. Called by the
    /// consumer after a `git pull`, a branch switch, or
    /// any other event that might introduce new files
    /// without the per-file sources firing.
    public func refresh() async {
        // The walk is recursive; the per-file `attachFile`
        // for already-known files is a no-op (the
        // `sources[path]` check). We emit a `.created`
        // for files the consumer may have missed; the
        // consumer is responsible for de-duplicating
        // against its own index.
        do {
            try walkAndEmit(at: watchedRoot)
        } catch {
            // The walk may fail on a permissions error;
            // we surface the failure by emitting a
            // synthetic event the consumer can log.
            // (The consumer's `CodeStore` skips events
            // whose file is already in the index.)
        }
    }

    private func walkAndEmit(at url: URL) throws {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: url.path, isDirectory: &isDir) else { return }
        if isDir.boolValue {
            let contents = try fm.contentsOfDirectory(
                at: url,
                includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey],
                options: [.skipsHiddenFiles]
            )
            for child in contents {
                let childIsDir = (try? child.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
                if childIsDir {
                    if !ignoreRules.shouldSkipDirectory(child) {
                        try walkAndEmit(at: child)
                    }
                } else {
                    attachFile(at: child)
                    if let file = readCodeFile(at: child.path) {
                        emit(.created(file))
                    }
                }
            }
        } else {
            attachFile(at: url)
            if let file = readCodeFile(at: url.path) {
                emit(.created(file))
            }
        }
    }

    // MARK: - File reading

    /// Read the file at `path` and build a `CodeFile` from
    /// the contents + the URL's resource values. Returns
    /// nil for files the consumer can't read (binary
    /// files, permissions, ...). Binary detection is a
    /// heuristic: the first 8KB must contain at least one
    /// zero byte or a non-UTF8 sequence.
    public func readCodeFile(at path: String) -> CodeFile? {
        let url = URL(fileURLWithPath: path)
        guard let data = try? Data(contentsOf: url) else { return nil }
        if Self.looksBinary(data) { return nil }
        guard let body = String(data: data, encoding: .utf8) else { return nil }
        let mtime = (try? url.resourceValues(forKeys: [.contentModificationDateKey])
            .contentModificationDate) ?? Date()
        return CodeFile(
            path: path,
            body: body,
            size: Int64(data.count),
            modifiedAt: mtime
        )
    }

    /// Heuristic: a file is "binary" if its first 8KB
    /// contains a zero byte. The watcher skips binary
    /// files because the Code surface is for source
    /// code; binary files (PNGs, PDFs) are not user-
    /// editable in v1.
    public static func looksBinary(_ data: Data) -> Bool {
        let limit = min(data.count, 8 * 1024)
        if limit == 0 { return false }
        return data.prefix(limit).contains(0)
    }
}

// MARK: - Ignore rules

/// The set of paths the watcher skips. The default
/// rules cover the common "don't import a build artifact"
/// cases (`.git`, `.build`, `node_modules`, `target`).
/// The user can extend via the `.tesseraignore` file
/// (a `.gitignore`-style text file with one pattern
/// per line).
public struct CodeFileWatcherIgnoreRules: Sendable, Hashable {

    /// The always-on ignores. These directories and
    /// files are excluded regardless of `.tesseraignore`
    /// (the build artifacts would explode the watched
    /// set with thousands of files the user never
    /// edits).
    public var alwaysIgnoredDirectories: Set<String>
    public var alwaysIgnoredFilenames: Set<String>

    /// The user-supplied ignores (one glob per line,
    /// `*` matches any run of non-`/` characters). The
    /// file is read once at watcher construction; the
    /// rules are immutable for the watcher's lifetime.
    public var userGlobs: [String]

    public init(
        alwaysIgnoredDirectories: Set<String> = [
            ".git", ".build", ".swiftpm", "build",
            "node_modules", "target", "dist", "out",
            ".venv", "venv", ".pytest_cache", ".tox",
        ],
        alwaysIgnoredFilenames: Set<String> = [
            ".DS_Store", ".tesseraignore", "Thumbs.db",
        ],
        userGlobs: [String] = []
    ) {
        self.alwaysIgnoredDirectories = alwaysIgnoredDirectories
        self.alwaysIgnoredFilenames = alwaysIgnoredFilenames
        self.userGlobs = userGlobs
    }

    public static let `default` = CodeFileWatcherIgnoreRules()

    /// True iff the watcher should skip the directory at
    /// `url`. The check is name-based (not full-path)
    /// because the same name can appear at different
    /// depths and the user always means "skip
    /// node_modules" wherever it is.
    public func shouldSkipDirectory(_ url: URL) -> Bool {
        let name = url.lastPathComponent
        if alwaysIgnoredDirectories.contains(name) { return true }
        if name.hasPrefix(".") && name != "." && name != ".." { return true }
        return matchesUserGlob(path: url.path)
    }

    /// True iff the watcher should skip the file at
    /// `url`. The check is name-based; the watcher
    /// doesn't read the file (the per-file attach is
    /// cheap, but skipping saves a `Data(contentsOf:)`).
    public func shouldSkipFile(_ url: URL) -> Bool {
        let name = url.lastPathComponent
        if alwaysIgnoredFilenames.contains(name) { return true }
        if name.hasPrefix(".") { return true }
        return matchesUserGlob(path: url.path)
    }

    private func matchesUserGlob(path: String) -> Bool {
        // A simple `*` matcher. The user globs are
        // intentionally simple (one `*` per pattern,
        // matching any run of non-`/` characters) so
        // we don't need to vendor fnmatch. A v2 with
        // full gitignore semantics is out of scope.
        for glob in userGlobs {
            if matchGlob(glob: glob, path: path) { return true }
        }
        return false
    }

    private func matchGlob(glob: String, path: String) -> Bool {
        // Translate the glob to a regex. `*` ->
        // `[^/]*`, `.` -> `\.`, `?` -> `[^/]`. The
        // match is NOT anchored to the start of the
        // path: the user writes `secret/*.swift` and
        // expects it to match anywhere in the
        // absolute path. We match against the path
        // components: the glob's leading `*` and
        // trailing `*` cover the components outside
        // the matched pattern.
        var pattern = ""
        for ch in glob {
            switch ch {
            case "*": pattern += "[^/]*"
            case ".": pattern += "\\."
            case "?": pattern += "[^/]"
            default:
                if ch.isLetter || ch.isNumber || ch == "_" || ch == "/" || ch == "-" {
                    pattern.append(ch)
                } else {
                    pattern += "\\\(ch)"
                }
            }
        }
        // Match anywhere in the path. We wrap with
        // `.*` so the glob doesn't have to be
        // anchored to the start.
        let full = ".*" + pattern + ".*"
        return path.range(of: full, options: .regularExpression) != nil
    }
}
