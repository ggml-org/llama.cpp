# Tessera Productivity — Code Material Surface

> Phase 5 of the productivity surface (see
> `docs/tessera-productivity-design.md` §12.10). The
> Code surface treats source files as first-class
> materials, in parallel with Documents, Notes, and
> the rest of the productivity slice.

## 1. Problem

The agent's primary value-add in v1 is "AI-assisted
writing". The most powerful form of that assistance
is when the user is doing real engineering work —
writing Swift code, debugging a script, drafting a
SQL migration. The agent should be a first-class
collaborator on code, not just on prose. The user
working on a Swift file should be able to type into
the chat panel "add an async version of this
function" and have the agent propose a diff.

v1 ships the source-code editor with:

- line numbers, monospaced font, syntax highlighting
- find-in-file with regex
- multi-cursor
- file-system watch + git read-only integration
- cross-surface links (code ↔ documents ↔ contacts)
- chat-panel-driven code mutations
- the same constitutional receipt chain as Documents

## 2. Why this design

**Code is plain text, not an AST.** Documents have
a rich block tree (headings, lists, code blocks,
tables). Code is one opaque text block. Introducing
a per-language AST would balloon the data model
without adding value — the agent's reasoning operates
on ranges + find/replace, not on a structured tree.
The `CodeMutation` family (`replaceCodeBlock`,
`replaceCodeRange`, `insertCodeAt`) is the code
surface's equivalent of the document mutation API;
it's narrower and dedicated.

**One text view engine for all editor surfaces.**
The per-surface differences (line numbers, monospaced
font, find bar) are configuration on
`EditorMode.code` + `CodeEditorConfiguration`, not
different code paths. Phase 2's `TesseraEditorView`
is the canvas; the Code surface passes the
`CodeEditorConfiguration` payload and the
renderer flips the right switches. The same engine
serves Documents, Notes, and Code.

**DispatchSource per file, not FSEvents.** FSEvents
is a higher-level API that batches events at the
kernel level. We use `DispatchSource.makeFileSystemObjectSource`
per file because:

1. the kernel's per-file events are finer-grained
   (a `.write` per file, not "something in this
   directory tree changed")
2. we already need per-file metadata (size, mtime,
   checksum) to produce the `CodeFile` payload
3. v1 watches at most a few hundred files per
   project
4. the user's root is recursively walked at
   `startWatching()`

A directory-level source is ALSO attached at every
directory in the walk, so file creations in already-
watched directories are detected via the directory's
write event.

**`Process` for git, not a Swift lib.** The mature
Swift Git libraries (SwiftGit, GitSwift) are either
unmaintained or carry a large surface for what we
need. `Process` is a 100-line wrapper around `git
log`, `git diff`, and `git blame`; the parsing is
straightforward (the `--format=...` flags give us
machine-readable output). v2 can swap to vendored
libgit2 if the surface grows.

**No LSP in v1.** A regex-based outline extractor
covers the most common cases (Swift, Python,
TypeScript, JavaScript, Rust, Go, Ruby, Java,
Kotlin, C, C++). Languages without a regex table
get an empty outline + a "no outline available" hint
in the panel. v2 swaps in a Language Server Protocol
integration for proper go-to-definition + completion.

## 3. CodeFile model

```swift
public struct CodeFile: Codable, Sendable, Identifiable, Hashable {
    public let id: UUID
    public var path: String             // absolute path
    public var filename: String         // "Foo.swift"
    public var language: String         // "swift" | "python" | ...
    public var body: String             // the source text
    public var size: Int64
    public var modifiedAt: Date
    public var checksum: String         // "sha256:<hex>"
    public var linkedEntityIDs: [UUID]
    public var tags: [String]
    public var createdAt: Date
    public var updatedAt: Date
}
```

A `CodeFile` is a `graph_entity` row with
`entity_type = 'code'`, `subtype` = the language
tag, `body` = the source text (stored as JSONB with
a single `text` field — the JSON wrapper keeps the
schema extensible; future "code metadata" fields
attach without a migration). The two indexes from
migration 0009 (`idx_entities_code_path`,
`idx_entities_code_language`) make the per-file
lookup and the language-filter dropdown O(log n)
instead of a full scan.

The `checksum` is the SHA-256 of the body
(`"sha256:<hex>"` — same format as
`DocumentAST.contentHash()`). The watcher and the
import path both verify the checksum after reading
the file; the receipt chain is content-addressed via
this field.

Language detection is extension-based; the table
in `CodeFile.detectLanguage(forPath:)` covers the
24 languages the design doc lists (swift, python,
typescript, sql, json, yaml, markdown, shell, rust,
go, c, cpp, ruby, java, kotlin, scala, haskell, lua,
elixir, r, matlab, dockerfile, makefile, ...) plus a
handful of useful additions (toml, xml, html, css,
scss, vue, svelte, proto, graphql, dart). Unknown
extensions fall through to `unknownLanguage` =
`"plain"`.

## 4. Code editor configuration

```swift
public struct CodeEditorConfiguration: Codable, Sendable, Hashable {
    public var showLineNumbers: Bool
    public var syntaxHighlightingLanguage: String?
    public var codeFolding: Bool
    public var multiCursor: Bool
    public var findInFile: Bool
    public var minimap: Bool
    public static let `default` = CodeEditorConfiguration()
}
```

`CodeEditorConfiguration` is a value type the
SwiftUI host passes to `TesseraEditorView`. The
`EditorMode` enum gets a new `codeWithConfig` case
that carries an associated configuration; the bare
`case code` is kept source-compatible with the
Phase 2 callers. The default configuration is
conservative (line numbers on, folding on, multi-
cursor on, find on, minimap off); the host view
shows the user a settings panel to opt into the
minimap and to opt out of folding for tiny files.

The renderer reads the configuration and applies:

- `showLineNumbers` → the gutter is on
- `monospaceFont` from the theme → the text view's font
- `codeFolding` → STTextView's folding region markers
- `multiCursor` → Cmd-click adds cursors
- `findInFile` → Cmd-F opens the inline find bar
- `minimap` → optional SwiftUI Canvas on the right

## 5. File-system watch

```swift
public actor CodeFileWatcher {
    public init(rootURL: URL, ignoreRules: CodeFileWatcherIgnoreRules = .default)
    public func startWatching() async throws
    public func stopWatching()
    public func events() async -> AsyncStream<CodeFileEvent>
    public var watchedRoot: URL { get }
    public func refresh() async
}

public enum CodeFileEvent: Sendable, Equatable, Hashable {
    case created(CodeFile)
    case modified(CodeFile)
    case deleted(path: String)
    case renamed(from: String, to: String)
}
```

The watcher is an `actor`; the consumer awaits
`events()` and iterates the stream. The
`DispatchSource` callback is the only non-actor code
path — it uses a `Sendable` closure to bounce the
event back to the actor for emission.

**Per-file sources.** Every regular file in the
recursive walk gets a `DispatchSource` attached to
an `O_EVTONLY` file descriptor. The kernel fires
`.write` on save, `.delete` on removal, `.rename` on
rename. The actor debounces write bursts (a save
fires multiple `.write` events; the watcher
collapses them into one `CodeFile` read + one
event).

**Per-directory sources.** A directory-level source
is attached at every directory in the walk. When a
new file is created in a directory, the directory's
`.write` event fires; the actor re-walks that
directory level and attaches a per-file source to
any new files, emitting a `.created` event for each.

**Ignore rules.** The default ignores cover the
common "don't import a build artifact" cases
(`.git`, `.build`, `node_modules`, `target`). The
user can extend via a `.tesseraignore` file (a
`.gitignore`-style text file with one pattern per
line; the v1 matcher is a simple `*` glob that
translates to `[^/]*` in a regex).

**Binary files.** A file is treated as binary if its
first 8KB contains a NUL byte. The watcher skips
binary files because the Code surface is for source
code; binary files (PNGs, PDFs) are not user-
editable in v1.

## 6. Git read-only integration

```swift
public actor GitReadOnly {
    public init(repoURL: URL)
    public func validate() async throws
    public func recentCommits(file: CodeFile, limit: Int = 50) async throws -> [GitCommit]
    public func diff(file: CodeFile, since: String) async throws -> GitDiff
    public func blame(file: CodeFile) async throws -> [GitBlame]
}

public struct GitCommit: Codable, Sendable, Identifiable, Hashable {
    public var hash: String
    public var authorName: String
    public var authorEmail: String
    public var date: Date
    public var message: String
    public var filesChanged: [String]
}

public struct GitDiff: Codable, Sendable, Hashable {
    public var file: String
    public var hunks: [DiffHunk]
}

public struct GitBlame: Codable, Sendable, Hashable {
    public var line: Int
    public var commit: GitCommit
    public var originalLine: String
}
```

v1 is read-only (`git log`, `git diff`, `git
blame`). v2 adds `git commit`, `git push`, branch
operations, and PR workflows.

**Repository discovery.** The actor's `init` takes
a `repoURL` (the working copy root). The actor
resolves every git command to that root; relative
paths in the output are relative to the root.

**Error model.** `GitReadOnlyError.notARepository`
when the root isn't a git repo; `gitBinaryUnavailable`
when `/usr/bin/env` can't find git; `gitFailed` when
the subprocess exits non-zero. The actor's
`validate()` is the canonical "is this a repo?"
check; the panel renders the failure as a "not a
git repository" banner.

**v2 write operations.** Adding `git commit`,
`git push`, branch operations, and PR workflows is
out of scope for v1. The actor's API is
append-only-friendly — `commitChanges(message:)`
and `pushTo(remote:branch:)` can be added without
breaking the read-side.

## 7. Code surface UI

Three panes:

- **Sidebar (left)** — the watched root directory,
  expanded as a tree, with file icons by language.
  The view is a SwiftUI `List` + `OutlineGroup` over
  the `CodeFileTree` (which the `CodeFileTreeBuilder`
  rebuilds from the `CodeStore`'s file set on every
  watcher event).
- **Editor (middle)** — the selected file in
  `EditorMode = .code` + `CodeEditorConfiguration`.
  The text view is an `NSTextView` wrapper
  (`CodeTextView`) that reads/writes the file body
  and signals `isDirty`. The save gesture (Cmd-S)
  fires a `CodeMutation.replaceCodeBlock` via
  `viewModel.saveBody`.
- **Detail (right)** — a tab picker (Outline / Git /
  Search); each tab owns its own sub-view. The
  Outline tab shows the regex-extracted function
  list. The Git tab shows recent commits (newest
  first). The Search tab shows workspace-wide search
  results (file → line → match range).

macOS: `NavigationSplitView` with three columns. iOS:
`NavigationStack` with the tree on a sidebar sheet,
the editor as the main view, the git panel collapsed
by default.

**Find in file.** Cmd-F opens the inline find bar
with regex support (the underlying text view's
NSFindInteraction). The find scope is the current
file; workspace-wide search is a separate panel
(Search tab).

**Multi-cursor.** Cmd-click adds cursors; the
underlying text view's NSTextView supports this
natively when `isRichText = false`.

**Minimap.** Off by default per the design doc. The
host view shows a settings panel to opt in; the
minimap is a small SwiftUI Canvas on the right
(scrolls proportionally to the editor).

## 8. Chat panel integration

The user can use the chat panel to drive code
mutations:

- "add an async version of this function" → pending
  chat queue item, agent processes, suggests a diff
- "find all files that import this module" → chat
  panel shows the matching files
- "what's the git history of this file in the last
  month?" → chat panel shows the commits
- "add a test for this function" → agent writes
  the test as a code change

The agent's code changes are mutations on the
`CodeFile` body. The mutation API is the
`CodeMutation` family:

```swift
public enum CodeMutation: Codable, Sendable, Hashable {
    case replaceCodeBlock(fileID: UUID, newBody: String)
    case replaceCodeRange(fileID: UUID, match: String, replacement: String)
    case insertCodeAt(fileID: UUID, position: Int, text: String)
    case addTag(fileID: UUID, tag: String)
    case removeTag(fileID: UUID, tag: String)
    case linkTo(fileID: UUID, otherEntityID: UUID, linkType: String)
    case unlinkFrom(fileID: UUID, otherEntityID: UUID, linkType: String)
}
```

`CodeMutation` is a separate type from `Mutation`
(the document mutation API). The document API is
15+ cases; the code API is 7. Splitting keeps each
API clean and lets the code surface own its own
snapshot semantics (the pre-mutation snapshot is the
prior `body` string + tags + links, not a
`[UUID: Block]` map).

The agent's diff-driven mutations all use the
`replaceCodeBlock` case (the engine computes the
diff stats from the pre/post bodies in the apply
path). The receipt type is `code_file_body_replaced`.

## 9. Receipt model

Every code mutation is a constitutional receipt:

- `code_file_imported` — when added
- `code_file_modified` — when edited (user or agent)
- `code_file_deleted` — when deleted (voidedBy)
- `code_file_renamed` — when renamed
- `code_file_tagged` / `code_file_untagged` — tag
  changes
- `code_file_linked` / `code_file_unlinked` — link
  changes
- `code_file_body_replaced` — generic body change
  (the agent's diff-driven mutations all use this)
- `code_file_git_fetched` — informational; the
  payload carries the latest commit list

Receipts are append-only; the `voidedBy` field on
prior receipts is updated when a delete receipt is
appended.

## 10. Cross-surface links

Code files can be linked to:

- **Documents** (the spec the file implements)
- **Contacts** (the file's author or maintainer)
- **Other code files** (related modules, dependencies)

Links are stored in `entity_links` with a `linkType`
of `implements`, `authored_by`, `depends_on`, or
`related_to`. The graph view uses them as edges; the
file's detail panel lists them in "Related". The
`CodeStore.link(_:to:linkType:weight:)` method
creates the link + a `code_file_linked` receipt.

## 11. Graph view integration (Phase 6)

Code files appear in the graph view. The
`GraphStore.loadAll()` already iterates the
`"code"` entity type (the v1 list at the top of
`loadAllNodes` includes it). The `GraphNode.iconName`
maps `"code"` to `"chevron.left.forwardslash.chevron.right"`
and the color to `.gray` (the existing mapping;
no change needed).

Clicking a code node in the graph view opens the
file in the Code surface. The integration is a
window-level navigation gesture: the graph view's
`onTapGesture` calls the host's `openCodeSurface(file:)`
method, which pushes a new `CodeSurfaceView` onto
the navigation stack.

## 12. Library survey

| Need | Library | Decision |
|---|---|---|
| STTextView | `STTextView` (krzyzanowskim) | Adopt — already in Phase 2 |
| Splash (syntax highlighting) | `Splash` (JohnSundell) | Adopt — already in Phase 2 |
| File-system watch | `DispatchSource.makeFileSystemObjectSource` | Adopt — native |
| Git | `Process` subprocess calling `git log`, `git diff`, `git blame` | Adopt — no Swift lib does it well |
| Minimap | Custom (small SwiftUI Canvas) | Build — design-driven |
| Outline (function list) | Custom regex extractor | Build — v1; LSP in v2 |
| Workspace search | Custom (NSRegularExpression over file bodies) | Build — pure-Swift, no deps |

## 13. Test strategy

The tests live in
`Tests/TesseraCoreTests/Productivity/Materials/Code/`:

- **CodeFileTests** (20 tests) — JSON round-trip,
  filename derivation, language detection (20+
  extensions), checksum computation, entity type /
  subtype conventions.
- **CodeMutationTests** (25 tests) — the
  `CodeMutation` enum + the `CodeMutationEngine`:
  happy paths for every variant, validation errors
  (ambiguous match, position out of range, ...),
  inverse computation for undo.
- **CodeFileTreeTests** (13 tests) — the
  `CodeFileTreeBuilder` + the tree's `flatten()` /
  `node(withID:)` helpers. Empty / single file /
  nested / deeply nested / sort order / stable IDs.
- **CodeOutlineTests** (21 tests) — the
  `CodeOutlineExtractor`: per-language regex tables
  for Swift, Python, TypeScript, JavaScript, Rust,
  Go, Java, Ruby, Kotlin. Comments are skipped.
  Line numbers are correct.
- **CodeSearchIndexTests** (13 tests) — literal
  search, case sensitivity, regex with capture
  groups, max results, upsert + remove, group by
  file.
- **CodeFileWatcherTests** (13 tests) — real
  temp dir + real `DispatchSource`. Create / modify
  / delete events; binary detection; ignore rules;
  refresh.
- **CodeStoreTests** (13 tests) — no data layer
  (in-memory index only): upsert, get by id + by
  path, list / list by language, search, apply
  (mutation engine integration), rename, delete,
  tag, receipts (empty without data layer).
- **GitReadOnlyTests** (14 tests) — `parseCommits`,
  `parseHunks`, `parseBlame` against canned output;
  validate fails for non-repo; real git integration
  on a temp repo (skipped when git is not on PATH).
- **CodeEditorConfigurationTests** (5 tests) — the
  `CodeEditorConfiguration` value type: defaults,
  JSON round-trip, `Hashable`, the `EditorMode`
  extension's `isCodeMode` and `codeDefault`.

Total: **137 new tests, all passing.**

The watcher and git tests use real fs + real git
(per architect's decision to use real-fs testing
over a mock). The other 124 tests are pure logic
(no I/O, no subprocess, no DB).

The pre-existing baseline has 2 known failures in
`ExportFormatTests` (unrelated to this work) and a
crash in `TesseraImporterEventParsingTests.testMalformedUUIDIsSkipped`
(unrelated; pre-existing on main). After this PR,
the same 2 failures remain and the same crash occurs;
the new tests don't introduce any new failures.

## 14. Out of scope

- **Other Materials surfaces** (Tasks, Reminders,
  Calendar, Notes, Email) — separate workers.
- **Git push / PR (v2)** — read-only in v1.
- **LSP integration (v2)** — regex-based outline
  in v1.
- **Multi-file refactor as a single agent action
  (v2)** — v1 is one-file-at-a-time.
- **Terminal integration (v2)** — running shell
  commands from the Code surface.
- **Debugger integration (v2)**.
- **Code review / linting (v2)**.

## Files

### New (TesseraCore)

- `Sources/TesseraCore/Productivity/Materials/Code/CodeFile.swift`
- `Sources/TesseraCore/Productivity/Materials/Code/CodeFileWatcher.swift`
- `Sources/TesseraCore/Productivity/Materials/Code/GitReadOnly.swift`
- `Sources/TesseraCore/Productivity/Materials/Code/CodeMutation.swift`
- `Sources/TesseraCore/Productivity/Materials/Code/CodeReceiptType.swift`
- `Sources/TesseraCore/Productivity/Materials/Code/CodeOutline.swift`
- `Sources/TesseraCore/Productivity/Materials/Code/CodeSearchIndex.swift`
- `Sources/TesseraCore/Productivity/Materials/Code/CodeFileTree.swift`
- `Sources/TesseraCore/Productivity/Materials/Code/CodeStore.swift`
- `Sources/TesseraCore/Productivity/Materials/Code/CodeSurfaceViewModel.swift`

### Modified (TesseraCore)

- `Sources/TesseraCore/Editor/EditorMode.swift` — added
  `CodeEditorConfiguration` and the `codeWithConfig`
  case to `EditorMode`.

### New (TesseraStudioMac)

- `Sources/TesseraStudioMac/Views/Code/CodeSurfaceView.swift`
- `Sources/TesseraStudioMac/Views/Code/CodeFileTreeView.swift`
- `Sources/TesseraStudioMac/Views/Code/CodeEditorPaneView.swift`
- `Sources/TesseraStudioMac/Views/Code/CodeOutlineView.swift`
- `Sources/TesseraStudioMac/Views/Code/CodeGitPanelView.swift`
- `Sources/TesseraStudioMac/Views/Code/CodeSearchPanelView.swift`

### New (tests)

- `Tests/TesseraCoreTests/Productivity/Materials/Code/CodeFileTests.swift`
- `Tests/TesseraCoreTests/Productivity/Materials/Code/CodeMutationTests.swift`
- `Tests/TesseraCoreTests/Productivity/Materials/Code/CodeFileTreeTests.swift`
- `Tests/TesseraCoreTests/Productivity/Materials/Code/CodeOutlineTests.swift`
- `Tests/TesseraCoreTests/Productivity/Materials/Code/CodeSearchIndexTests.swift`
- `Tests/TesseraCoreTests/Productivity/Materials/Code/CodeFileWatcherTests.swift`
- `Tests/TesseraCoreTests/Productivity/Materials/Code/CodeStoreTests.swift`
- `Tests/TesseraCoreTests/Productivity/Materials/Code/GitReadOnlyTests.swift`
- `Tests/TesseraCoreTests/Productivity/Materials/Code/CodeEditorConfigurationTests.swift`

### New (data layer)

- `tools/tessera/db/migrations/0009_code_files.sql`

## "How to use" snippet

```swift
// Set up the store + watcher.
let store = CodeStore(dataLayer: dataLayer)
let root = URL(fileURLWithPath: "/Users/me/Developer/MyProject")
let viewModel = CodeSurfaceViewModel(store: store, watchedRoot: root)

// Drive the SwiftUI surface.
CodeSurfaceView(viewModel: viewModel)

// Open a file (e.g. from the search results).
let file = try store.get(path: "/Users/me/Developer/MyProject/Sources/Foo.swift")!
await viewModel.open(file: file)

// Save a body change.
await viewModel.saveBody("let x = 1\n")

// Get the git history.
let commits = try await viewModel.git?.recentCommits(file: file, limit: 20)

// Search the workspace.
viewModel.searchQuery = "URLProtocol"
await viewModel.runSearch()
```

## Surface sketch

```
+---------------------------------------------------------------------------+
|  Code                                                ~/Developer/MyProj  |
+---------------------+-------------------------+-------------------------+
| Filter files    [x]  |  Foo.swift     L 1-100   |  Outline | Git | Search |
+---------------------+  let x = 1             *  +-------------------------+
| v MyProject         |  func bar() {           |  function bar()        |
|   v Sources         |      print("hi")        |      L3                |
|     > Foo.swift     |  }                      |  function baz()        |
|     > Bar.swift     |                         |      L7                |
|   > Tests           |                         |  class Helper          |
|                     |                         |      L11               |
+---------------------+-------------------------+-------------------------+
| ~/Developer/MyProj  |  3 files   sha256:abc.. |                         |
+---------------------+-------------------------+-------------------------+
```
