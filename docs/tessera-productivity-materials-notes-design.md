# Tessera Productivity Surface — Notes Material Design

**Status:** Implemented (Phase 5 of the productivity surface)
**Date:** 2026-08-05
**Companion:** `docs/tessera-productivity-design.md` §12.5 (the spec); `docs/tessera-productivity-ux-research.md` (the evidence)
**Branch:** `feat/prod-materials-notes` · **Worktree:** `worktrees/prod-materials-notes/`

---

## 1. Problem

The productivity surface (per `docs/tessera-productivity-design.md`) needs six material types — Tasks, Reminders, Calendar, Notes, Email, Documents — backed by a single `graph_entity` table and a constitutional-receipts backbone. Phases 1-4 landed the foundations, the editor, the chat panel + receipts drawer, and the import / export pipeline. Phase 6 (parallel) added Contacts + Graph. **This worker delivers the Notes Material surface** — a Bear-style Markdown focus mode that rides the same `TesseraEditorView` (Phase 2) as the Documents surface.

The Notes surface is the third editor surface (Documents, Notes, Code all share the same `TesseraEditorView` engine; the per-surface differences are configuration, not different code paths). It also surfaces the constitutional-receipts backbone most visibly: every tag, pin, archive, link, or body change is a signed receipt the user can inspect in the receipt drawer.

---

## 2. Why this design

| Choice | Rationale |
|---|---|
| **`graph_entity` row with `entity_type = 'note'`** | The universal "one row per thing" pattern from the data layer (Postgres + Valkey). One polymorphic table covers every material; the note-specific columns live in the `body` JSON. The hybrid_search query walks the graph without per-type joins. |
| **Block AST in `body`** | The note body is a `DocumentAST` (Phase 1's Block AST). The note editor binds to the AST the same way the Documents surface does; the `TesseraEditorView` engine handles block-level rendering, mutation, and undo. |
| **Note-level metadata in the struct (not in the AST)** | `title`, `tags`, `pinnedAt`, `archivedAt`, `linkedEntityIDs` are note-level concerns, not block-level. They live on the `Note` struct so the chat panel + the receipt payload can refer to them without parsing the AST. The store is the seam that emits the note-level receipts. |
| **Bear-style Markdown focus mode** | The user wants to write. Bear's signature is the focus mode: click into the note, the chrome fades, the text fills the window. We implement it with SwiftUI animation transitions + Escape-key exit, matching the spec's Bear-style requirement. |
| **Same `TesseraEditorView` as Documents** | Per spec §9, the editor engine is shared. `EditorMode.notes` configures the toolbar (callouts, quotes, but no tables / code blocks) and the animation set (lighter than Documents). One engine, one set of code paths, one team to maintain. |
| **Notes are a per-document editor, not a per-page column** | The note editor is the editor surface for a single note. The list view + tag bar + pin/archive toggles + linked entities wrap around it. The chat panel is a sibling surface; the spec's per-document chat queue stays per-document, but the notes surface has its own global chat (one queue per surface) so the agent can create + edit notes that don't exist yet. |
| **Tags via explicit input (not `#hashtag` parsing)** | v1 uses explicit tag input + normalization. `#hashtag` parsing is a v2 follow-up. The spec lists it as out of scope for Phase 5. |
| **Constitutional receipts are first-class** | Every mutation produces a signed receipt. The receipt chain is the audit trail the user sees in the receipt drawer. |

---

## 3. Note model

```swift
public struct Note: Codable, Sendable, Identifiable, Hashable {
    public let id: UUID
    public var title: String                  // auto-derived from the first heading, or user-set
    public var body: DocumentAST              // the Block AST (Phase 1)
    public var tags: [String]
    public var pinnedAt: Date?                // for the pinned list
    public var archivedAt: Date?              // for the archive list
    public var linkedEntityIDs: [UUID]        // contacts, documents, events, tasks
    public var createdAt: Date
    public var updatedAt: Date
}
```

**Storage.** `graph_entity` row with `entity_type = 'note'`, `subtype = 'markdown'`, `body` = JSONB with the note's Block AST. `label` mirrors the title so the graph view + search by label work without decoding the AST.

**Tag normalization.** Tags are lowercased + trimmed + de-duplicated in `Note.normalizeTags(_:)` (and the init). The normalization is idempotent so re-saving the same tags is a no-op.

**Display title.** `Note.displayTitle` returns the user-set title if non-empty, else the first heading in the body, else `"Untitled"`. The auto-derivation is the v1 substitute for Bear's "first line is the title" rule; the spec lists it as the v1 behavior.

**Snippet.** `Note.snippet(maxLength:)` is the first 200 characters of the body in plain text, with markdown decoration stripped. Used by the list rows + the graph view's hover preview.

**Word count + reading time.** `Note.wordCount` splits the plain text on whitespace; `Note.readingTimeMinutes` is `ceil(wordCount / 250)`. The 250 wpm baseline is the commonly-cited average silent-reading speed (Brysbaert 2019).

---

## 4. Notes list view

Three lists, sharing one `NoteListFilter` enum:

| Filter | Content | Sort order |
|---|---|---|
| `.all` | Every note, excluding archived | `updated_at DESC` |
| `.pinned` | Pinned + non-archived | `pinned_at DESC`, then `updated_at DESC` |
| `.archived` | Archived | `archived_at DESC`, then `updated_at DESC` |

**macOS layout.** `NavigationSplitView` with three columns:
- Sidebar — the three filters + the tag chip strip + the "New Note" button
- Middle — the note rows for the active filter
- Detail — the note editor column (toolbar + tag bar + `TesseraEditorView` + linked-entities section + focus mode status bar)

**iOS layout.** `NavigationStack` with a `TabView` (segmented picker) for the three filters at the top, the list below, and a `.navigationDestination` push for the note editor.

**Each row shows:**
- Title (bold, single line)
- First 200 chars of the body (subtitle, 2 lines max)
- Tags as pills (top 3 + `+N` if more)
- "edited N days ago" relative time
- Pin / archive icons

The relative time is computed by `NoteRow.relativeTimeString(for:now:)` and picks the most natural unit: just now, N min ago, N hr ago, yesterday, N days ago, N weeks ago, formatted date.

The active tag chip (selected from the sidebar's tag list) filters the active list to notes with the given tag. Setting the chip to `nil` clears it.

**Local search.** The macOS list has a `.searchable` field; the input is passed to `NotesViewModel.applyLocalSearch(_:)` which filters the current list by title + body + tags (case-insensitive). v1 keeps the search in memory; v2 will push it to the data layer's `hybrid_search`.

---

## 5. Note editor

The note editor is the same `TesseraEditorView` (Phase 2) as the Documents surface, configured with `EditorMode.notes`. The toolbar promotes callouts, quotes, and lighter animation; it drops tables, code blocks, and images (those are Documents-only).

**The editor column has four sections (top to bottom):**

1. **Title bar** — `TextField` for the title, submitted on Enter / on commit
2. **Tag bar** — the existing tags as removable pills + an "Add tag…" field
3. **Toggles row** — "Pin" toggle, "Archive" toggle, "Link…" button (search-and-link sheet)
4. **Editor** — `TesseraEditorView` in `.notes` mode
5. **Linked entities section** — chips for the linked entity IDs
6. **Focus mode status bar** (only when focus mode is on) — word count + reading time + "Exit Focus" button

**The tag bar** uses a custom `FlowLayout` so the tags wrap onto multiple lines when the row is narrow. The "Add tag…" field submits on Enter and the draft clears on success.

**The pin / archive toggles** are bound to `Note.isPinned` / `Note.isArchived`. The toggle calls `NoteEditorViewModel.togglePinned()` / `toggleArchived()` which go through the store (each one is a separate receipt).

**The "Link…" button** opens a search sheet where the user can paste a target entity UUID. v1 is a placeholder — v2 wires the search to the data layer's `hybrid_search` and resolves the entity by name. The link creates an `entity_link` row AND appends a `note_link_created` receipt.

**The "Delete" toolbar action** opens a confirmation sheet. The user has to confirm; the delete cascades to `entity_links` and `graph_receipts` via the foreign keys, but the receipt chain for the note is preserved (the row count on the receipts table doesn't decrease — receipts are append-only).

**The chat panel can also create / edit / tag / pin / archive notes** via `NoteChatCommand`. The chat panel parses the user's text into one of:

- `createNote(title:tags:)` — "create a new note titled 'Q3 review'"
- `addTag(noteID:tag:)` — "add a tag 'q3-2026' to this note"
- `removeTag(noteID:tag:)` — "remove tag 'q3-2026' from this note"
- `setPinned(noteID:pinned:)` — "pin this note" / "unpin this note"
- `setArchived(noteID:archived:)` — "archive this note" / "unarchive this note"
- `link(noteID:targetEntityID:linkType:)` — "link this note to <entity>"
- `replaceBody(noteID:body:)` — agent-driven (e.g. "summarize this article")
- `delete(noteID:)` — irreversible, requires confirmation

Each command is the typed boundary between the chat panel and the notes surface. The `parse(message:activeNoteID:targetEntityID:)` static method does best-effort parsing; the agent refines the typed command for the long tail.

---

## 6. Chat panel integration

The chat panel can drive the notes surface via `NoteChatCommand.apply(to: NotesViewModel)`. The flow:

1. User types a command in the chat panel
2. The chat panel parses it with `NoteChatCommand.parse(...)` (returns `nil` for non-matching messages, the agent does the long tail)
3. The chat panel calls `command.apply(to: viewModel)` which dispatches to the store
4. The store mutates the note + appends a signed receipt
5. The view model re-projects the rows and the editor

The chat-driven path sets `NotesViewModel.isChatDriven = true` for the duration of the command so the view can show the "working in background" chip (per spec §6.1). The chip clears when the command completes.

**Example: "create a new note titled 'Meeting notes for Q3 review'":**

```swift
let parsed = NoteChatCommand.parse(
    message: "create a new note titled 'Meeting notes for Q3 review'"
)
// parsed?.command == .createNote(title: "Meeting notes for Q3 review", tags: [])

let note = try await parsed!.command.apply(to: viewModel)
// note is the new Note, persisted to the data layer with a note_upsert receipt
```

**Example: "add a tag 'q3-2026' to this note":**

```swift
let parsed = NoteChatCommand.parse(
    message: "add a tag 'q3-2026' to this note",
    activeNoteID: activeNote.id
)
// parsed?.command == .addTag(noteID: activeNote.id, tag: "q3-2026")

let updated = try await parsed!.command.apply(to: viewModel)
// updated has the new tag, with a note_tag_added receipt
```

---

## 7. Receipt model

Every note mutation produces a signed receipt via `NoteStore.appendReceipt(entityID:receiptType:payload:)` which delegates to `TesseraDataLayer.appendReceipt(...)`.

**Receipt types (`NoteReceiptType`):**

| Case | Raw value | Triggered by |
|---|---|---|
| `.upsert` | `note_upsert` | `upsert(_:)` (create + update) |
| `.delete` | `note_delete` | `delete(id:)` |
| `.titleChanged` | `note_title_changed` | `setTitle(_:for:oldNote:)` |
| `.bodyChanged` | `note_body_changed` | `setBody(_:for:)` |
| `.pinned` | `note_pinned` | `pin(_:)` |
| `.unpinned` | `note_unpinned` | `unpin(_:)` |
| `.archived` | `note_archived` | `archive(_:)` |
| `.unarchived` | `note_unarchived` | `unarchive(_:)` |
| `.tagsChanged` | `note_tags_changed` | `setTags(_:for:)` |
| `.tagAdded` | `note_tag_added` | `addTag(_:to:)` |
| `.tagRemoved` | `note_tag_removed` | `removeTag(_:from:)` |
| `.linkCreated` | `note_link_created` | `link(noteID:to:linkType:weight:)` |

The raw values are persisted to `graph_receipts.receipt_type`; changing them is a schema migration. The test suite pins every raw value.

**Idempotency.** Pinning a pinned note is a no-op for the data layer, but the store still appends a `note_pinned` receipt with a `wasAlreadyPinned: true` flag so the audit trail captures the intent. The same applies to unpin, archive, unarchive, and add tag.

**Receipt payloads.** Each receipt's `payload` is a `[String: JSONValue]` map. Examples:

- `note_upsert` → `{ "title": ..., "tagCount": 3, "pinned": true, "archived": false, "linkedEntityCount": 2 }`
- `note_title_changed` → `{ "newTitle": ..., "oldTitle": ... }`
- `note_tag_added` → `{ "tag": "q3-2026", "wasAlreadyPresent": false }`
- `note_link_created` → `{ "targetEntityID": "...", "linkType": "related_to", "weight": 1.0 }`

The payloads are the surface the receipt drawer shows in its expanded view.

**Receipt chain.** The receipt chain is the audit trail the user sees in the receipt drawer. The store's `receipts(forNote:)` returns every receipt for a note, oldest first. The chain is append-only — deletes don't remove prior receipts; the link from the deleted note's `id` to the chain is just the foreign-key target.

---

## 8. Focus mode

Bear's signature: click into the note, the chrome fades, you write, click outside, the chrome returns. v1 implementation:

- **A "Focus" toggle** in the toolbar (Cmd-\\), bound to `NotesViewModel.isFocusMode`
- **The chrome fades** — the title bar, tag bar, pin/archive toggles, linked-entities section, and toolbar all transition out with `.opacity` when focus mode is on
- **The note text fills the window** — the `TesseraEditorView` expands to the full window
- **A subtle status bar at the bottom** shows the word count + reading time
- **Press Escape to exit focus mode** — `onExitCommand` on the macOS view; a "Exit Focus" button on iOS

**Word count + reading time.** `Note.wordCount` (whitespace-split plain text) and `Note.readingTimeMinutes` (`ceil(wordCount / 250)`). The status bar shows `{wordCount} words · {readingTimeMinutes} min read`.

**Animation.** The transition uses `withAnimation(.easeInOut(duration: 0.25))` for both entering and exiting focus mode. The 250ms duration is the spec's recommended fade time (per `docs/tessera-productivity-design.md` §8).

**Accessibility.** Focus mode is a no-op when Reduce Motion is on (the spec's animation primitives all have Reduce Motion fallbacks). The status bar has a VoiceOver label reading "word count N, M minutes read".

---

## 9. Cross-surface links

Notes can be linked to:
- **Documents** (the article the note summarizes)
- **Contacts** (people mentioned in the note)
- **Calendar events** (the meeting the note is from)
- **Tasks** (action items in the note)
- **Other notes** (related notes)

The linking goes through `entity_links` (Phase 1's data layer). The note's `linkedEntityIDs` is a denormalized cache of the linked ids; the store's `link(noteID:to:linkType:weight:)` creates the `entity_link` row AND keeps the cache in sync.

**Link types.** The default is `related_to` (the same vocabulary the Contacts surface uses). Per-link-type customization (e.g. `summarizes` for documents, `attendee_of` for events) is a v2 follow-up.

**The "Link…" sheet.** v1 is a placeholder: the user pastes a target entity UUID, the store creates the link. v2 wires the search to the data layer's `hybrid_search` and resolves the entity by name.

**The linked-entities section** in the note editor shows chips for the linked ids. v1 displays the UUID prefix (`ABCDEF12…`); v2 resolves the UUID to the entity's display label via the data layer.

---

## 10. Graph view integration

Notes appear in the graph view alongside other materials. The `GraphNode` type already maps `entity_type = 'note'` to the `doc.text` icon and `.blue` color (see `GraphModel.swift`). Clicking a note node opens the note in the Notes surface — the link is through the `entity_type` filter the graph view uses for double-click navigation.

**No graph-view code changes** are needed for Phase 5. The graph view's type mapping already covers notes (and Phase 6's Contacts surface added the contact mapping).

**The note's linkedEntityIDs are the graph's edges.** The graph view's edge set is `entity_links`, which the note's `link(...)` call populates. Notes show up as nodes, the linked documents / contacts / events / tasks / notes show up as edges.

**Importance score.** The graph view's `GraphNode.importance` is computed from degree-centrality (number of incoming + outgoing edges) and recency. Pinned notes are always rendered (matching the spec's "pinned nodes are always in the initial view" requirement).

---

## 11. Library survey

| Need | Library | Decision |
|---|---|---|
| Markdown rendering | `MarkdownUI` (gonzalezreal) | **Defer** — the note editor uses the same `TesseraEditorView` as the Documents surface, so Markdown rendering is already covered by the Block AST. `MarkdownUI` would be a v2 read-only fallback for the snippet / graph hover preview. |
| Bear-style chrome | Custom SwiftUI animations | **Build** — small, design-driven. The fade transition + Escape-exit are 30 lines of SwiftUI. |
| Word count | Custom (split on whitespace, count) | **Build** — `Note.wordCount` is 3 lines. No library needed. |
| Reading time | Custom (word count / 250 words per minute) | **Build** — `Note.readingTimeMinutes` is 1 line. The 250 wpm baseline is documented in the spec. |
| Tag chip layout | Custom `FlowLayout` (SwiftUI Layout protocol) | **Build** — `FlowLayout` is a 50-line SwiftUI Layout. No library needed. |
| Search | In-memory title + body + tag filter | **Build (v1)** — push to `hybrid_search` in v2. |

---

## 12. Test strategy

| Test file | What it covers |
|---|---|
| `NoteTests.swift` | Note model: JSON round-trip, tag normalization, display title fallbacks, plain text extraction, first-heading extraction, snippet computation, word count, reading time, pin/archive convenience |
| `NoteStoreTests.swift` | Receipt type vocabulary (raw value pinning), JSON helpers, store construction |
| `NoteListFilterTests.swift` | All / Pinned / Archived filter application + sort order |
| `NoteRowTests.swift` | Row construction, relative time formatter (just now, N min ago, N hr ago, yesterday, N days ago, N weeks ago, formatted date, future date) |
| `NoteChatCommandTests.swift` | Chat-panel command parsing (create / add tag / remove tag / pin / unpin / archive / unarchive / link / summarize) |
| `NotesViewModelTests.swift` | Filter application, local search, tag-chip logic, focus mode toggle |
| `NoteEditorViewModelTests.swift` | Init, document local state, refresh with new note |
| `NoteFocusModeTests.swift` | Focus mode toggle + exit, word count / reading time on the editor view-model |
| `NoteLinkedEntitiesTests.swift` | Linked-entity IDs init / append / remove / JSON round-trip |
| `NoteReceiptChainTests.swift` | Receipt type raw values are stable + Codable + Sendable |
| `NoteGraphIntegrationTests.swift` | Note entity type mapping, graph view integration smoke |
| `NoteMigrationTests.swift` | `0007_notes.sql` migration file exists, is idempotent, indexes are partial |

**121 tests** across the 12 files. All pass. The 3 pre-existing failures (Slack mrkdwn formatting + encrypted-volume env-dependent tests) are unrelated to this worker.

**Integration test** (`NoteStoreIntegrationTests.swift`) is env-gated on `TESSERA_DB_INTEGRATION=1` and exercises the end-to-end `upsert -> receipt -> fetch` flow against a real Postgres connection. Not in v1 (the existing test pattern matches the contact surface — the integration test is a follow-up).

---

## 13. Out of scope

- **Bear-style `#hashtag` parsing for tags** (defer to v2; v1 is explicit tag input)
- **Apple Notes import** (v2)
- **Notion / Evernote import** (v2)
- **Note-to-note backlinks** (`[[Note Title]]` syntax) (v2)
- **Read-only Markdown preview** (`MarkdownUI` fallback for the snippet / graph hover) (v2)
- **Per-link-type customization** (`summarizes`, `attendee_of`, ...) (v2)
- **Full-text search via `hybrid_search`** (v2; v1 is in-memory)
- **Live-update relative time** (`TimelineView` for "edited N seconds ago") (v2)

---

## 14. Files touched

**Production code (`TesseraStudio/Sources/`):**

| File | Lines | Purpose |
|---|---|---|
| `TesseraCore/Productivity/Materials/Notes/Note.swift` | 269 | Note model + JSON + plain-text helpers + tag normalization + snippet + word count + reading time |
| `TesseraCore/Productivity/Materials/Notes/NoteStore.swift` | 470 | CRUD + search + listing + mutations + linking + receipts + errors |
| `TesseraCore/Productivity/Materials/Notes/NoteListFilter.swift` | 113 | Filter enum + per-filter sort + NoteRow view-model row + relative time formatter |
| `TesseraCore/Productivity/Materials/Notes/NotesViewModel.swift` | 367 | View-model for the Notes surface + editor view-model + chat-driven integration |
| `TesseraCore/Productivity/Materials/Notes/NoteChatCommand.swift` | 333 | Chat-panel command enum + parse + apply |
| `TesseraStudioMac/Views/Notes/NotesView.swift` | 305 | macOS view: `NavigationSplitView` + sidebar + list + detail + toolbar |
| `TesseraStudioMac/Views/Notes/NoteEditorColumn.swift` | 318 | macOS note editor column + focus mode + tag bar + linked entities + delete confirm + link search |
| `TesseraStudioMac/Views/Notes/FlowLayout.swift` | 79 | Custom SwiftUI Layout for the tag chip strip |
| `TesseraStudioiOS/Views/Notes/NotesView_iOS.swift` | 280 | iOS view: `NavigationStack` + `TabView` + list + editor push |

**Migration:**

| File | Lines | Purpose |
|---|---|---|
| `tools/tessera/db/migrations/0007_notes.sql` | 39 | 3 partial B-tree indexes for the note rows |

**Tests (`TesseraStudio/Tests/`):**

| File | Tests | Pass | Fail |
|---|---|---|---|
| `NoteTests.swift` | 32 | 32 | 0 |
| `NoteStoreTests.swift` | 4 | 4 | 0 |
| `NoteListFilterTests.swift` | 11 | 11 | 0 |
| `NoteRowTests.swift` | 11 | 11 | 0 |
| `NoteChatCommandTests.swift` | 17 | 17 | 0 |
| `NotesViewModelTests.swift` | 13 | 13 | 0 |
| `NoteEditorViewModelTests.swift` | 5 | 5 | 0 |
| `NoteFocusModeTests.swift` | 6 | 6 | 0 |
| `NoteLinkedEntitiesTests.swift` | 5 | 5 | 0 |
| `NoteReceiptChainTests.swift` | 8 | 8 | 0 |
| `NoteGraphIntegrationTests.swift` | 4 | 4 | 0 |
| `NoteMigrationTests.swift` | 5 | 5 | 0 |
| **Total** | **121** | **121** | **0** |

**Documentation:**

| File | Lines | Purpose |
|---|---|---|
| `docs/tessera-productivity-materials-notes-design.md` | 273 | This doc |

---

## 15. ASCII sketch of the Notes surface

### macOS — All tab (focus mode OFF)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Notes                                              [Focus]  [Refresh]   │
├──────────────┬─────────────────────────────┬────────────────────────────┤
│  Library     │  All Notes                  │  Q3 Review                  │
│              │  ───────────                │  ─────────                  │
│  All      4  │  ┌───────────────────────┐  │  [Pinned]  [Archive]        │
│  Pinned   2  │  │ 📌 Q3 Review          │  │  [#q3 ✕] [#review ✕]        │
│  Archived 1  │  │  First paragraph…     │  │  [+ Add tag]    [Link…]     │
│              │  │  2 hr ago · #q3 #rev…│  │                             │
│  Tags        │  └───────────────────────┘  │  ┌─────────────────────┐    │
│  [q3] [rev]  │  ┌───────────────────────┐  │  │ Q3 Review            │    │
│  [urgent]    │  │ Sprint planning        │  │  │                      │    │
│              │  │  Outline for the       │  │  │ First paragraph…     │    │
│  [+ New]     │  │  upcoming sprint.      │  │  │                      │    │
│              │  │  1 day ago             │  │  │                      │    │
│              │  └───────────────────────┘  │  └─────────────────────┘    │
│              │  ┌───────────────────────┐  │                             │
│              │  │ Untitled               │  │  Linked entities            │
│              │  │  just now              │  │  (none)                     │
│              │  └───────────────────────┘  │                             │
└──────────────┴─────────────────────────────┴────────────────────────────┘
```

### macOS — Focus mode ON (editor fills the window)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│                                                                          │
│      Q3 Review                                                           │
│      ─────────                                                           │
│                                                                          │
│      First paragraph of the note. The user is writing; the              │
│      chrome is faded; the editor fills the window.                       │
│                                                                          │
│      ## Heading                                                          │
│                                                                          │
│      Second section of the note. The user types; the                     │
│      cursor blinks.                                                      │
│                                                                          │
│                                                                          │
│                                                                          │
│                                                                          │
│                                                                          │
│                                                                          │
│                                                                          │
│                                                                          │
│      47 words · 1 min read                              [Exit Focus]    │
└──────────────────────────────────────────────────────────────────────────┘
```

### iOS — All tab

```
┌──────────────────────────────────────┐
│  Notes                       [✎]    │
├──────────────────────────────────────┤
│  [ All │ Pinned │ Archived ]         │
├──────────────────────────────────────┤
│  📌 Q3 Review                         │
│     First paragraph…                  │
│     2 hr ago · #q3 #review           │
│  ─────────────────────────────────    │
│  Sprint planning                      │
│     Outline for the upcoming…        │
│     1 day ago                         │
│  ─────────────────────────────────    │
│  Untitled                             │
│     just now                          │
└──────────────────────────────────────┘
```

---

## 16. How to use

**In code:**

```swift
import TesseraCore

// 1. Set up the data layer + store
let dataLayer = TesseraDataLayer(configuration: ...)
await dataLayer.start()
let noteStore = NoteStore(dataLayer: dataLayer)

// 2. Wire the macOS view
let viewModel = NotesViewModel(store: noteStore, dataLayer: dataLayer)
let view = NotesView(viewModel: viewModel)
```

**Create a note (user or agent):**

```swift
let note = try await noteStore.upsert(Note(
    id: UUID(),
    title: "Q3 Review",
    body: DocumentAST.empty,
    tags: ["q3", "review"],
    createdAt: Date(),
    updatedAt: Date()
))
// `note` is persisted; a `note_upsert` receipt is appended.
```

**Pin a note:**

```swift
let pinned = try await noteStore.pin(note.id)
// `pinned` is the updated Note; a `note_pinned` receipt is appended.
```

**Add a tag:**

```swift
let updated = try await noteStore.addTag("q3-2026", to: note.id)
// `updated` has the new tag; a `note_tag_added` receipt is appended.
```

**Link to a document:**

```swift
let link = try await noteStore.link(
    noteID: note.id,
    to: documentID,
    linkType: "summarizes"
)
// `link` is the EntityLink; a `note_link_created` receipt is appended.
```

**Chat panel integration:**

```swift
let parsed = NoteChatCommand.parse(
    message: "add a tag 'q3-2026' to this note",
    activeNoteID: activeNote.id
)
if let parsed {
    let updated = try await parsed.command.apply(to: viewModel)
    // `updated` is the new note; viewModel.rows is re-projected.
}
```

---

## 17. Receipt chain example

For a note that's been created, pinned, tagged, and edited:

```
note_upsert              { title: "Q3 Review", tagCount: 0, pinned: false, archived: false, linkedEntityCount: 0 }
note_pinned              { wasAlreadyPinned: false }
note_tag_added           { tag: "q3", wasAlreadyPresent: false }
note_tag_added           { tag: "review", wasAlreadyPresent: false }
note_title_changed       { newTitle: "Q3 Review (final)", oldTitle: "Q3 Review" }
note_body_changed        { blockCount: 12, rootChildCount: 12 }
note_link_created        { targetEntityID: "...", linkType: "summarizes", weight: 1.0 }
```

The receipt drawer shows these in order; the user can click any one to see the full payload + the C2PA manifest.

---

## 18. Hard-constraint compliance

- ✅ Every note mutation produces a constitutional receipt (`note_upsert`, `note_title_changed`, `note_body_changed`, `note_pinned`, `note_unpinned`, `note_archived`, `note_unarchived`, `note_tags_changed`, `note_tag_added`, `note_tag_removed`, `note_link_created`, `note_delete`)
- ✅ Uses the same `TesseraEditorView` as the Documents surface, configured with `EditorMode.notes`
- ✅ macOS + iOS both supported
- ✅ No SaaS, no API keys (all in-process, no network)
- ✅ 121 new tests pass; pre-existing 836 tests still green (3 unrelated pre-existing failures remain: Slack mrkdwn formatting + 2 encrypted-volume env-dependent tests)
- ✅ Per `AGENTS.md`: `Assisted-by: MiniMax`, no push, no PR
