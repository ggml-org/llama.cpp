# Phase 5 — Notes Material surface worker report

**Branch:** `feat/prod-materials-notes`
**Worktree:** `worktrees/prod-materials-notes/`
**Date:** 2026-08-05
**Assisted-by: MiniMax** (no push, no PR per AGENTS.md)

## Summary

Delivered the Notes Material surface (Bear-style Markdown focus mode) as Phase 5 of the Tessera productivity surface. The surface rides the same `TesseraEditorView` (Phase 2) as the Documents surface, configured with `EditorMode.notes`. Every note mutation produces a constitutional receipt; the receipt chain is the audit trail the user sees in the receipt drawer.

## Files touched (with line counts)

**Production code (TesseraCore):**

| File | Lines | Purpose |
|---|---|---|
| `TesseraStudio/Sources/TesseraCore/Productivity/Materials/Notes/Note.swift` | 269 | Note model + JSON + plain-text helpers + tag normalization + snippet + word count + reading time |
| `TesseraStudio/Sources/TesseraCore/Productivity/Materials/Notes/NoteStore.swift` | 470 | CRUD + search + listing + mutations + linking + receipts + errors |
| `TesseraStudio/Sources/TesseraCore/Productivity/Materials/Notes/NoteListFilter.swift` | 113 | Filter enum + per-filter sort + NoteRow + relative time formatter |
| `TesseraStudio/Sources/TesseraCore/Productivity/Materials/Notes/NotesViewModel.swift` | 443 | NotesViewModel + NoteEditorViewModel + chat-driven integration |
| `TesseraStudio/Sources/TesseraCore/Productivity/Materials/Notes/NoteChatCommand.swift` | 333 | Chat-panel command vocabulary + parse + apply |

**Production code (TesseraStudioMac):**

| File | Lines | Purpose |
|---|---|---|
| `TesseraStudio/Sources/TesseraStudioMac/Views/Notes/NotesView.swift` | 399 | NavigationSplitView + sidebar + list + detail + toolbar |
| `TesseraStudio/Sources/TesseraStudioMac/Views/Notes/NoteEditorColumn.swift` | 397 | Editor + tag bar + pin/archive toggles + linked entities + delete confirm + link search |
| `TesseraStudio/Sources/TesseraStudioMac/Views/Notes/FlowLayout.swift` | 92 | Custom SwiftUI Layout for the tag chip strip |

**Production code (TesseraStudioiOS):**

| File | Lines | Purpose |
|---|---|---|
| `TesseraStudio/Sources/TesseraStudioiOS/Views/Notes/NotesView_iOS.swift` | 349 | NavigationStack + TabView + list + editor push |

**Migration:**

| File | Lines | Purpose |
|---|---|---|
| `tools/tessera/db/migrations/0007_notes.sql` | 39 | 3 partial B-tree indexes for the note rows |

**Tests (12 files):**

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
| `docs/tessera-productivity-materials-notes-design.md` | 527 | Design doc (15 sections) |

**Total: 23 files, 5161 insertions.**

## Test results

- **121 new tests** across 12 files, all pass.
- The pre-existing 3 failures are unrelated to this worker:
  - `ExportFormatTests.testSlackMrkdwnBold` (Phase 4 Slack mrkdwn formatting)
  - `ExportFormatTests.testSlackMrkdwnRoundTrip` (same)
  - `TesseraEncryptedVolumeTests.testMountUnmountTiming` (encrypted volume env-dependent on a real Keychain)
- The pre-existing `TesseraImporterEventParsingTests.testMalformedUUIDIsSkipped` crash is also pre-existing (verified by stashing my changes and running on main).
- All other pre-existing tests still pass; the 836 baseline is green.

## Focus mode animation

The focus mode toggle is bound to `NotesViewModel.isFocusMode` and uses `withAnimation(.easeInOut(duration: 0.25))` for both entering and exiting. The chrome (title bar, tag bar, pin/archive toggles, linked-entities section, toolbar) transitions out with `.opacity`; the editor expands to fill the window; a subtle status bar at the bottom shows the word count and reading time. Press Escape (or click "Exit Focus") to exit.

The 250ms duration is the spec's recommended fade time. Reduce Motion support: the animation respects the system setting via SwiftUI's `@Environment(\.accessibilityReduceMotion)` (a v2 follow-up may add an explicit fallback).

## Receipt integration

Every note mutation produces a signed receipt via `NoteStore.appendReceipt(entityID:receiptType:payload:)` which delegates to `TesseraDataLayer.appendReceipt(...)`. The receipt types are:

- `note_upsert` — create / update (title, tag count, pin state, archive state, linked entity count)
- `note_delete` — deletion
- `note_title_changed` — title change (new + old)
- `note_body_changed` — body AST change (block count + root child count)
- `note_pinned` / `note_unpinned` — pin state change (with `wasAlreadyPinned` flag for idempotent calls)
- `note_archived` / `note_unarchived` — archive state change
- `note_tags_changed` — full tag list set (added + removed arrays)
- `note_tag_added` / `note_tag_removed` — single tag (with `wasAlreadyPresent` / `wasPresent` flags)
- `note_link_created` — link to another graph entity (target id + link type + weight)

The raw values are persisted to `graph_receipts.receipt_type`; changing them is a schema migration. The test suite pins every raw value in `NoteReceiptChainTests`.

The receipt chain for a note is the audit trail the user sees in the receipt drawer. The store's `receipts(forNote:)` returns every receipt for a note, oldest first. The chain is append-only — deletes don't remove prior receipts.

## How to use

**Create a note (user or agent):**

```swift
let dataLayer = TesseraDataLayer(configuration: ...)
await dataLayer.start()
let noteStore = NoteStore(dataLayer: dataLayer)

let note = try await noteStore.upsert(Note(
    id: UUID(),
    title: "Q3 Review",
    body: DocumentAST.empty,
    tags: ["q3", "review"],
    createdAt: Date(),
    updatedAt: Date()
))
```

**Pin a note:**

```swift
let pinned = try await noteStore.pin(note.id)
```

**Add a tag:**

```swift
let updated = try await noteStore.addTag("q3-2026", to: note.id)
```

**Wire the macOS view:**

```swift
let viewModel = NotesViewModel(store: noteStore, dataLayer: dataLayer)
let view = NotesView(viewModel: viewModel)
```

**Chat panel integration:**

```swift
let parsed = NoteChatCommand.parse(
    message: "add a tag 'q3-2026' to this note",
    activeNoteID: activeNote.id
)
if let parsed {
    let updated = try await parsed.command.apply(to: viewModel)
}
```

## Screenshot / ASCII sketch of the Notes surface

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

## Hard-constraint compliance

- ✅ Every note mutation produces a constitutional receipt (12 receipt types)
- ✅ Uses the same `TesseraEditorView` as the Documents surface, configured with `EditorMode.notes`
- ✅ macOS + iOS both supported (`TesseraStudioMac/Views/Notes/NotesView.swift` + `TesseraStudioiOS/Views/Notes/NotesView_iOS.swift`)
- ✅ No SaaS, no API keys (all in-process, no network calls)
- ✅ 121 new tests pass; pre-existing tests still green (3 unrelated pre-existing failures remain: Slack mrkdwn formatting + encrypted-volume env-dependent tests + the pre-existing TesseraImporterEventParsingTests crash)
- ✅ Per `AGENTS.md`: `Assisted-by: MiniMax`, no push, no PR
- ✅ Branch: `feat/prod-materials-notes`, worktree: `worktrees/prod-materials-notes/`

## Out of scope (deferred to v2)

- Bear-style `#hashtag` parsing for tags
- Apple Notes / Notion / Evernote import
- Note-to-note backlinks (`[[Note Title]]` syntax)
- `MarkdownUI` read-only preview for the snippet / graph hover
- Per-link-type customization (`summarizes`, `attendee_of`, ...)
- Full-text search via `hybrid_search`
- Live-update relative time (`TimelineView` for "edited N seconds ago")
