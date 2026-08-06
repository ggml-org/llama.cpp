# Tessera Studio — Phase 5: Tasks Material

**Status:** implemented on `feat/prod-materials-tasks` (off
`feat/prod-chat-receipts`).
**Sources read:** `docs/tessera-productivity-design.md`
§12.2 (Tasks), §15 (Phase 5), §13 (cross-platform).
**Branch:** `feat/prod-materials-tasks`. Worktree:
`worktrees/prod-materials-tasks/`. No push, no PR.

---

## 1. Problem

The productivity surface needs a Tasks material — the
Things 3-style inbox/today/upcoming/anytime/someday view
with natural language input. The data layer
(`TesseraDataLayer`) is on `main`; Phases 1-4 + 6 are
merged. Phase 5 ships the Tasks surface alongside the
other Materials slices (Reminders, Calendar, Notes, Email,
Code, Documents, Contacts, Graph).

Tasks are special among the Materials:

- They are the most frequently mutated entity (the user
  types into the Inbox constantly).
- They have rich filtering semantics (Today, Upcoming,
  Anytime, Someday) that don't fit the cross-cutting
  `hybrid_search` cleanly.
- They are the primary input for natural language
  parsing (the chat panel's most common command is "add a
  task").
- They drive the receipt chain the most (every task
  mutation is a receipt).

The Phase 5 work is the v1 of the Tasks surface. v2 will
add recurrence rules, reminder notifications, Apple
Reminders import, and Google Tasks import.

---

## 2. Why this design

The locked-in choices, with one-line rationale:

| Choice | Rationale |
|---|---|
| **Tasks are `graph_entity` rows** with `entity_type = 'task'` | Phase 1's universal pattern; the agent's `hybrid_search` returns them with the same shape as documents, contacts, and other materials. |
| **Five Things-3-style lists** (Inbox / Today / Upcoming / Anytime / Someday) | The spec's §12.2 is explicit about this; Things 3's vocabulary is the user's mental model. |
| **Today auto-populates from due date** (overdue + next 24h) | The spec's §12.2 rule: the user doesn't manually move tasks to Today when they become due. |
| **Rule-based NLU parser** for v1 (no LLM call) | The patterns are well-known (Things 3, Todoist, Fantastical all do this); the LLM-based enhancement is a v2. |
| **Every task mutation produces a receipt** | The constitutional-receipt backbone is non-negotiable; the receipt chain answers "who created this task, when, with what input". |
| **Filter + sort client-side** (read all tasks, filter in memory) | The data layer's `idx_entities_task_list` and `idx_entities_task_due` partial indexes (migration 0004_tasks.sql) make the SQL scan O(matching rows) for 1k+ tasks; the client-side filter is what the UI binds to without a round-trip. |

The deep design rationale — the entity model, the list
semantics, the NLU grammar, the receipt types — lives in
the rest of this doc. The source of truth is the code; the
spec is this doc.

---

## 3. Task entity model

```swift
public struct ProductivityTask: Codable, Sendable, Identifiable, Hashable {
    public let id: UUID
    public var title: String
    public var notes: String
    public var dueAt: Date?
    public var completedAt: Date?
    public var priority: Priority            // .none | .low | .medium | .high
    public var tags: [String]
    public var list: List                    // .inbox | .today | .upcoming | .anytime | .someday
    public var linkedEntityIDs: [UUID]       // linked contacts, documents, calendar events
    public var sourceURL: String?            // where the task came from
    public var createdAt: Date
    public var updatedAt: Date
}
```

**Naming note:** the spec calls this `Task`, but the
Swift `Task` concurrency type is in the same module
(`TesseraCore`). To avoid shadowing the Swift type
(which would break pre-existing files that use
`Task.detached`, `Task.isCancelled`, etc.), the
implementation uses the namespaced name
`ProductivityTask`. The other Productivity materials
(Contact, Document) have no such collision.

**Storage.** Tasks are `graph_entity` rows with
`entity_type = 'task'`, `subtype` = the list name, `body`
= JSONB with the task fields. `label` is the title (for
the graph view / search prefix).

**Migration `0004_tasks.sql`.** Two partial B-tree
indexes:

- `idx_entities_task_due` on `(body->>'dueAt')` — the
  Today / Upcoming filters use due date as the primary
  axis.
- `idx_entities_task_list` on `(body->>'list')` — the
  manual-list filters (Inbox / Anytime / Someday) use
  the list field.

The partial `WHERE entity_type = 'task'` predicate keeps
the indexes narrow and write-cheap.

**JSON helpers.** `ProductivityTask.jsonData()` /
`ProductivityTask.from(jsonData:)` produce deterministic
JSON (sorted keys + ISO-8601 dates) so the receipt chain
can compare task content across receipts.

---

## 4. List views

Five lists, each with its own UI:

| List | Filter | Sort | macOS column | iOS tab |
|---|---|---|---|---|
| **Inbox** | `task.list == .inbox` AND not completed | priority desc, then title asc | Sidebar | Tab strip |
| **Today** | `dueAt <= now + 24h` AND not completed (overdue + next 24h) | due date asc | Sidebar | Tab strip |
| **Upcoming** | `now + 24h < dueAt <= now + 7d` AND not completed | due date asc | Sidebar | Tab strip |
| **Anytime** | `task.list == .anytime` AND not completed | priority desc, then title asc | Sidebar | Tab strip |
| **Someday** | `task.list == .someday` AND not completed | priority desc, then title asc | Sidebar | Tab strip |

The Today list **auto-populates** from the due date —
tasks with `list = .anytime` but a due date in the next
24h appear in Today. This matches the spec's §12.2
explicit rule.

**macOS layout.** `NavigationSplitView` with three
columns:

- Sidebar: the five lists with their active-task counts.
- Middle: the tasks in the selected list, with the NLU
  input bar at the top.
- Detail: the selected task's metadata (notes, due date,
  priority, linked entities, receipt chain).

**iOS layout.** `NavigationStack` with a horizontal
scrollable tab strip at the top, the tasks in a `List`,
and a navigation push to the task detail. The NLU input
is a sheet.

**Receipt integration.** Every list mutation (move,
complete, reopen, set priority, set due date) appends a
receipt with a `receipt_type` from
`ProductivityTaskReceiptType`. The receipt drawer (Phase
3) shows the chain when the user taps a task.

---

## 5. Natural language input

Things 3-style natural language input parses input like:

- "tomorrow at 3pm, call John about the contract" →
  `dueAt = tomorrow 3pm`, `title = "Call John about the
  contract"`, linked to the contact "John".
- "buy milk" → Anytime, no due date.
- "high priority: review the Q3 report" → priority =
  `.high`, `title = "Review the Q3 report"`, linked to
  the document "Q3 report".
- "every monday, take out the trash" → (v2) recurrence;
  v1 just creates one task.

The parser is a Swift struct
`ProductivityTaskNLUParser(contacts:documents:now:)`. The
contact and document lookups are synchronous
(`ContactsAdapter`, `DocumentStoreNLU` protocols) — the
chat panel integration wraps the async data layer in an
actor that maintains an in-memory cache.

**Patterns recognised:**

| Fragment | Effect |
|---|---|
| `high priority:`, `medium priority:`, `low priority:`, `urgent:` | priority |
| `!` suffix | priority = .high (Things 3 convention) |
| `today`, `tonight`, `tomorrow` | due date on that day |
| `at 3pm`, `at 14:30`, `at noon`, `at midnight` | time on the resolved date |
| `next monday` / `next tuesday` / ... | next weekday instance |
| `in N days`, `in N weeks`, `in a day`, `in a week` | relative offset |
| `on Jan 15`, `on January 15`, `on 1/15` | absolute date |
| bare noun (matches a contact's name) | link to that contact |
| bare noun (matches a document's title) | link to that document |
| `someday` (as a keyword in the input) | routes to Someday list |

The parser is **lenient**: ambiguous input falls back to
Anytime / no due date / normal priority. The user can
always edit the parsed values in the triage UI.

**Why rule-based v1, not LLM:** the patterns are
well-known and an LLM call would be overkill. v1 is
deterministic and runs in <1ms; the LLM path is a v2
that can be added behind the same parser interface
without changing the rest of the system.

---

## 6. Chat panel integration

The chat panel (Phase 3) emits `ChatQueueItem` values
when the user types "add a task to …". The bridge is
`ProductivityTaskChatPanelBridge`:

```swift
let task = try await bridge.createTaskFromChat(
    chatItemID: chatItem.id,
    documentID: documentID,
    message: chatItem.message,
    parser: parser
)
```

The bridge:

1. Runs the message through the NLU parser.
2. Persists the task via `ProductivityTaskStore.upsert`.
3. Emits a `task_created_from_chat` receipt carrying the
   chat item id and document id, so the audit trail can
   answer "which chat command created this task".

The chat panel's receipt chip ("Task created: Review the
Q3 report") links to the new task; tapping the chip
navigates to the Tasks surface.

---

## 7. Receipt model

Every task mutation is a constitutional receipt:

| `receipt_type` | When |
|---|---|
| `task_upsert` | A task was created or updated (the upsert path) |
| `task_delete` | A task was deleted |
| `task_completed` | A task was marked completed |
| `task_reopened` | A completed task was reopened |
| `task_moved` | A task was moved between lists |
| `task_priority_changed` | A task's priority was set |
| `task_due_date_changed` | A task's due date was set or cleared |
| `task_link_created` | A task was linked to another graph entity |
| `task_link_deleted` | A task's link to another graph entity was removed |
| `task_created_from_chat` | A task was created from a chat panel queue item |
| `task_created_from_nlu` | A task was created by the NLU parser |

The receipt payload includes:

- `list`, `priority`, `hasDueAt`, `isCompleted` for the
  upsert path.
- `fromList`, `toList` for the moved path.
- `fromPriority`, `toPriority` for the priority path.
- `chatItemID`, `documentID`, `rawMessage` for the
  chat-provenance path.

**Receipt chain.** The task's full history is the
sequence of receipts for the task's `entity_id`,
ordered by `witnessed_at`. The receipt drawer (Phase 3)
shows this chain when the user opens a task.

**Voided tasks.** When a task is deleted, the `graph_entity`
row is removed but the `graph_receipts` rows are
preserved (foreign keys cascade). The user can recover a
voided task by re-creating it with the same `id` (a
future worker will add a "Recover" gesture to the receipt
drawer; v1 only exposes the receipts).

---

## 8. Inbox triage

When a task is in the Inbox (newly created, no due date,
not triaged), the user can:

- **Set a due date** via a context menu (drag-to-day
  calendar gesture is a v2).
- **Move to a list** (Inbox → Today / Upcoming / Anytime /
  Someday).
- **Delete** (a `task_delete` receipt is appended; the
  receipts are preserved).
- **Open in the editor** (currently disabled; the editor
  surface's "open in editor" gesture is Phase 2's
  `TesseraTextContentManager` and is wired in a
  follow-up).

Every triage action calls the typed API
(`store.move`, `store.setPriority`, etc.) so the receipt
chain is intact.

---

## 9. Cross-surface links

Tasks can be linked to:

- **Documents** (the "review Q3 report" task → the Q3
  report document).
- **Contacts** ("call John" task → the John contact).
- **Calendar events** ("prep for the 3pm standup" task →
  the meeting event).
- **Other tasks** (parent / subtask relationships).

The linking is via `entity_links` (Phase 1's data layer).
The Tasks surface shows the linked entities in the task
detail view, with quick links to open them in their
native surface.

`ProductivityTaskStore.linkTask(_:to:linkType:weight:actor:)`
and `.unlinkTask(_:from:linkType:actor:)` are the typed
entry points; both append `task_link_created` /
`task_link_deleted` receipts.

---

## 10. Graph view integration (Phase 6)

Tasks appear in the graph view alongside other materials.
The graph view's `GraphNode` type already maps
`entity_type = "task"` to the green icon (see
`Sources/TesseraCore/Productivity/Graph/GraphModel.swift`'s
`color(for:)` and `iconName(for:)`).

`ProductivityTaskGraphIntegration` is the Tasks-specific
adapter that turns a `ProductivityTask` into a
`GraphNode`. The graph view calls
`integration.loadAllNodes(limit:)` when the user toggles
the "tasks" filter in the sidebar.

Clicking a task node opens the task in the Tasks surface
(double-click is the gesture).

---

## 11. Library survey

| Need | Library | Decision |
|---|---|---|
| SwiftUI list UI | `SwiftUI` `List`, `NavigationSplitView` | Adopt |
| Natural language date parser | Custom (rule-based) | Build — patterns are well-known, no LLM needed for v1 |
| Recurrence rules (v2) | None | Defer |
| Reminder notifications (v2) | `UserNotifications` framework | Defer |

---

## 12. Test strategy

**Unit tests (`ProductivityTaskTests`):**

- JSON round-trip for the full entity (with all fields
  set).
- Each `List` case serialises distinctly.
- Each `Priority` case serialises distinctly.
- Priority is `Comparable` (sort by rank).
- Due-date helpers (`isOverdue`, `isDueWithin24Hours`,
  `isDueWithin7DaysButNotToday`).
- List auto-classification.
- Display names + system image names.
- Notes preview truncation.

**NLU parser tests (`ProductivityTaskNLUParserTests`):**

- Simple inputs.
- All priority prefixes (`high priority:`, `medium
  priority:`, `low priority:`, `urgent:`, `!`).
- All relative-date prefixes (`today`, `tonight`,
  `tomorrow`, `next monday`, `in N days`, `in N weeks`,
  `in a day`, `in a week`, `on Jan 15`).
- All time suffixes (`at 3pm`, `at noon`).
- List inference.
- Ambiguous input falls back to Anytime / no due date /
  normal priority.

**Linking tests (`ProductivityTaskNLULinkingTests`):**

- Contact linking by name.
- Document linking by title.
- Quoted name linking.
- Case-insensitive contact matching.
- No link when name doesn't match.

**Filter tests (`ProductivityTaskFilterTests`):**

- Inbox / Today / Upcoming / Anytime / Someday filter
  semantics.
- Overdue tasks belong in Today.
- Sort by due date asc for Today / Upcoming.
- Sort by priority desc + title asc for Inbox / Anytime
  / Someday.

**Store tests (`ProductivityTaskStoreTests`):**

- Receipt type strings are pinned (no schema migration).
- `entity_type = "task"` is pinned.
- `subtypeString` matches the list.
- Error equality.
- `isCompleted` semantics.
- JSON helpers round-trip.

**Integration tests (`ProductivityTaskStoreIntegrationTests`):**

- env-gated on `TESSERA_DB_INTEGRATION=1`.
- End-to-end upsert → receipt → fetch → delete.
- List returns inserted task.
- Move records receipt.
- Complete / reopen flow.

---

## 13. Out of scope (v2+)

- **Recurrence rules** (`every monday, take out the trash`)
  — the parser doesn't extract them in v1; v2 adds a
  `recurrence` field on `ProductivityTask` and the parser
  populates it.
- **Reminder notifications** via `UserNotifications` — v1
  just records the due date; v2 schedules a notification
  at `dueAt - leadTime`.
- **Apple Reminders import** — opt-in path via EventKit;
  the importer maps Reminders' `dueDateComponents` to
  `ProductivityTask.dueAt`.
- **Google Tasks import** — opt-in via Google Tasks API
  + OAuth; the importer maps the Tasks list to a
  `ProductivityTask.list` value.
- **Drag-to-day calendar gesture** in the Inbox triage
  UI — the v1 context menu exposes the same actions.
- **v2 NLU via LLM** — the v1 rule-based parser is
  deterministic and fast; v2 can add an LLM fallback for
  ambiguous inputs.
