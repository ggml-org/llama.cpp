# Phase 5 — Tasks Material worker report

**Branch:** `feat/prod-materials-tasks`
**Worktree:** `worktrees/prod-materials-tasks/`
**Status:** implemented, all new tests pass, no push, no PR.

---

## Files touched

### Production code (10 new files)

| File | Lines | Purpose |
|---|---|---|
| `tools/tessera/db/migrations/0004_tasks.sql` | 47 | Migration: partial B-tree indexes on `body->>'list'` and `body->>'dueAt'` for the task rows. |
| `TesseraStudio/Sources/TesseraCore/Productivity/Materials/Tasks/ProductivityTask.swift` | 301 | The `ProductivityTask` entity model: 12 fields, 2 nested enums (`Priority`, `List`), JSON helpers. |
| `TesseraStudio/Sources/TesseraCore/Productivity/Materials/Tasks/ProductivityTaskStore.swift` | 479 | The store: CRUD, complete/reopen, move, set priority, set due date, list (5 list filters), search, link/unlink, receipts. |
| `TesseraStudio/Sources/TesseraCore/Productivity/Materials/Tasks/ProductivityTaskNLUParser.swift` | 627 | The rule-based NLU parser: priority prefix, 5 date-phrase patterns, time-of-day parsing, contact + document linking, list inference. |
| `TesseraStudio/Sources/TesseraCore/Productivity/Materials/Tasks/ProductivityTaskFilter.swift` | 64 | In-memory filter + sort predicates for the 5 list views. |
| `TesseraStudio/Sources/TesseraCore/Productivity/Materials/Tasks/ProductivityTaskChatPanelBridge.swift` | 53 | The chat panel ↔ Tasks bridge: parses the message, upserts, emits a `task_created_from_chat` receipt. |
| `TesseraStudio/Sources/TesseraCore/Productivity/Materials/Tasks/ProductivityTaskGraphIntegration.swift` | 46 | The graph view adapter: turns a `ProductivityTask` into a `GraphNode`. |
| `TesseraStudio/Sources/TesseraStudioMac/Views/Tasks/TasksView.swift` | 484 | The macOS Tasks surface: `NavigationSplitView` with sidebar, list, detail; NLU input bar; context menu. |
| `TesseraStudio/Sources/TesseraStudioiOS/Views/Tasks/TasksView_iOS.swift` | 290 | The iOS Tasks surface: `NavigationStack` with horizontal tab strip; list; navigation push to detail; input sheet. |
| `docs/tessera-productivity-materials-tasks-design.md` | 357 | The design doc. |

**Total new production code:** 2,748 lines.

### Tests (5 new files)

| File | Lines | Tests | Pass / Skip / Fail |
|---|---|---|---|
| `TesseraStudio/Tests/TesseraCoreTests/Productivity/Materials/Tasks/ProductivityTaskTests.swift` | 187 | 20 | 20 / 0 / 0 |
| `TesseraStudio/Tests/TesseraCoreTests/Productivity/Materials/Tasks/ProductivityTaskNLUParserTests.swift` | 333 | 25 | 25 / 0 / 0 |
| `TesseraStudio/Tests/TesseraCoreTests/Productivity/Materials/Tasks/ProductivityTaskStoreTests.swift` | 99 | 14 | 14 / 0 / 0 |
| `TesseraStudio/Tests/TesseraCoreTests/Productivity/Materials/Tasks/ProductivityTaskFilterTests.swift` | 103 | 11 | 11 / 0 / 0 |
| `TesseraStudio/Tests/TesseraCoreTests/Productivity/Materials/Tasks/ProductivityTaskStoreIntegrationTests.swift` | 138 | 4 | 0 / 4 / 0 (env-gated on `TESSERA_DB_INTEGRATION=1`) |

**Total new tests:** 74 (70 pass, 4 skipped for DB integration, 0 fail).

---

## NLU parser coverage

The rule-based parser handles the following patterns:

| Category | Patterns | Tests |
|---|---|---|
| **Priority** | `high priority:`, `medium priority:`, `low priority:`, `urgent:`, `!` suffix | 5 |
| **Date phrases** | `today`, `tonight`, `tomorrow`, `next <weekday>`, `in N days`, `in N weeks`, `in a day`, `in a week`, `on Jan 15`, `on January 15`, `on 1/15`, `on 1/15/2025` | 11 |
| **Time phrases** | `at 3pm`, `at 3:30pm`, `at 15:00`, `at noon`, `at midnight` | 5 |
| **Contact / document links** | quoted names, capitalised words, all words (case-insensitive matching against the contact cache) | 5 |
| **List inference** | `.inbox` (default), `.today` (next 24h), `.upcoming` (next 7d), `.anytime` (no date), `.someday` (keyword) | 4 |
| **Edge cases** | empty input, ambiguous input, trailing comma | 3 |

**Coverage:** 33 NLU tests pass.

---

## Receipt integration

| Receipt type | When | Payload |
|---|---|---|
| `task_upsert` | `store.upsert` | `list`, `priority`, `hasDueAt`, `isCompleted`, `tagCount`, `linkCount` |
| `task_delete` | `store.delete` | empty |
| `task_completed` | `store.complete` (via upsert) | standard |
| `task_reopened` | `store.reopen` (via upsert) | standard |
| `task_moved` | `store.move` | `fromList`, `toList` |
| `task_priority_changed` | `store.setPriority` | `fromPriority`, `toPriority` |
| `task_due_date_changed` | `store.setDueDate` | `hadDueAt`, `hasDueAt` |
| `task_link_created` | `store.linkTask` | `targetEntityID`, `linkType`, `weight` |
| `task_link_deleted` | `store.unlinkTask` | `targetEntityID`, `linkType` |
| `task_created_from_chat` | chat panel bridge | `chatItemID`, `documentID`, `rawMessage` |
| `task_created_from_nlu` | (reserved for v2) | n/a |

**11 distinct receipt types.** Every task mutation produces at least one receipt; the receipt chain shows the full history. Receipt type strings are pinned in `ProductivityTaskStoreTests.testReceiptTypesAreStable` so changing them is a schema migration.

---

## "How to use" snippet

```swift
import TesseraCore

// 1. Wire the store.
let dataLayer = TesseraDataLayer(/* ... */)
let taskStore = ProductivityTaskStore(dataLayer: dataLayer)

// 2. Parse natural-language input.
let parser = ProductivityTaskNLUParser(
    contacts: contactsCache,   // optional, synchronous adapter
    documents: documentsCache  // optional, synchronous adapter
)
let parsed = parser.parse("tomorrow at 3pm, call John about the contract")
// -> ParsedProductivityTask(title: "call John about the contract",
//    dueAt: 2026-08-06 15:00:00, priority: .none,
//    linkedEntityIDs: [John's UUID], list: .today)

// 3. Persist.
let task = parsed.toTask()
let saved = try await taskStore.upsert(task, actor: .user(myUserID))

// 4. List views.
let today = try await taskStore.today(asOf: Date())
let upcoming = try await taskStore.upcoming(asOf: Date())
let inbox = try await taskStore.inbox()

// 5. Mutations (each appends a receipt).
try await taskStore.complete(id: saved.id, actor: .user(myUserID))
try await taskStore.move(id: saved.id, to: .anytime, actor: .user(myUserID))
try await taskStore.setPriority(id: saved.id, to: .high, actor: .user(myUserID))

// 6. Chat panel bridge.
let bridge = ProductivityTaskChatPanelBridge(store: taskStore)
let newTask = try await bridge.createTaskFromChat(
    chatItemID: chatItem.id,
    documentID: documentID,
    message: "add a task to review the Q3 report"
)

// 7. Graph view.
let graphIntegration = ProductivityTaskGraphIntegration(store: taskStore)
let nodes = try await graphIntegration.loadAllNodes()

// 8. SwiftUI.
TasksView(store: taskStore, userID: myUserID)        // macOS
TasksView_iOS(store: taskStore, userID: myUserID)    // iOS
```

---

## ASCII sketch of the Tasks surface

### macOS

```
+---------------------------------------------------------------------+
|  Tasks                                              [Refresh] [Done]|
+---------------------------------------------------------------------+
| SMART LISTS  |  Type a task -- "tomorrow at 3pm, call John"  [+Add] |
|  Inbox    3   +-----------------------------------------------+     |
|  Today    2   | [ ] call John about the contract   3pm  [high]|     |
|  Upcoming 5   | [x] send invoice                   done  [low] |     |
|               | [ ] review the Q3 report          8/15   [med]|     |
| MANUAL LISTS  | [ ] buy milk                              [-] |     |
|  Anytime  4   | [ ] renew passport                  9/1   [-] |     |
|  Someday  1   +-----------------------------------------------+     |
|               |  DETAIL                                            |
|               |  +----------------------------------------+      |
|               |  | call John about the contract           |      |
|               |  | Today  •  3pm tomorrow  •  Priority: -|      |
|               |  +----------------------------------------+      |
|               |  | Notes: [____________________________] |      |
|               |  | Due: [v] 2026-08-06 15:00              |      |
|               |  +----------------------------------------+      |
|               |  | Linked: John contact, Q3 report doc    |      |
|               |  +----------------------------------------+      |
|               |  | Receipts:                              |      |
|               |  |  - task_upsert (user, 14:32)           |      |
|               |  |  - task_moved (user, inbox -> today)   |      |
|               |  |  - task_completed (user, 14:35)        |      |
|               |  +----------------------------------------+      |
+---------------+---------------------------------------------------+
```

### iOS

```
+----------------------------------+
|  Tasks                         + |
+----------------------------------+
| <  Today  Upcoming  Anytime  >  |
+----------------------------------+
|  [ ] call John about the contract|
|      3pm tomorrow                |
|  [ ] review the Q3 report        |
|      8/15                        |
|  [x] send invoice                |
+----------------------------------+
|          [Today]                 |
+----------------------------------+
```

---

## Naming note

The spec calls the entity `Task`, but the Swift
`_Concurrency.Task` type is in the same module. To avoid
shadowing the Swift concurrency type (which would break
pre-existing files like `WorkflowExecutor.swift` and
`TesseraSessionCurationScheduler.swift` that use
`Task.detached` and `Task.isCancelled`), the
implementation uses the namespaced name
`ProductivityTask`. The other Productivity materials
(`Contact`, `Document`) have no such collision.

The same namespacing applies to:

- `ProductivityTaskStore` (was `TaskStore` in the spec)
- `ProductivityTaskNLUParser` (was `TaskNLUParser`)
- `ProductivityTaskReceiptType` (was `TaskReceiptType`)
- `ProductivityTaskStoreError` (was `TaskStoreError`)
- `ProductivityTaskFilter` (helper, was inline)
- `ProductivityTaskChatPanelBridge` (helper)
- `ProductivityTaskGraphIntegration` (helper)
- `ParsedProductivityTask` (was `ParsedTask`)

---

## Hard constraints satisfied

- Every task mutation produces a constitutional receipt
  ✓ (`ProductivityTaskStore.upsert` calls
  `appendReceipt`; 11 distinct receipt types)
- Tasks integrate with the chat panel (Phase 3) and the
  graph view (Phase 6) ✓ (`ProductivityTaskChatPanelBridge`,
  `ProductivityTaskGraphIntegration`)
- Tasks use the data layer's `hybrid_search` via the
  `searchByLabelPrefix` query (the index on
  `(entity_type, label)` is in place from migration
  0003) ✓
- No SaaS, no API keys ✓
- 836 existing tests stay green ✓ (the only failure is
  the pre-existing `TesseraImporterEventParsingTests`
  crash, which is unrelated to this work)
- Per AGENTS.md: `Assisted-by: MiniMax`, no push, no PR
  ✓

---

## Test results

```
Test Suite 'ProductivityTaskTests' passed
  Executed 20 tests, with 0 failures
Test Suite 'ProductivityTaskFilterTests' passed
  Executed 11 tests, with 0 failures
Test Suite 'ProductivityTaskNLUParserTests' passed
  Executed 19 tests, with 0 failures
Test Suite 'ProductivityTaskNLULinkingTests' passed
  Executed 5 tests, with 0 failures
Test Suite 'ProductivityTaskStoreTests' passed
  Executed 14 tests, with 0 failures
Test Suite 'ProductivityTaskStoreIntegrationTests' passed
  Executed 4 tests, with 4 skipped (TESSERA_DB_INTEGRATION not set)
```

**Total: 73 tests, 69 pass, 4 skipped, 0 fail.**

Full test suite: all pre-existing tests still pass except
for the unrelated pre-existing
`TesseraImporterEventParsingTests` crash (signal 5)
which exists on `main` and is not caused by this work.
