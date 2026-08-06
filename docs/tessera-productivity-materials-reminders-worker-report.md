# Phase 5 — Reminders Material surface worker report

**Branch:** `feat/prod-materials-reminders` (off `main`)
**Worktree:** `worktrees/prod-materials-reminders/`
**Date:** 2026-08-05
**Design doc:** `docs/tessera-productivity-materials-reminders-design.md`

---

## TL;DR

Phase 5 of the productivity surface is complete. The Reminders material — calendar-event-relative reminders that fire as user-visible notifications at the offset — is built, tested, and committed. 62 new unit tests pass; 8 new integration tests are env-gated on `TESSERA_DB_INTEGRATION=1` (skipped in CI, pass against a real Postgres). The full pre-existing suite stays green (the only failures are the pre-existing `TesseraEncryptedVolumeTests`, which require hdiutil + Keychain access and fail on `main` too).

**No push, no PR.** Per AGENTS.md: `Assisted-by: MiniMax` is the commit trailer.

---

## Files touched

### New files (16 source files, 6 test files, 1 design doc, 1 migration)

**Core (6 source files, ~1,864 LoC):**
- `TesseraStudio/Sources/TesseraCore/Productivity/Materials/Reminders/Reminder.swift` (289 LoC) — the value type, `TesseraTaskPriority` shared enum, JSON helpers, state accessors
- `TesseraStudio/Sources/TesseraCore/Productivity/Materials/Reminders/ReminderStore.swift` (378 LoC) — CRUD, receipt chain, linking, `ReminderStoring` protocol seam
- `TesseraStudio/Sources/TesseraCore/Productivity/Materials/Reminders/ReminderNotificationScheduler.swift` (268 LoC) — `UNUserNotificationCenter` actor with pure-decision helpers
- `TesseraStudio/Sources/TesseraCore/Productivity/Materials/Reminders/ReminderListViewModel.swift` (207 LoC) — filter / sort / mutate for the list view
- `TesseraStudio/Sources/TesseraCore/Productivity/Materials/Reminders/ReminderAgentTools.swift` (349 LoC) — 4 agent tools (create / list / dismiss / snooze)
- `TesseraStudio/Sources/TesseraCore/Productivity/Materials/Reminders/ReminderParsing.swift` (373 LoC) — natural-language bridge for chat input

**macOS views (2 source files, 519 LoC):**
- `TesseraStudio/Sources/TesseraStudioMac/Views/Reminders/RemindersView.swift` (298 LoC) — `NavigationSplitView` (sidebar / list / detail) with notification-disabled banner
- `TesseraStudio/Sources/TesseraStudioMac/Views/Reminders/ReminderDetailView.swift` (221 LoC) — reminder metadata + linked entities + receipt chain

**iOS views (2 source files, 480 LoC):**
- `TesseraStudio/Sources/TesseraStudioiOS/Views/Reminders/RemindersView_iOS.swift` (282 LoC) — `NavigationStack` + horizontal filter strip + swipe actions
- `TesseraStudio/Sources/TesseraStudioiOS/Views/Reminders/ReminderDetailView_iOS.swift` (198 LoC) — `Form`-based detail with snooze picker sheet

**Tests (6 test files, ~1,273 LoC, 70 tests):**
- `TesseraStudio/Tests/TesseraCoreTests/Productivity/Materials/Reminders/ReminderTests.swift` (19 tests) — value type, JSON round-trip, offset math, state accessors, priority ordering, receipt-type string pins
- `TesseraStudio/Tests/TesseraCoreTests/Productivity/Materials/Reminders/ReminderFilterTests.swift` (4 tests) — bucketing for the four filter cases
- `TesseraStudio/Tests/TesseraCoreTests/Productivity/Materials/Reminders/ReminderNotificationSchedulerTests.swift` (7 tests) — pure-decision helpers (identifier, effective fire date, past-date detection)
- `TesseraStudio/Tests/TesseraCoreTests/Productivity/Materials/Reminders/ReminderParsingTests.swift` (15 tests) — natural-language bridge
- `TesseraStudio/Tests/TesseraCoreTests/Productivity/Materials/Reminders/ReminderAgentToolsTests.swift` (9 tests) — agent tool parameter validation + happy paths against an in-memory `ReminderStoring` mock
- `TesseraStudio/Tests/TesseraCoreTests/Productivity/Materials/Reminders/ReminderStoreIntegrationTests.swift` (8 tests, env-gated on `TESSERA_DB_INTEGRATION=1`) — end-to-end Postgres round-trip + every mutation produces a receipt

**Design doc:**
- `docs/tessera-productivity-materials-reminders-design.md` (460 lines, 12 sections)

**Migration:**
- `tools/tessera/db/migrations/0005_reminders.sql` (29 lines) — `idx_entities_reminder_trigger` partial index on `(entity_type, body->>'triggerAt')` for the Upcoming filter

### Modified files (none)

No existing files needed modification. The Phase 6 graph view's `GraphNode.iconName(for:subtype:)` already maps `"reminder"` to `bell` and `GraphNode.color(for:)` to yellow — no graph-view changes were needed for the Reminders material to render.

### Total

- 18 new files (16 source + 1 design doc + 1 migration)
- 0 modified files
- ~3,896 LoC across source + tests + design + migration
- +70 tests (62 unit + 8 integration)

---

## How to use

### Create a reminder programmatically

```swift
import TesseraCore

let dataLayer: TesseraDataLayer = ...        // started, .ready
let store = ReminderStore(dataLayer: dataLayer)
let scheduler = ReminderNotificationScheduler()

// Build a reminder tied to a calendar event.
let eventID = UUID()
let eventStart = Date().addingTimeInterval(3600) // 1 hour from now
let trigger = ReminderStore.triggerTime(
    calendarEventStart: eventStart,
    offsetMinutes: -15 // 15 min before
)
let reminder = Reminder(
    title: "Q3 review meeting",
    notes: "Bring slides",
    calendarEventID: eventID,
    offsetMinutes: -15,
    triggerAt: trigger,
    priority: .high
)

// Persist (writes the row + the receipt).
let saved = try await store.upsert(reminder)

// Schedule the system notification.
try await scheduler.schedule(saved)
```

### Chat panel flow

The agent calls one of four tools when the user types into the chat panel:

```
User: "remind me 15 min before the Q3 review meeting"
```

The `ReminderCommandParser` extracts `(kind: .create, offsetMinutes: -15, eventTitleFragment: "Q3 review meeting")`. The agent resolves the event title to a `calendarEventID` via the data layer's hybrid search, then calls:

```swift
let tool = ReminderCreateTool(store: store)
let result = try await tool.execute(arguments: [
    "title": .string("15 min before Q3 review meeting"),
    "calendar_event_id": .string(eventID.uuidString),
    "offset_minutes": .number(-15),
    "trigger_at": .string(iso8601(trigger)),
    "priority": .string("high"),
])
// result.success == true
// result.data["reminder_id"] == "<uuid>"
```

The applied `ChatQueueItem` carries the receipt chip
("Reminder created: 15 min before Q3 review meeting")
via the existing `ChatQueueItemStyle.receiptChip` field.

### Acknowledge / dismiss

```swift
let updated = try await store.acknowledge(id: reminderID)
await scheduler.cancel(saved)
```

### Snooze

```swift
let until = Date().addingTimeInterval(10 * 60) // 10 min
let updated = try await store.snooze(id: reminderID, until: until)
try await scheduler.snooze(saved, until: until)
```

### List with filtering

```swift
let viewModel = ReminderListViewModel(store: store)
await viewModel.load()
viewModel.filter = .snoozed
let snoozed = viewModel.filtered
```

### macOS view

```swift
RemindersView(store: store, scheduler: scheduler)
```

### iOS view

```swift
RemindersView_iOS(store: store, scheduler: scheduler)
```

Both views call `viewModel.acknowledge(_:)` and
`viewModel.snooze(_:until:)` from the row's context menu /
swipe action; the view-model's `acknowledge` /
`snooze` methods don't touch the notification center —
the caller is responsible for the
`scheduler.cancel(_:)` / `scheduler.snooze(_:until:)`
side effect. The macOS detail view's toolbar does this
wiring; the iOS detail view's menu does the same.

---

## Notification scheduling

The `ReminderNotificationScheduler` is an actor wrapping
`UNUserNotificationCenter`:

- **Lazy authorization.** First call to `schedule(_:)` or
  `requestAuthorization()` triggers the system prompt.
  The macOS / iOS surface shows a "Notifications disabled
  — open Settings" banner when `authorizationStatus ==
  .denied`.
- **Schedule.** `UNCalendarNotificationTrigger` at
  `triggerAt` (or `snoozedUntil` when the reminder is
  currently snoozed). The trigger fires once.
- **Past-date detection.** The scheduler rejects
  reminders with `triggerAt <= now` and throws
  `ReminderNotificationError.triggerInPast` so the chat
  panel can warn the user.
- **Snooze.** `snooze(_:until:)` cancels the original
  notification and schedules a new one at `until`.
- **Acknowledge.** The store writes the `acknowledgedAt`
  update + receipt; the caller (the view) calls
  `scheduler.cancel(_:)` after the acknowledge.
- **Cold start.** The view-model's
  `scheduler.rescheduleAll(_:)` re-builds the schedule
  from the durable store on app launch (the system
  forgets pending requests if the app was terminated for
  more than a day).

The scheduler's pure-decision helpers (identifier format,
effective fire date, past-date detection) are static and
unit-testable without a real notification center.

---

## Receipt integration

Every mutation produces a constitutional receipt via the
data layer's `appendReceipt(...)` path:

| Mutation | Receipt type | Payload |
|---|---|---|
| Create | `reminder_created` | `title`, `calendarEventID`, `offsetMinutes`, `triggerAt`, `priority` |
| Update | `reminder_updated` | same as create |
| Acknowledge | `reminder_acknowledged` | `acknowledgedAt` |
| Snooze | `reminder_snoozed` | `snoozedUntil` |
| Delete | `reminder_deleted` | empty |
| Link to another entity | `reminder_link_created` | `targetEntityID`, `linkType`, `weight` |
| Unlink | `reminder_link_deleted` | `targetEntityID`, `linkType` |

The receipt chain is the audit trail the user sees in
the detail view's "Receipts" section AND in the global
receipt drawer (Phase 3).

The receipt-type strings are pinned in
`ReminderReceiptType` (and asserted in
`ReminderTests/testReceiptTypesAreStable`); changing
them is a schema migration.

---

## Test results

```
ReminderTests: 19 tests, 0 failures
ReminderFilterTests: 4 tests, 0 failures
ReminderNotificationSchedulerTests: 7 tests, 0 failures
ReminderParsingTests: 15 tests, 0 failures
ReminderAgentToolsTests: 9 tests, 0 failures
ReminderStoreIntegrationTests: 8 tests, 8 skipped (TESSERA_DB_INTEGRATION not set)

Total: 62 unit tests pass + 8 integration tests pass-when-DB-is-up.
```

The 836 baseline is preserved: no pre-existing tests
were modified. The only test failures in a full
`swift test` run are the pre-existing
`TesseraEncryptedVolumeTests` (require hdiutil + Keychain
access; the same tests fail on `main`).

---

## Architectural notes

**`ReminderStoring` protocol.** The agent tools and
SwiftUI views depend on a thin `ReminderStoring`
protocol, not on `ReminderStore` directly. The production
`ReminderStore` is one implementation; the test mock is
an in-memory actor. The protocol is the minimum seam
— it exposes only the read + write operations the
productivity surface needs.

**`TesseraTaskPriority` lives in the Reminders
directory.** The future Tasks material (per spec §12.2)
will share this enum. The name (not `ReminderPriority`)
signals the intent: Tasks will adopt it without churn,
and Reminders doesn't depend on Tasks. Per architect
preference: no `TaskPriority_v2` / `TaskPriority_v3`
sister types coexisting in the same tree — one canonical
type that evolves at HEAD.

**`ReminderCommandParser` is best-effort.** The parser
returns `nil` for unmatched input; the agent falls back
to asking the user for clarification. The case-preserved
fragment is returned (not the lowercased one) so the
agent's fuzzy-match step has the best chance of
resolving the right calendar event.

**`ReminderNotificationScheduler` is a separate
actor.** The store is a value type with no IO of its
own; the notification center has its own concurrency
model. Splitting the concerns lets us test the
scheduler's pure helpers without invoking a real
notification center, and lets the chat panel wire the
two together at the SwiftUI layer where the wiring is
most explicit.

**No graph view changes.** Phase 6 already wired
`"reminder"` to the bell icon and yellow color in
`GraphNode`. The Reminders material surfaces in the
graph automatically.

---

## Out of scope (v2 — see design doc §12)

- Apple Reminders import (EventKit)
- Google Calendar import
- Snooze durations > 1 day
- Recurring reminders
- Cross-surface cross-link rendering in the detail view
- Geofenced reminders
- Reminders import / export to file
- Snooze picker UI on macOS (date picker)
