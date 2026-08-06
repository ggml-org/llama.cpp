# Calendar Material Surface - Worker Report

**Branch:** `feat/prod-materials-calendar`
**Base:** `main` @ `2a6da6cdf`
**Worktree:** `worktrees/prod-materials-calendar/`
**Author:** Mavis (general-purpose worker)

---

## Summary

The §12.4 Calendar material surface is implemented: events
live as `graph_entities` (`entity_type = 'calendar_event'`)
with full receipt chains, RFC 5545-style recurrence, a
Fantastical-style natural-language quick-add parser, a chat
handler with create/list/move/respond/delete intents, shared
macOS/iOS SwiftUI day/week/month views, and graph-view
open-in-calendar deep links. Migration `0006_calendar.sql`
adds the time-range index. 120 new calendar tests, 0
failures (11 self-skip: env-gated Postgres integration). No
push, no PR (per AGENTS.md).

```
            natural language                 chat message
            "lunch with Ada tomorrow"        "move standup to 3pm"
                     |                              |
                     v                              v
        +---------------------+        +------------------------+
        | CalendarNLUParser   |        | CalendarChatHandler    |
        | (NSDataDetector +   |------->| (actor queue: create/  |
        |  resolver protocol) |        |  list/move/respond/    |
        +---------------------+        |  delete)               |
                     |                 +------------------------+
                     v                              |
        +-------------------------------------------+
        | CalendarStore : CalendarStoring           |
        |  graph_entities rows + graph_receipts     |
        |  links: attendee_of / prep_document /     |
        |         prep_task / reminder_for          |
        +-------------------------------------------+
                     |                 |
                     v                 v
        +---------------------+  +---------------------+
        | CalendarViewModel   |  | GraphViewModel      |
        | day/week/month grid |  | openEntityHandler   |
        | (shared SwiftUI)    |<-| "Open" button       |
        +---------------------+  +---------------------+
```

---

## Files touched

### New: `Sources/TesseraCore/Productivity/Materials/Calendar/` (8 files, 2,774 LoC)

- `CalendarEvent.swift` (273) - Codable event model; JSON body stored in `graph_entities.body`, title in `label`
- `RecurrenceRule.swift` (422) - RRULE subset (FREQ/INTERVAL/COUNT/UNTIL/BYDAY), parse/serialize/expand, DST-safe stepping
- `CalendarNLUParser.swift` (753) - NSDataDetector-grounded quick-add parser with span-exclusion + title assembly
- `CalendarResolvers.swift` (128) - `ContactsAdapter` / `DocumentResolver` / `LocationResolver` protocols + snapshot/geocoding production adapters
- `CalendarStore.swift` (377) - data-layer CRUD, receipts, typed links, in-memory range/occurrence expansion
- `CalendarChatHandler.swift` (473) - actor queue; intent classification + execution against any `CalendarStoring`
- `CalendarViewModel.swift` (313) - mode switching (anchor date preserved), visible-range expansion, quick-add submission
- `CalendarGraphConnector.swift` (35) - wires `GraphViewModel.openEntityHandler` to calendar selection/jump

### New: `Sources/TesseraCore/Productivity/Materials/Calendar/Views/` (2 files, 860 LoC)

- `CalendarGridViews.swift` (382) - day/week hour grid + month grid
- `CalendarEventDetailView.swift` (478) - editable detail, recurrence summary, links, receipt history

### New: `Sources/TesseraStudioMac/Views/Calendar/` (1 file, 277 LoC)

- `CalendarSurfaceView.swift` (277) - macOS wrapper: mode switcher, quick-add field, nav, detail pane

### New: `Tests/TesseraCoreTests/Productivity/Materials/Calendar/` (8 files, 2,055 LoC, 120 tests)

- `CalendarEventTests.swift` (240), `RecurrenceRuleTests.swift` (295), `CalendarNLUParserTests.swift` (272), `CalendarChatHandlerTests.swift` (266), `CalendarViewModelTests.swift` (360), `CalendarStoreTests.swift` (50), `CalendarStoreIntegrationTests.swift` (424, env-gated), `CalendarTestSupport.swift` (148, `InMemoryCalendarStore` fake + fixtures)

### Modified (existing files)

- `Sources/TesseraCore/Productivity/Graph/GraphViewModel.swift` - additive `openEntityHandler` + `open(_:)`; existing behavior unchanged
- `Sources/TesseraCore/Productivity/Graph/GraphView.swift` - "Open" button in the entity detail panel when a handler is registered

### New docs + migration

- `docs/tessera-productivity-materials-calendar-design.md`
- `tools/tessera/db/migrations/0006_calendar.sql` (23 LoC) - partial index `idx_entities_event_start` on `(entity_type, body->>'startAt') WHERE entity_type = 'calendar_event'`

---

## Test results

- Calendar suite: **120 tests, 0 failures, 11 skipped**
  (`swift test --filter "Calendar|RecurrenceRule"`). The 11
  skips are `CalendarStoreIntegrationTests`, gated on
  `TESSERA_DB_INTEGRATION=1` with a provisioned Postgres -
  same convention as `ContactStoreIntegrationTests`.
- Full package suite: ran to completion. The pre-existing
  `ExportFormatTests.testSlackMrkdwnBold` /
  `testSlackMrkdwnRoundTrip` failures (Slack mrkdwn
  formatting) and a mid-suite SIGTRAP in
  `TesseraImporterEventParsingTests.testMalformedUUIDIsSkipped`
  (Phase 4 import-export code, unchanged on this branch)
  are unrelated to the calendar surface and unchanged from
  the pre-change baseline. No new failures introduced.
- Integration suite was attempted locally with the env var
  set: local Postgres is up but has no `tessera` role
  (SQLState 28000). Provisioning infra is out of scope; the
  tests self-skip by design.

## Design decisions

1. **Events are graph entities, not a new table.** Reuses
   receipts, links, provenance, and the Grape visualization.
   The 0006 index prepares a future server-side range query
   without changing schema now.
2. **`CalendarStoring` protocol seam.** Chat + view-model
   tests run against `InMemoryCalendarStore`; only the
   integration suite touches Postgres.
3. **Synchronous NLU with injected resolvers.** Parser is
   pure and testable; production loads contact/document
   snapshots async and hands them in. No LLM dependency.
4. **Client-side recurrence expansion.** Range queries load
   the calendar's entities and expand in memory - correct,
   simple, and fine at personal-calendar scale; the index is
   the upgrade path.
5. **Own actor queue for calendar chat.** Calendar intents
   are not document-scoped, so they don't reuse
   `ChatQueueItem`.

---

## Things I punted on (and why)

- **EventKit / CalDAV / Google Calendar import** - §12.4
  does not require it; needs entitlement + auth decisions.
- **Server-side JSONB range queries** - data layer has no
  such query today; index is in place for when it does.
- **Push/notification reminders** - `reminder_for` link type
  exists; notification plumbing is a separate surface.
- **Conflict detection / scheduling** - not in §12.4.
- **Local integration DB provisioning** - infra, not code;
  tests are env-gated and self-skip.

## How to verify

```sh
cd worktrees/prod-materials-calendar/TesseraStudio
swift test --filter "Calendar|RecurrenceRule"     # 120 tests, 11 skipped

# With a provisioned Postgres (role: tessera):
TESSERA_DB_INTEGRATION=1 swift test --filter CalendarStoreIntegrationTests
```

Design rationale: `docs/tessera-productivity-materials-calendar-design.md`.
