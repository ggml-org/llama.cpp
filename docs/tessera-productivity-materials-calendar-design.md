# Tessera Studio - Calendar material surface

Phase: Productivity Materials (Calendar). Status: implemented
on `feat/prod-materials-calendar`.
Sources: `docs/tessera-productivity-design.md` §12.4, plus
the receipt/graph patterns established in the Contacts +
Graph phase.

---

## 1. Problem

§12.4 asks for a Fantastical-style calendar inside Tessera:

- Natural language quick-add ("lunch with Ada tomorrow at
  noon, every friday").
- Events are `graph_entity` rows with
  `entity_type = 'calendar_event'`, so they participate in
  the same receipt chain and graph as people, documents, and
  tasks.
- Events link to attendees (person entities), prep documents
  (document entities), and prep tasks (task entities).

The calendar is a material surface: it must read and write
through the constitutional data layer, produce receipts for
every mutation, and expose graph links - not behave like a
detached EventKit clone.

## 2. Why this design

- **Events as graph entities, not a new table family.** The
  data layer already persists `graph_entities` with typed
  links and receipts. Storing the event payload as JSON in
  `body` reuses that machinery end to end: receipts,
  provenance, links, deletion semantics, and the Grape
  visualization all apply for free.
- **`CalendarStoring` protocol seam.** Chat handler and
  view-model tests run against an in-memory fake; Postgres
  is only needed by the env-gated integration suite. This
  mirrors `ContactStore` / `DocumentStore` testability.
- **Synchronous rule-based NLU with injected resolvers.**
  `NSDataDetector` grounds date/time spans against the real
  clock; person/document/location references resolve through
  small protocols (`ContactsAdapter`, `DocumentResolver`,
  `LocationResolver`). Production wiring loads snapshots
  async and hands them to the pure parser. No LLM round
  trip: quick-add must work offline and deterministically.
- **RFC 5545-style RRULE subset, expanded client-side.**
  Recurrence is stored as a compact rule on the event and
  expanded against a visible range in memory. Server-side
  expansion is a future optimization; the migration already
  adds a partial index on `(entity_type, body->>'startAt')`
  for that
  path.
- **Own actor queue for calendar chat.** Calendar requests
  are not document-scoped, so `CalendarChatHandler` runs its
  own actor instead of reusing `ChatQueueItem`.

## 3. CalendarEvent model

`CalendarEvent` is a `Codable` value type stored JSON-encoded
in `graph_entities.body` (`label` holds the title, so list
queries and the graph can render events without decoding):

- identity: `id`, `calendar` (default `"default"`), `title`,
  `notes`, `location`
- time: `start`, `end`, `allDay`, `timezoneIdentifier`
- recurrence: optional `RecurrenceRule`
- links are materialized as typed graph edges, not fields:
  attendees -> `attendee_of`, prep docs -> `prep_document`,
  prep tasks -> `prep_task`, reminders -> `reminder_for`

Every mutation through `CalendarStore` appends a
`graph_receipts` row:

| action            | receipt kind          |
|-------------------|-----------------------|
| create            | `event_created`       |
| update            | `event_updated`       |
| delete            | `event_deleted`       |
| chat response     | `event_responded`     |
| link creation     | `event_link_created`  |

Receipt payloads carry the event id plus a JSON snapshot of
the event, matching the pre-mutation-snapshot convention from
the foundations phase.

## 4. Recurrence (RRULE subset)

`RecurrenceRule` supports the parts of RFC 5545 that matter
for a personal calendar:

- `FREQ`: `DAILY`, `WEEKLY`, `MONTHLY`, `YEARLY`
- `INTERVAL`, `COUNT`, `UNTIL`
- `BYDAY` for weekly rules (e.g. `MO,WE,FR`)

Parse (`RecurrenceRule(rrule:)`) and serialization
(`rruleString`) round-trip; malformed rules throw typed
`RecurrenceRuleError`s. Expansion
(`occurrences(of:in:calendar:)`) walks the rule inside a
query range, preserves the anchor's wall-clock time across
DST transitions (components-based stepping, not
`addingTimeInterval`), and stops at `COUNT`/`UNTIL`.

## 5. Natural-language quick-add parser

`CalendarNLUParser.parse(_:resolvers:referenceDate:)` turns a
phrase into a `ParsedCalendarIntent`:

1. `NSDataDetector` extracts date/time spans grounded in the
   real clock (relative phrases like "tomorrow at 3pm"
   resolve against now).
2. Recurrence phrases ("every monday", "weekly on ...") are
   matched by rule grammar first; their span is recorded so
   the date pass does not re-consume it - except where a
   candidate time-of-day span (e.g. "at 9am") merely
   overlaps the rule span, in which case the time survives
   and feeds the event window.
3. Remaining spans form the start window; default duration is
   1 hour, default start is the next round hour when no time
   is given.
4. People/document/location references resolve through the
   injected resolver protocols; unresolved text stays in the
   title.
5. Title assembly strips only the consumed spans (overlapping
   consumed spans are merged), preserving the user's wording
   otherwise.

Production adapters: `ContactSnapshotAdapter` (contacts
loaded once from the store), `DocumentSnapshotAdapter`
(document index snapshot), `GeocodingLocationResolver`
(CLGeocoder with an in-memory cache). All three are trivially
replaceable in tests.

## 6. CalendarStore

`CalendarStore` wraps `TesseraDataLayer`:

- `upsert(event)` -> `graph_entities` row
  (`entity_type = 'calendar_event'`), receipt appended
- `event(id:)`, `events(in:calendar:)` - range queries load
  the calendar's entities and filter/expand in memory
  (the data layer has no JSONB range query yet; the 0006
  partial index prepares the future server-side path)
- `link(event:to:type:)` -> typed graph edge +
  `event_link_created` receipt
- `delete(id:)` -> entity removal + receipt
- `respond(to:with:)` -> chat-response receipt path

`extension CalendarStore: CalendarStoring {}` is the seam the
fake (`InMemoryCalendarStore`) shares with tests.

## 7. Chat handler

`CalendarChatHandler` (actor) classifies calendar intents -
create, list, move, respond, delete - and executes them
against any `CalendarStoring`. Create intents route through
the NLU parser first. Each handled message returns a
user-facing summary string and leaves a receipt in the
store. The queue drains serially so "create X" followed by
"move X to 3pm" stays ordered.

## 8. Calendar surface (SwiftUI)

Shared cross-platform views live in TesseraCore so macOS and
a future iOS target reuse them:

- `CalendarSurfaceView` - mode switcher (day / week / month),
  quick-add field, navigation, detail pane
- `CalendarGridViews` - day/week hour grid and month grid;
  recurring occurrences are expanded for the visible range
- `CalendarEventDetailView` - editable title/time/location/
  notes, recurrence summary, attendee/prep links, receipt
  history

`CalendarViewModel` owns the visible range, selection, mode
switching (the anchor date survives mode changes), and
quick-add submission. It depends only on `CalendarStoring`.

## 9. Graph integration

Additive, non-breaking:

- `GraphViewModel` gained an optional `openEntityHandler`
  closure plus `open(_:)`; existing callers are unaffected.
- `GraphView`'s detail panel shows an "Open" button when a
  handler is registered.
- `CalendarGraphConnector.wire(graph:to:)` installs a
  handler that routes `calendar_event` entities into the
  calendar surface (selection + jump to the event's date).

## 10. Migration

`tools/tessera/db/migrations/0006_calendar.sql` adds the
partial index supporting future server-side time-range
queries over calendar entities:

```sql
CREATE INDEX IF NOT EXISTS idx_entities_event_start
    ON graph_entities (entity_type, body->>'startAt')
    WHERE entity_type = 'calendar_event';
```

Follows the 0001 conventions: `IF NOT EXISTS`, no
transaction wrapper, idempotent re-apply. No schema changes
to `graph_entities` itself - events reuse the existing
table, which is the point of the design.

## 11. Test strategy

- Pure unit tests: model + recurrence round-trips/expansion,
  NLU parser (spec example phrases, recurrence, defaults,
  doc linking), chat handler lifecycle + classification,
  view-model behavior, receipt/link taxonomy pins.
- `CalendarTestSupport` provides `InMemoryCalendarStore`
  (with failure injection) and shared fixtures.
- NLU tests with relative dates compute expectations from
  `Date()`; date-free phrases use a pinned reference date,
  so nothing is clock-flaky.
- `CalendarStoreIntegrationTests` is env-gated on
  `TESSERA_DB_INTEGRATION=1` (CRUD, receipt chains, links,
  range queries, migration index presence, graph open
  end-to-end) and self-skips otherwise.

Result: **120 calendar tests, 0 failures** (11 skipped =
the env-gated integration suite without a provisioned
database).

## 12. Out of scope (v2+)

- EventKit / CalDAV / Google Calendar import + sync
- Server-side JSONB range queries (index is ready)
- Busy/free scheduling, conflicts, invites
- Reminders as push notifications (link type exists, no
  notification plumbing)
- Timezone-aware recurring expansion across travel
  (wall-clock semantics for now)

## How to use

```swift
let dataLayer = TesseraDataLayer(...)
let calendarStore = CalendarStore(dataLayer: dataLayer)
let graph = GraphViewModel(store: GraphStore(dataLayer: dataLayer))

// Wire graph -> calendar deep links.
CalendarGraphConnector.wire(graph: graph, to: calendarViewModel)

// Quick-add from natural language.
let parser = CalendarNLUParser()
let intent = try parser.parse(
    "lunch with Ada tomorrow at noon",
    resolvers: resolvers
)
let event = CalendarEvent(from: intent)
_ = try await calendarStore.upsert(event)

// Range query expands recurrences.
let week = try await calendarStore.events(in: weekRange)

// Open the surface (macOS).
CalendarSurfaceView(viewModel: calendarViewModel)
```

## File index

```
Sources/TesseraCore/Productivity/Materials/Calendar/
  CalendarEvent.swift              (273 LoC)
  RecurrenceRule.swift               (422 LoC)
  CalendarNLUParser.swift          (753 LoC)
  CalendarResolvers.swift          (128 LoC)
  CalendarStore.swift              (377 LoC)
  CalendarChatHandler.swift        (473 LoC)
  CalendarViewModel.swift          (313 LoC)
  CalendarGraphConnector.swift     (35 LoC)
Sources/TesseraCore/Productivity/Materials/Calendar/Views/
  CalendarGridViews.swift          (382 LoC)
  CalendarEventDetailView.swift    (478 LoC)
Sources/TesseraStudioMac/Views/Calendar/
  CalendarSurfaceView.swift        (277 LoC)
Tests/TesseraCoreTests/Productivity/Materials/Calendar/
  CalendarEventTests.swift         (240 LoC)
  RecurrenceRuleTests.swift        (295 LoC)
  CalendarNLUParserTests.swift     (272 LoC)
  CalendarChatHandlerTests.swift   (266 LoC)
  CalendarViewModelTests.swift     (360 LoC)
  CalendarStoreTests.swift         (50 LoC)
  CalendarStoreIntegrationTests.swift (424 LoC)
  CalendarTestSupport.swift        (148 LoC)
tools/tessera/db/migrations/
  0006_calendar.sql                (23 LoC)
Modified:
  Sources/TesseraCore/Productivity/Graph/GraphViewModel.swift
  Sources/TesseraCore/Productivity/Graph/GraphView.swift
docs/tessera-productivity-materials-calendar-design.md (this file)
```

Total new lines (code + tests + migration): ~5,990 LoC.
