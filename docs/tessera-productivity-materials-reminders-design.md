# Tessera Studio — Phase 5: Reminders Material surface

**Status:** implemented on `feat/prod-materials-reminders`
(off `main`).
**Sources read:** `docs/tessera-productivity-design.md`
§12.3 (Reminders) + §15 (Phase 5 deliverables).
**Branch:** `feat/prod-materials-reminders`. Worktree:
`worktrees/prod-materials-reminders/`. No push, no PR.

---

## 1. Problem

The productivity surface needs a Reminders material —
calendar-event-relative reminders ("15 min before the Q3
review meeting") that fire as user-visible notifications at
the offset.

Reminders are the first material that has a non-database
side effect: a scheduled `UNUserNotificationCenter`
request. The store owns the durable state (`graph_entity`
row + receipt chain); a separate actor owns the
notification-center side. The chat panel needs to create,
list, and dismiss reminders; the macOS / iOS surface shows
the list + detail; the graph view renders reminder nodes.

**Why this is a separate phase from the calendar surface:**
reminders always link to a calendar event but are
independent of the calendar CRUD. A user can have a
reminder for a meeting on the same day they add the event
or a year before — the link is a UUID, the offset is a
minute count, the trigger is a wall-clock date. We ship
Reminders without waiting for the calendar material; the
`calendarEventID` field is a UUID and the data layer's
foreign-key enforcement is optional (we use
`graph_entities` as a polymorphic table, so the constraint
is at the data-model layer, not the SQL layer).

## 2. Why this design

**Why a `graph_entity` row, not a dedicated reminders
table:** the universal "one row per thing" pattern is the
data layer's design choice (see
`docs/tessera-data-layer-design.md` §3). Splitting
reminders into a `reminders` table would force the agent's
hybrid-search to be a parallel UNION. Storing reminders in
`graph_entities` with `entity_type = 'reminder'` lets the
agent's "what's on my plate today?" call be a single
filter on the same table the documents / contacts / tasks
use.

**Why a thin `ReminderStoring` protocol:** the agent tools
and the SwiftUI view need to be unit-testable without
Postgres. The production `ReminderStore` wraps
`TesseraDataLayer`; the test mock implements the protocol
with an in-memory map. The protocol is the minimum seam —
it exposes only the read + write operations the
productivity surface needs, not the full data-layer
surface.

**Why `UNUserNotificationCenter` and not APNs / a
Tessera-side scheduler:** Apple ships the system
notification center; APNs is the upstream of that. A
Tessera-side scheduler that runs in our process would have
to be alive to fire the notification (the system already
schedules a wake-and-fire on our behalf via
`UNCalendarNotificationTrigger`). The system scheduler is
always-on, OS-managed, and respects the user's
notification preferences (Focus, Do Not Disturb, system
mute, per-app Settings).

**Why a separate scheduler actor, not a method on the
store:** the store is a value type with no IO of its own
(all IO is via the data layer actor). The notification
center has its own concurrency model. Splitting the
concerns lets us test the scheduler's pure helpers
(identifier format, effective fire date, past-date
detection) without invoking a real notification center,
and lets the chat panel wire the two together at the
SwiftUI layer where the wiring is most explicit.

**Why a `ReminderCommandParser` for chat input:** the chat
panel's "remind me 15 min before the Q3 review meeting"
is a free-text user input; the agent needs a structured
`(kind, offsetMinutes, eventTitleFragment, snoozeMinutes)`
to call the right tool. A small regex-based parser
extracts the fields; the agent then resolves
`eventTitleFragment` to a `calendarEventID` via the data
layer's hybrid search. The parser is best-effort — when
it returns `nil` the agent falls back to asking the user
for clarification (rather than guessing).

## 3. Reminder model

```swift
public struct Reminder: Codable, Sendable, Identifiable, Hashable {
    public let id: UUID
    public var title: String
    public var notes: String
    public var calendarEventID: UUID
    public var offsetMinutes: Int   // negative = before, positive = after
    public var triggerAt: Date
    public var acknowledgedAt: Date?
    public var snoozedUntil: Date?
    public var priority: TesseraTaskPriority
    public var createdAt: Date
    public var updatedAt: Date
}
```

Stored as a `graph_entity` row with `entity_type =
'reminder'`. The `body` is the JSON-encoded reminder;
`label` is the title (the graph view uses it for the node
caption).

**Why `triggerAt` is stored redundantly with
`calendarEventID + offsetMinutes`:** the list view's "sort
by fire time" and "upcoming within 24h" queries are
single-table reads; computing `triggerAt` on every row
would force a join back to the event. The store recomputes
`triggerAt` when the user edits the offset, so the
persisted value is always the source of truth.

**Why `TesseraTaskPriority` lives in the Reminders
directory:** the future Tasks material (per spec §12.2)
will share this enum. Putting it in the Reminders folder
keeps the dependency direction one-way: Tasks will adopt
`TesseraTaskPriority` rather than Reminders depending on
Tasks. The name (not `ReminderPriority`) signals the
intent.

**`isUpcoming` / `isSnoozed` / `isAcknowledged`
accessors:** the list view's filter buckets the in-memory
result set. Centralizing the bucketing on the model
itself keeps the filter logic in one place — adding a
new filter ("fired" / "snooze expired") is a new method,
not a new code path in the view.

## 4. List view

**macOS — `RemindersView` (NavigationSplitView):**
sidebar (the four ``ReminderFilter`` cases), list in the
middle (the filtered reminder rows), detail on the right
(`ReminderDetailView` with the receipt chain). The
toolbar has a Reload button and (when notifications are
disabled) a banner that opens System Settings to the
Notifications pane.

**iOS — `RemindersView_iOS` (NavigationStack + tab
filter strip):** a horizontal filter strip at the top
with the four filter pills and their counts, a `List` of
rows below, and a `NavigationLink` push to the
`ReminderDetailView_iOS`. The swipe action on a row is
Acknowledge / Delete; the bottom toolbar's menu has
Snooze + Delete.

**Row contents:** trigger time (relative — "in 15 min"),
title, the linked calendar event's title, the offset
("15 min before 'Q3 review meeting'"), the priority
badge.

**Filtering** is in-memory: pull every reminder, bucket
by the four cases (`upcoming` / `acknowledged` / `snoozed`
/ `all`), sort by `triggerAt` ascending (or by
`acknowledgedAt` / `snoozedUntil` descending for the
buckets that need it). The reminders table is small
(hundreds, not millions), so four passes per filter
switch is fine; the alternative — a SQL filter per case —
would need four separate indexes and four round-trips.

**The receipt chain** is part of the detail view, not the
list. The list shows the current state; the detail shows
the full audit trail. The receipt drawer (Phase 3) is
the cross-surface view of the same chain; the Reminders
detail just shows the reminder-scoped slice.

## 5. Notification scheduling

The scheduler is an actor that wraps
`UNUserNotificationCenter`:

```swift
public actor ReminderNotificationScheduler {
    public init(center: UNUserNotificationCenter = .current(),
                identifierPrefix: String = "tessera.reminder.")
    public func authorizationStatus() async -> UNAuthorizationStatus
    public func requestAuthorization() async throws -> Bool
    public func schedule(_ reminder: Reminder) async throws
    public func cancel(_ reminder: Reminder) async
    public func snooze(_ reminder: Reminder, until: Date) async throws
    public func rescheduleAll(_ reminders: [Reminder]) async throws
}
```

**Authorization is lazy.** The first call to ``schedule(_:)``
or ``requestAuthorization()`` triggers the system prompt;
the scheduler never prompts on app launch. The macOS /
iOS surface shows a "Notifications disabled — open
Settings" banner when `authorizationStatus == .denied`.

**Scheduling.** `UNCalendarNotificationTrigger` at
`triggerAt` (or `snoozedUntil` when the reminder is
currently snoozed). The trigger fires once (no
`repeats: true`). Past fire dates are rejected with
``ReminderNotificationError.triggerInPast`` — the
notification center silently drops them, but the store
still wants to log the rejection so the user can see why
a snoozed reminder didn't fire.

**Snooze.** ``snooze(_:until:)`` cancels the original
`triggerAt` notification and schedules a new one at
`until`. The store writes the `snoozedUntil` row update;
the scheduler owns the cancel + re-schedule. The two
happen in two `await`s — the receipt chain shows the
intent before the scheduler mutates the system state.

**Acknowledge.** The store writes the `acknowledgedAt`
update + receipt; the caller (the SwiftUI view or the
agent tool) calls `scheduler.cancel(_:)` after the
acknowledge. The store does not own the notification
center, so the cancel is a separate call.

**Reschedule on cold start.** The notification center
forgets pending requests on cold start if the app was
terminated for more than a day. The view-model's
``rescheduleAll(_:)`` method (called on app launch)
reads every reminder, cancels the reminder-prefixed
pending requests, and re-schedules the not-yet-fired
ones. The system list of pending requests is filtered
by the prefix so other apps' notifications are
untouched.

**Testability.** The scheduler's pure-decision helpers
(`identifier(for:prefix:)`,
`effectiveFireDate(for:now:)`,
`isFireDateInPast(for:now:)`) are static and tested
without invoking a real `UNUserNotificationCenter`.
The actual `center.add(request)` call is exercised by
hand-driven integration tests (not in CI — the system
notification center is not headless).

## 6. Chat panel integration

The chat panel surfaces four tools for the agent to
call:

- `reminder_create` — create a reminder linked to a
  calendar event. The agent resolves the calendar event
  id via hybrid search; the tool persists the row + the
  receipt.
- `reminder_list` — list reminders, optionally scoped
  to one event. Returns a compact text table the chat
  panel renders inline. The agent uses this for "what
  are my reminders today?" and "what reminders do I
  have for the Q3 review meeting?".
- `reminder_dismiss` — acknowledge / dismiss a
  reminder. The store writes the `acknowledgedAt`
  update + the receipt; the caller cancels the pending
  notification.
- `reminder_snooze` — snooze a reminder for N minutes.
  Capped at 1440 (one day) per spec §12.3 v1. The
  store writes the `snoozedUntil` update + the receipt;
  the caller cancels + re-schedules the notification.

The `ReminderCommandParser` is a best-effort
natural-language bridge from the chat panel's free-text
input to the structured tool calls. The matching is
case-insensitive on the verb ("remind me", "dismiss",
"snooze") and a regex on the offset ("15 min before",
"2 hours after"); the extracted event-title fragment
preserves the user's case (so the agent's fuzzy-match
step has the best chance of resolving the right
event). The parser is deliberately not clever — it
returns `nil` for unmatched input and the agent falls
back to asking the user for clarification.

The applied `ChatQueueItem` carries the receipt chip
("Reminder created: 15 min before Q3 review meeting")
via the existing `ChatQueueItemStyle.receiptChip`
field. Tapping the chip opens the reminder in the
Reminders surface; the chat panel's `onOpenReceipt`
callback routes the receipt id to the surface router.

## 7. Receipt model

Every mutation produces a constitutional receipt:

| Mutation | Receipt type |
|---|---|
| Create | `reminder_created` |
| Update (offset / title / notes / priority) | `reminder_updated` |
| Acknowledge | `reminder_acknowledged` |
| Snooze | `reminder_snoozed` |
| Delete | `reminder_deleted` (the receipt survives via the per-entity chain) |
| Link to another entity | `reminder_link_created` |
| Unlink | `reminder_link_deleted` |

The receipt chain is the audit trail the user sees in
the receipt drawer. The detail view's "Receipts"
section shows the reminder-scoped slice; the receipt
drawer (Phase 3) shows the same chain in the
inspector pane.

The payload is a `[String: JSONValue]` map:

- `created` / `updated`: `title`, `calendarEventID`,
  `offsetMinutes`, `triggerAt` (ISO-8601), `priority`.
- `acknowledged`: `acknowledgedAt`.
- `snoozed`: `snoozedUntil`.
- `linkCreated` / `linkDeleted`:
  `targetEntityID`, `linkType`, `weight`.

The receipt type strings are pinned in
``ReminderReceiptType`` (and asserted in
``ReminderTests/testReceiptTypesAreStable``); changing
them is a schema migration.

## 8. Cross-surface links

Reminders are linked to other graph entities via
`entity_links` (Phase 1's data layer seam):

- **Calendar events** (the source) — `linkType` defaults
  to `reminder_for`; the chat panel and the
  Reminders view both use this when resolving a
  reminder to its event.
- **Tasks** (follow-up) — `linkType = follow_up_to`. A
  reminder "remind me to review the Q3 slides" can
  be a follow-up to the task "review Q3 slides".
- **Contacts** (attendees) — `linkType = attendee_of`
  (delegated to the contact side; the link is
  per-attendee). The reminder is a notification
  surface; the contact is the human context.
- **Notes / Documents** (prep) —
  `linkType = prep_for`. The reminder fires
  before the meeting; the linked doc is what the
  user reviews when the reminder fires.

The links are visible in the reminder detail view's
"Linked" section (per spec §12.3 v2 — v1 ships the
data; the view's section header is a stub for the
worker that owns the cross-surface rendering).

The graph view's "force-directed layout" (Phase 6) is
unchanged: reminder nodes appear with the yellow
color and bell icon already wired into
``GraphNode.iconName(for:subtype:)`` and
``GraphNode.color(for:)``. Clicking a reminder node
opens the reminder in the Reminders surface via the
existing `openEntity(_:)` callback.

## 9. Graph view integration

`GraphModel.swift` already maps `"reminder"` to the
bell icon and yellow color (this was added in Phase 6
when the graph view's color map was extended). No
changes are needed in the graph view itself.

The graph view's node-creation loop reads every
`graph_entity` row; the Reminders material surfaces
automatically. The link types drive the edge colors
(per the existing `GraphEdge.color` mapping). A
reminder linked to a calendar event renders as a
yellow → purple edge; a reminder linked to a task
renders as a yellow → green edge; a reminder linked
to a contact renders as a yellow → orange edge.

**Click handling.** The graph view's
`onNodeTapped(_:)` callback receives the node's
`entityID`. The surface router looks up the entity
type; for `entity_type == "reminder"`, it opens
the Reminders detail view with the reminder's id.
The router is owned by the productivity surface
view coordinator, not by the Reminders surface
itself.

## 10. Library survey

| Need | Library | Decision |
|---|---|---|
| Notifications | `UserNotifications` framework (`UNUserNotificationCenter`, `UNCalendarNotificationTrigger`) | Adopt (Apple's standard, OS-managed, Focus-aware) |
| SwiftUI list UI | `SwiftUI` `List`, `NavigationSplitView` (macOS), `NavigationStack` (iOS) | Adopt (already in use across the surface) |
| Time-zone handling | Foundation `TimeZone` + `DateComponents` | Adopt (the system notification center schedules in the user's local TZ; we don't need to track event-start TZ explicitly) |
| JSON | `JSONEncoder` / `JSONDecoder` (Foundation) | Adopt (same pattern as `Contact` / `Document`) |
| ID generation | `UUID` (Foundation) | Adopt (matches the data layer's `uuid` PKs) |

No new third-party dependencies. The receipt chain
re-uses the existing `graph_receipts` table; the
notifications re-use `UNUserNotificationCenter`; the
SwiftUI list re-uses the patterns from the Contacts
and Tasks surfaces.

## 11. Test strategy

- **Type / JSON tests** (``ReminderTests`` — 19
  tests): the value type, JSON round-trip, offset
  math, trigger-time computation, state accessors,
  display line, priority ordering, error equality,
  hashable contract, receipt-type strings.
- **Filter tests** (``ReminderFilterTests`` — 4
  tests): bucketing by `upcoming` / `acknowledged` /
  `snoozed` / `all`; sort orders per bucket.
- **Scheduler tests**
  (``ReminderNotificationSchedulerTests`` — 7 tests):
  pure-decision helpers (`identifier`,
  `effectiveFireDate`, `isFireDateInPast`).
  Real `UNUserNotificationCenter` calls are
  hand-driven, not in CI.
- **Parser tests** (``ReminderParsingTests`` — 15
  tests): the natural-language bridge. Create /
  list / dismiss / snooze intents, case
  preservation, "the" prefix stripping, "reminder"
  suffix stripping, unmatched inputs.
- **Agent tool tests**
  (``ReminderAgentToolsTests`` — 9 tests): parameter
  validation (missing title, invalid UUIDs, invalid
  ISO-8601, snooze duration cap), create + list +
  dismiss + snooze flows against an in-memory
  ``ReminderStoring`` mock.
- **Integration tests**
  (``ReminderStoreIntegrationTests`` — 8 tests,
  env-gated on `TESSERA_DB_INTEGRATION=1`):
  end-to-end Postgres round-trip, every mutation
  appends a receipt, list sorts by `triggerAt`,
  list-for-event scopes correctly.

Total: 62 unit tests + 8 integration (skipped
without DB). All 62 unit tests pass; the 8
integration tests are skipped in CI but pass when
run against a real Postgres.

## 12. Out of scope (v2)

- **Apple Reminders import.** The system Reminders
  app's `EKReminder` / `EventKit` import is a v2
  feature; the spec lists it as a Phase 5 v2.
- **Google Calendar import.** v2.
- **Snooze durations > 1 day.** The spec's v1 cap
  is 1440 min; the snooze tool rejects anything
  larger. v2 lifts the cap (e.g. "snooze until
  tomorrow morning").
- **Recurring reminders.** v1 is one-shot. v2
  adds `UNCalendarNotificationTrigger(repeats:
  true)` for "every Monday at 9am" style
  reminders.
- **Cross-surface cross-link rendering.** The
  detail view's "Linked" section is a stub in
  v1; v2 renders the linked entities as a list
  with navigation to each one.
- **Geofenced reminders.** v1 is purely
  time-based. v2 adds `UNLocationTrigger`-based
  reminders ("when I arrive at the office").
- **Reminders import / export to file.** v1
  surfaces only system notifications. v2 adds
  a JSON / ICS export path through the
  `TesseraExporter` / `TesseraImporter`
  pipelines (the Phase 4 import / export work).
- **Snooze picker UI on macOS.** v1 uses the
  context-menu preset list (5/10/15/30/60 min);
  v2 adds a date picker for "snooze until …".
