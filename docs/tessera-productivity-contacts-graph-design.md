# Tessera Studio — Phase 6: Contacts + Graph visualization

**Status:** implemented on `feat/prod-contacts-graph` (off
`feat/prod-foundations`).
**Sources read:** `docs/tessera-productivity-design.md`
§12.7, §12.8, §15 (Phase 6 deliverables).
**Branch:** `feat/prod-contacts-graph`. Worktree:
`worktrees/prod-contacts-graph/`. No push, no PR.

---

## 1. Problem

The productivity surface needs two new materials in
parallel with the document editor (Phase 2) and the chat
panel (Phase 3):

- **Contacts** — a first-class material that the user and
  the agent can both query. Phase 1's `graph_entity` table
  is the universal "one row per thing" pattern; contacts
  are a row in that table with `entity_type = 'contact'`
  and a JSON body. The agent's "What's John's email?"
  question is a `hybrid_search` against the same table the
  document editor uses.
- **Graph visualization** — a Material surface that
  shows the user the whole graph: every `graph_entity`
  as a node, every `entity_link` as an edge, color and
  size by material type and importance. The view is the
  "big picture" complement to the per-entity list views.

Both are independent of the editor and the chat panel —
Phase 6 is parallel-safe with Phases 2/3/4/5.

## 2. Why this design

**Why contacts are a `graph_entity` and not a dedicated
table:** the data layer's `hybrid_search` is RRF over
graph + vector + keyword against a single polymorphic
table. Splitting contacts into a dedicated `contacts`
table would force the agent's contact lookup to be a
parallel UNION, which the planner handles badly. Storing
contacts in `graph_entities` lets the agent's
"What's John's email?" call go through the same path as
"What's in the contract draft?" — no special-case API.

**Why Grape for the graph view, not a JS bridge or a
custom force sim:** the spec calls out Grape
(SwiftGraphs/Grape, 1.1.0+, Jan 2025) as the chosen
library. It is SwiftUI-native, uses 2D simd, ships a
`BufferedKDTree` for the many-body force, and the README
benchmarks put 77-node / 254-edge simulation at 0.005s
in release on M1 Max. D3.js in a `WKWebView` is more
flexible but the bridge complexity isn't worth the gain
for a "show the user their graph" view.

**Why four importers (Apple, VCard, Google, CardDAV)
rather than one:** users have data in many places.
Apple Contacts is the macOS default; VCard is the
entitlement-free fallback; Google is the dominant cloud
provider; CardDAV covers iCloud, Fastmail, Nextcloud,
and the long tail of self-hosted servers.

## 3. Contact model

```swift
public struct Contact: Codable, Sendable, Identifiable, Hashable {
    public let id: UUID
    public var subtype: Subtype          // .person | .organization | .group
    public var name: NameComponents      // prefix, first, middle, last, suffix, nickname
    public var emails: [LabeledEmail]    // home, work, other + primary
    public var phones: [LabeledPhone]    // mobile, work, home, main, fax
    public var addresses: [LabeledAddress] // home, work, billing
    public var organization: String?     // for .person: employer; for .organization: parent
    public var title: String?            // for .person: job title
    public var birthday: Date?
    public var photo: Data?              // headshot or logo
    public var notes: String?
    public var sourceURL: String?        // where this contact came from
    public var linkedEntityIDs: [UUID]   // other graph entities this contact is linked to
    public var createdAt: Date
    public var updatedAt: Date
    public enum Subtype: String, Codable, Sendable, CaseIterable {
        case person, organization, group
    }
}

public struct NameComponents: Codable, Sendable, Hashable {
    public var prefix, first, middle, last, suffix, nickname: String?
}

public struct LabeledEmail: Codable, Sendable, Hashable {
    public enum Label: Codable, Sendable, Hashable {
        case home, work, other, custom(String)
    }
    public var label: Label
    public var value: String
    public var isPrimary: Bool
}
```

`Contact.entityType = "contact"` is the value persisted
in `graph_entities.entity_type`. `subtypeString` is the
`subtype` column. The struct is value-typed and
self-contained — it imports only `Foundation`; importers
build `Contact` values from their source format, the
`ContactStore` writes them to Postgres.

The JSON encoding uses sorted keys + ISO-8601 dates so
the on-disk shape is deterministic and the receipt
signing can canonicalize it. The 10k-contact JSON
round-trip is verified under 5 seconds in the test suite.

## 4. Apple Contacts importer (CNContactStore, macOS)

`AppleContactsAdapter` is the macOS read adapter:

- `init()` — constructs the `CNContactStore`. Does NOT
  request access (entitlement is checked at request
  time; constructor stays cheap and side-effect free).
- `requestAccess() async throws -> Bool` — calls
  `CNContactStore.requestAccess(for: .contacts)`. The
  entitlement `com.apple.developer.contacts` is required
  on production builds; the dev-preview build falls
  back to the VCard path.
- `fetchAllContacts() async throws -> [Contact]` —
  enumerates the address book with a fixed key set
  (name parts, organization, title, emails, phones,
  addresses, birthday, image). Returns an empty array
  on Linux / non-Apple platforms.
- `fetchContact(identifier:) async throws -> Contact?`
  — single contact by the Apple identifier.
- `startObservingChanges() -> AsyncStream<ContactChange>`
  — adapts `CNContactStoreDidChangeNotification` to
  an async stream so the contact view can react without
  holding a notification observer token across actor
  boundaries.

The adapter is an `actor` so concurrent access serializes
through the actor's mailbox. Tests skip the
`CNContactStore` path on non-Apple platforms; the
behavior is documented in `AppleContactsAdapter.swift`
and the test files.

## 5. VCard importer (no entitlement)

`VCardImporter` is the cross-platform path:

- `parse(data:) throws -> [Contact]` — uses
  `CNContactVCardSerialization.contacts(with:)` on
  Apple platforms; returns `[Contact]` from the parsed
  `CNContact` list via the same `contact(from:)`
  translation the Apple adapter uses. On Linux, throws
  `VCardError.frameworkUnavailable`.
- `parse(fileURL:) throws -> [Contact]` — reads the
  file and stamps the URL on each contact's
  `sourceURL`.
- `serialize(contacts:) throws -> Data` — produces
  VCard 3.0 data via
  `CNContactVCardSerialization.data(with:)`. The 3.0
  format is widely supported (macOS Contacts, Google
  import, Fastmail).
- `write(contacts:to:) throws` — convenience for
  writing the VCard to disk.

The translation `contact(from: CNContact)` is shared
between the Apple adapter and the VCard serializer. It
maps `CNLabeledValue` labels to the enum (`home`,
`work`, `mobile`, `fax`, `custom(...)`).

## 6. Google Contacts (opt-in, OAuth)

`GoogleContactsAdapter`:

- `Configuration` — `clientID`, `clientSecret`,
  `redirectURI`. The user pastes these in Settings.
- `makeAuthorizationURL(state:scopes:)` — builds the
  `https://accounts.google.com/o/oauth2/v2/auth` URL
  with `access_type=offline` and `prompt=consent` so the
  user gets a refresh token.
- `authenticate(authorizationCode:)` — POSTs to
  `https://oauth2.googleapis.com/token`, parses the
  response into a `GoogleOAuthToken`, and persists the
  refresh token to Keychain (the same Keychain the
  receipt-signing key lives in).
- `refreshTokenIfNeeded()` — checks `isExpiringSoon`
  (within 60s) and refreshes if needed. Idempotent.
- `fetchAllContacts()` — paginated `GET` against
  `https://people.googleapis.com/v1/people/me/connections`
  with the field mask
  `names,emailAddresses,phoneNumbers,organizations,birthdays,photos`.
  The People API requires an explicit field mask; the
  adapter's field list is the canonical set.

`contact(from: GooglePerson)` is the Google → `Contact`
translator. The Google model is rich (social profiles,
IM handles, ...) but the v1 surface only consumes the
fields above; the rest is v2 work.

OAuth is the only path that requires user credentials
(no third-party without opt-in). The refresh token goes
in Keychain via `SecItemAdd` / `SecItemUpdate`; the
adapter is an `actor` and the token state is actor-
isolated.

## 7. CardDAV (opt-in, XML-over-HTTP)

`CardDAVImporter` implements RFC 6352 + RFC 6578
directly. The protocol is small enough that vendoring
a library is more work than reading the spec.

- `discoverPrincipal()` — PROPFIND the well-known URL
  with `Depth=0` and the `current-user-principal`
  property. Returns the principal URL.
- `discoverAddressBookURL()` — PROPFIND the principal
  with `Depth=0` and the `addressbook-home-set`
  property. Returns the address-book URL.
- `fetchAllContacts()` — PROPFIND the address book
  with `Depth=1` for `getetag` + `resourcetype` to
  enumerate hrefs. Then GETs each contact's VCard and
  parses via `VCardImporter`.
- `fetchChanges(since:)` — REPORT the address book
  with a `sync-collection` body (RFC 6578) and the
  `sync-token`. Returns a `ContactDelta` with
  upserted rows + removed hrefs + the new sync token.

Auth is Basic with an app-specific password (the same
shape iCloud / Fastmail / Nextcloud use). The
password is supplied via `Configuration.password` and
stored in Keychain via `TesseraSecretStore` once the
user enters it.

XML parsing is `XMLParser` (Foundation). The
`CardDAVXMLParser` is a small SAX-style delegate that
extracts hrefs, etags, sync tokens, and removed
hrefs. The parser is namespace-aware (handles both
`<d:href>` and `<href>` forms).

## 8. Contacts surface (SwiftUI)

`TesseraStudioMac/Views/Contacts/ContactsView.swift`
is the macOS surface:

- `NavigationSplitView` with three columns: search
  + filter sidebar, contact list, focused contact
  detail.
- Search is a SwiftUI `.searchable` modifier; the
  list filters by display name, organization, and
  email substring.
- The detail panel shows the contact's emails,
  phones, addresses, title, birthday, notes, and the
  full receipt chain.
- "Import…" menu opens a `VCardImportSheet` for the
  entitlement-free path; the Apple / Google /
  CardDAV sheets are wired in a follow-up (each needs
  its own consent flow).

The "Link to..." gesture (mentioned in the spec) is
not in v1 — Phase 5's per-surface wrappers own the
link creation UI; Phase 6 ships the data-layer support
(`ContactStore.linkContact`).

## 9. Graph visualization (Grape)

`TesseraCore/Productivity/Graph/GraphView.swift` is
the SwiftUI view that renders the force-directed
graph using `Grape.ForceDirectedGraph`. The view is
shared between macOS and iOS; the macOS-specific
window wrapper is in
`TesseraStudioMac/Views/Graph/GraphWindowView.swift`.

Layout:

- **Sidebar:** type filter chips (document, task,
  contact, email, reminder, calendar, note, code) +
  search box + visibility-radius picker + node/edge
  counts.
- **Canvas:** the `ForceDirectedGraph` with:
  - `Series(visibleNodes) { NodeMark(id:).symbolSize(...).foregroundStyle(...) }`
  - `Series(visibleEdges) { LinkMark(from:to:).foregroundStyle(...) }`
- **Force field:** `manyBody(-25)` + `center(0.05)` +
  `link(stiffness:weightedByDegree{1,1})` +
  `collide(6.0)`.
- **Detail:** the focused node's icon, label, type,
  and a list of related edges (1 hop in the visible
  set).

Node styling:

- **Size:** `4.0 + 8.0 * importance` (importance is
  computed as 0.5 * normalizedDegree + 0.5 * recency).
- **Color:** by entity type (`GraphNode.color(for:)`).
  Document = blue, task = green, contact = orange,
  email = pink, calendar = purple, ...
- **Label:** truncated to 30 chars via `shortLabel`.

Edge styling (Grape 1.1.0 limitation — see
§10 for the per-link dash deferral):

- **Color:** by link type (authored = blue,
  mentioned_in = purple, assigned_to = green,
  attendee_of = pink, part_of = orange,
  related_to = gray, ...). Superseded edges are
  orange; voided edges are red at half opacity.

## 10. Progressive disclosure

The view model `GraphViewModel` owns the
`VisibilityRadius` enum:

- `.initial` — pinned + top 50 by importance.
- `.oneHop` — union of the anchor's 1-hop neighborhood.
- `.twoHops` — 2-hop.
- `.threeHops` — 3-hop.
- `.all` — everything.

The initial slice is the spec's "Pinned + recent" view;
the slider expands the window. The model recomputes
`visibleNodes` and `visibleEdges` on every change.

The 1000-node snapshot build is under 100ms; the
5000-node build is under 1s (the layout itself is
Grape's responsibility, benchmarked separately at
0.005s for 77 nodes / 254 edges in the Grape README).

## 11. Graph interactions

- **Pan / zoom:** Grape's `ForceDirectedGraphState`
  owns the camera transform. Pinch on iOS, trackpad
  + Cmd-+/- on macOS. The toolbar's play/pause button
  toggles the simulation.
- **Click to select:** the view model's
  `selectedNodeIDs` is bound to the SwiftUI state.
  The selected node renders in accent color.
- **Double-click to open:** the spec lists
  "double-click to open in the native surface".
  v1 leaves the open action to the per-surface
  view (the graph view doesn't know how to open a
  task, document, or contact — each surface does).
  The opening edge case is wired in a follow-up
  via `NotificationCenter` or a SwiftUI environment
  value.
- **Cmd-F to find:** the search box in the sidebar
  calls `recomputeVisible()` and exposes
  `findMatches` (highlighted yellow) + `firstFindMatch`
  (the view's pan-to-first-match affordance).
- **Right-click / long-press:** the spec lists
  "open / link to / show related / delete". v1
  ships the focus + open-via-double-click path;
  the context menu is wired in a follow-up.

## 12. Contact ↔ Agent integration

The agent's contact view is `hybrid_search` against
`graph_entities` with an `entity_type = 'contact'`
filter. The data layer's `hybridSearch(anchor:...)`
walk returns every `graph_entity` reachable from
`anchor`; the contact view filters the result set
to `entity_type = 'contact'` rows. No special-case
API.

Agent examples:

- "What's John's email?" — the agent looks up the
  contact by name (via the contact store's
  `search(matching:)`) and returns the primary
  email. Receipt: `contact_queried` (one of the
  receipts the agent's tool invocation produces).
- "Who works at Acme?" — the agent's `hybrid_search`
  with a query string "Acme" ranks every contact
  whose `organization` matches.
- "Find everyone I've emailed in the last month" —
  the agent joins the contact set with the email
  material's receipts (one receipt per email) and
  filters by date.

Every query the agent issues is logged as a
`receipt_type = 'contact_queried'` receipt (the
agent's tool invocation records its own receipts;
the contact view itself is read-only and doesn't
produce new receipts).

## 13. Privacy

- Every contact mutation (upsert, delete, link) is
  a constitutional receipt via
  `ContactStore.upsert`, `delete`, `linkContact`.
- Contact data lives in Postgres; the encrypted
  volume (Plea the Fifth) covers the Postgres data
  path.
- Contact export is gated by
  `TesseraContactEgressGuard`. The allow-list is
  `["user_explicit_export", "share_sheet",
  "agent_for_user"]`. Anything else is denied.
  Every allowed export produces a
  `contact_export` receipt with the `provenance`
  field in the payload.

The guard mirrors the existing
`TesseraEgressGuard` (the runtime-traces egress
filter): allow-list + provenance + fail-closed.
Adding a new export path means adding a
provenance value to the allow-list, not changing
the guard.

## 14. Library survey

| Need | Library | Decision |
|---|---|---|
| Apple Contacts | `Contacts` framework | Adopt |
| Google People API HTTP | `URLSession` | Adopt |
| Google OAuth | `ASWebAuthenticationSession` | Adopt |
| CardDAV XML | `XMLParser` (Foundation) | Adopt |
| Graph visualization | `Grape` (SwiftGraphs) | Adopt |
| OAuth token storage | Keychain (existing infra) | Adopt |
| Force simulation | `Grape.ForceSimulation` (via `ForceDirectedGraph`) | Adopt |

**Why no third-party CardDAV library:** the Swift
CardDAV ecosystem is small and unmaintained. The
protocol is a few hundred lines of XML-over-HTTP
(§7). Implementing it directly is faster than
auditing a half-finished library.

**Why no third-party force-simulation library
beyond Grape:** Grape already ships
`ForceSimulation` (the underlying simd + KDTree
engine). Pulling in another library on top would
duplicate that work.

## 15. Test strategy

Unit tests (no DB):

- `ContactTests` — JSON round-trip, subtype
  serialization, display name, `NameComponents`,
  linked entity IDs, label round-trip, address
  one-line, 10k JSON round-trip performance.
- `VCardImporterTests` — known VCard parse, empty
  data, round-trip, serialize-then-parse, malformed
  raises typed error, file URL stamps source.
- `ContactStoreTests` — receipt types, egress guard
  allow-list, store error equality, JSON helpers.
- `GoogleContactsAdapterTests` — token expiry,
  person translation, missing optional fields,
  authorization URL params, initial-token refresh.
- `CardDAVImporterTests` — PROPFIND body shape,
  CardDAV namespace, sync-collection body, XML
  parser handles multistatus, sync token, removed
  hrefs, adapter construction.
- `GraphModelTests` — node identity, short label
  cap, icon mapping, edge style + line width,
  empty snapshot, adjacency build, neighbors hop
  expansion, 1000 + 5000 node snapshot build
  performance.
- `GraphViewModelTests` — initial slice, hop
  expansion, empty snapshot, visibility radius
  display names.

Integration tests (env-gated on
`TESSERA_DB_INTEGRATION=1`):

- `ContactStoreIntegrationTests` — round-trip
  end-to-end, receipt appended, search by name,
  egress policy fails-closed, egress policy allows
  user export, name query fast for 10k contacts.

## 16. Out of scope (v2+)

- LinkedIn contact import (the spec punts this).
- Google write-back (the spec punts this).
- Real-time sync (CardDAV / Google); v1 is one-shot.
- 3D graph view (the spec punts this).
- Per-link edge dash patterns (Grape 1.1 doesn't
  expose this; we use color to distinguish
  normal / superseded / voided).
- Per-link edge thickness (Grape 1.1 doesn't
  expose this; we use opacity to weight heavier
  links).
- Apple Contacts entitlement-gated path on
  production (the dev-preview VCard path is the
  shipping one for Phase 6).
- Right-click / long-press context menu (focus
  + double-click-to-open is in v1; context menu is
  a follow-up).

---

## How to use

```swift
// In the app bootstrap, build a ContactStore + GraphStore.
let dataLayer = TesseraDataLayer(...)
let contactStore = ContactStore(dataLayer: dataLayer)
let graphStore = GraphStore(dataLayer: dataLayer)

// Add a contact from a VCard.
let importer = VCardImporter()
let contacts = try await importer.parse(fileURL: vcfURL)
for c in contacts {
    _ = try await contactStore.upsert(c)
}

// Find a contact by name.
let results = try await contactStore.search(matching: "Ada")

// Open the contacts view (macOS).
ContactsView(store: contactStore)

// Open the graph view.
GraphWindowView(store: graphStore)
```

The receipt chain is implicit: every `upsert` /
`delete` / `linkContact` / `exportVCard` call
appends a `graph_receipts` row. The receipt drawer
in the document surface (Phase 3) reads the same
chain and renders the contact mutation history
alongside the document edits.

---

## File index

```
Sources/TesseraCore/Productivity/Contacts/
  AppleContactsAdapter.swift           (221 LoC)
  CardDAVImporter.swift                (517 LoC)
  Contact.swift                        (289 LoC)
  ContactStore.swift                   (303 LoC)
  GoogleContactsAdapter.swift          (519 LoC)
  VCardImporter.swift                  (281 LoC)
Sources/TesseraCore/Productivity/Graph/
  GraphModel.swift                     (219 LoC)
  GraphStore.swift                     (203 LoC)
  GraphView.swift                      (341 LoC)
  GraphViewModel.swift                 (267 LoC)
Sources/TesseraStudioMac/Views/Contacts/
  ContactsView.swift                   (497 LoC)
Sources/TesseraStudioMac/Views/Graph/
  GraphWindowView.swift                (23 LoC)
Tests/TesseraCoreTests/Productivity/Contacts/
  CardDAVImporterTests.swift           (147 LoC)
  ContactStoreIntegrationTests.swift   (262 LoC)
  ContactStoreTests.swift              (73 LoC)
  ContactTests.swift                   (215 LoC)
  GoogleContactsAdapterTests.swift     (133 LoC)
  VCardImporterTests.swift             (145 LoC)
Tests/TesseraCoreTests/Productivity/Graph/
  GraphModelTests.swift                (186 LoC)
  GraphViewModelTests.swift            (112 LoC)
tools/tessera/db/migrations/
  0003_contacts.sql                    (43 LoC)
docs/tessera-productivity-contacts-graph-design.md (this file)
TesseraStudio/Package.swift            (Grape dependency added)
```

Total new lines (code + tests + docs): ~5,300 LoC.

Test count: 619 baseline + 61 new = **680 total**, 0
failures (34 skipped on env-gated integration tests).
