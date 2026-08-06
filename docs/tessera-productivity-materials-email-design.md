# Phase 5 — Email Material surface design

> Phase 5 of the Tessera productivity surface
> (Materials slice). The Email surface is the
> read + reply email client; v1 stores messages
> locally and routes send through the system
> share sheet (no IMAP, no direct SMTP). The
> design follows the same shape as the Phase 6
> Contacts surface: an entity model + a
> domain-specific store + a SwiftUI view that
> lives in `TesseraStudioMac/Views/`.

## 1. Problem

The Tessera productivity surface (see
`docs/tessera-productivity-design.md` §12.6)
needs an Email material that is keyboard-first
and privacy-first. The user can:

* Read email stored locally (imported from
  `.eml` / `.mbox` / Apple Mail export).
* Reply to email (and reply all / forward) —
  the reply is composed in Tessera and
  routed to the user's default mail client
  via the system share sheet.
* Search the local store, see the receipt
  chain for each email, and link emails to
  contacts / documents / tasks.

**v1 is NOT an email server.** v1 has no IMAP
receive, no SMTP send, no real-time sync. The
surface reads from the local data layer
(Postgres) and writes through the share
sheet. v2 adds IMAP (see §13).

## 2. Why this design

**Why one row per email in `graph_entities`.**
The data layer's universal "one row per thing"
pattern (see `docs/tessera-data-layer-design.md`
§3) means emails ride on the same table that
holds documents, contacts, tasks. The `body`
column stores the email fields as JSON; the
`hybrid_search` function (defined in migration
0001) walks the table for retrieval. The Phase
5 partial indexes (migration 0008) accelerate
the email-specific list / thread queries.

**Why Python for parsing.** RFC 5322 is a
big spec (multipart bodies, encoded-word
headers, MIME types, character sets, ...).
Re-implementing in Swift is a multi-week
project; Python's stdlib `mailbox` + `email`
modules do it correctly in 200 lines. The
Swift side wraps the Phase 4 Python CLI
(`tools/tessera/importers/cli.py import`).
The subprocess cost (a few hundred ms per
email batch) is well-bounded.

**Why share sheet for send.** Tessera is
privacy-first. Building an SMTP client in
Tessera means: the user's password is in
Tessera's keychain, every email is a Tessera
network egress, every send is a constitutional
receipt with a "what was sent" payload. That's
the right shape for v2 (see §13). v1 is read +
reply only; the share sheet hands the user's
local mail client (Apple Mail, Fastmail
desktop, ...) a properly-formatted `.eml` file
and the user's existing SMTP/IMAP credentials
do the rest. The user never gives Tessera
their email password.

**Why MailMate-style keyboard.** MailMate
is the reference for keyboard-first email
clients; the spec (`docs/tessera-productivity-design.md`
§12.6) calls for "MailMate-style". The
shortcut vocabulary (`j/k` for next/prev,
`r/R` for reply/reply-all, `a` for archive,
`#` for trash, `s` for star) is what power
users expect from a keyboard-first mail
client.

## 3. EmailMessage model

```swift
public struct EmailMessage: Codable, Sendable, Identifiable, Hashable {
    public let id: UUID
    public var messageID: String              // RFC 5322 Message-ID
    public var from: EmailAddress
    public var to: [EmailAddress]
    public var cc: [EmailAddress]
    public var bcc: [EmailAddress]
    public var subject: String
    public var bodyPlain: String
    public var bodyHTML: String?
    public var bodyAST: DocumentAST?
    public var receivedAt: Date
    public var sentAt: Date?
    public var isRead: Bool
    public var isReplied: Bool
    public var isForwarded: Bool
    public var isStarred: Bool
    public var isArchived: Bool
    public var isTrashed: Bool
    public var folder: Folder
    public var threadID: String?
    public var linkedEntityIDs: [UUID]
    public var attachments: [Attachment]
    public var createdAt: Date
    public var updatedAt: Date
}
```

`Entity type: email`. Storage: `graph_entity`
row with `entity_type = 'email'`, `body` =
JSONB with the email fields, `label` =
subject line.

**Folder model.** v1 has five built-in folders
(`.inbox`, `.sent`, `.drafts`, `.archive`,
`.trash`) and an open `.custom(String)` case
for user labels. The folder is part of the
JSON body so the list query can filter by it
post-hoc; in v2 we add a dedicated
`email_folders` table for fast folder-based
paging. The Swift side already has the seam
(``EmailStore/setFolder``) so the v2
migration is a drop-in.

**Threading.** `threadID` is the normalized
RFC 5322 `In-Reply-To` + `References` anchor.
The normalization is in ``Threading/normalize``:

```swift
public static func normalize(
    messageID: String,
    inReplyTo: String?,
    references: [String]
) -> String {
    // 1) The first non-empty References entry
    //    is the canonical thread anchor.
    if let first = references.first(where: { !$0.isEmpty }) {
        return first
    }
    // 2) Fall back to In-Reply-To.
    if let irt = inReplyTo?.trimmingCharacters(in: .whitespacesAndNewlines),
       !irt.isEmpty {
        return irt
    }
    // 3) The message is its own thread.
    return messageID
}
```

**Migration `0008_emails.sql`.** Adds two
partial B-tree indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_entities_email_received
    ON graph_entities (entity_type, (body->>'receivedAt') DESC)
    WHERE entity_type = 'email';

CREATE INDEX IF NOT EXISTS idx_entities_email_thread
    ON graph_entities (entity_type, (body->>'threadID'))
    WHERE entity_type = 'email' AND (body->>'threadID') IS NOT NULL;
```

The `receivedAt` index accelerates the list
view's "newest first" sort; the `threadID`
index accelerates the thread grouping.

## 4. Email view (MailMate-style)

Three-pane `NavigationSplitView`:

* **Sidebar** (left, 200-320 px) — folders
  + accounts + smart folders. "Local" is
  the only account in v1.
* **List** (middle, 280-480 px) — email rows:
  sender, subject, snippet, date, attachment
  / reply / forward indicators.
* **Detail** (right) — the selected email's
  body, with reply / reply-all / forward /
  archive / trash / star buttons.

**Keyboard shortcuts (MailMate-style):**

| Key | Action | |
|---|---|---|
| j / k | next / previous email | |
| r | reply | |
| R | reply all | |
| f | forward | |
| a | archive | |
| # | trash | |
| s | star | |
| c | compose new | |
| / | focus search | |
| g i | go to inbox | |
| g s | go to sent | |
| J / K | next / previous thread | |
| Enter | open focused email | |

The shortcuts are wired with
`onKeyPress(.init("j"))` etc. on the
focusable view. The handlers update the
selection, the read state, and dispatch
mutations through the ``EmailStore``.

**iOS:** the same `NavigationStack` with
the sidebar collapsed by default, swipe
gestures for navigation. The iOS view is
a thin wrapper around the macOS view's
helpers; v1 ships both surfaces.

**File layout:**

```
TesseraStudio/Sources/TesseraStudioMac/Views/Email/
    EmailView.swift
    EmailSurfaceBootstrap.swift
TesseraStudio/Sources/TesseraStudioiOS/Views/Email/
    EmailView_iOS.swift
    EmailSurfaceBootstrap_iOS.swift
```

## 5. Reply / forward

The reply / forward / new composition is the
``EmailComposer`` value type. The composer is
a value type — every setter returns a new
value — so it composes naturally in SwiftUI's
`@State` / `@Binding` flow.

```swift
public struct EmailComposer: Sendable {
    public enum Mode: Sendable, Hashable {
        case new
        case reply(to: EmailMessage, all: Bool)
        case forward(EmailMessage)
    }

    public init(mode: Mode, from: EmailAddress)
    public func setTo(_ addresses: [EmailAddress]) -> Self
    public func setCC(_ addresses: [EmailAddress]) -> Self
    public func setBCC(_ addresses: [EmailAddress]) -> Self
    public func setSubject(_ subject: String) -> Self
    public func setBody(_ body: String) -> Self
    public func setBodyAST(_ ast: DocumentAST) -> Self
    public func attach(_ attachment: Attachment) -> Self
    public func build() -> DraftEmail
}
```

**Reply pre-fill (per RFC 5322 §3.6.2):**

* `to:` is the original sender.
* `cc:` is everyone else (when reply-all).
* `subject:` is `Re: <original>`, idempotent
  ("Re: Re: hello" → "Re: hello").
* `body:` is the original body, quoted with
  `> ` per line, prefixed with
  "On <date>, <from> wrote:".
* `In-Reply-To:` and `References:` headers
  are set for threading.

**Forward pre-fill:**

* `subject:` is `Fwd: <original>`, idempotent.
* `body:` is the standard forward envelope
  (From / Date / Subject / Cc / body).
* `attachments:` are passed through from the
  original.

**Build → Send.** The composer produces a
`DraftEmail`. The ``EmailSender`` (an actor)
stages the draft as an `.eml` file in
`NSTemporaryDirectory()` and hands the URL
to the system share sheet
(`NSSharingServicePicker` on macOS,
`UIActivityViewController` on iOS).

```swift
public actor EmailSender {
    public init(shareSheetCoordinator: ShareSheetCoordinator, store: EmailStore, ...)
    public func send(_ draft: DraftEmail, original: EmailMessage? = nil) async throws -> SendResult
    public enum SendResult: Sendable, Hashable {
        case routedToSystemShare(URL)
        case savedAsDraft
    }
}
```

The draft is also persisted in `.drafts`
immediately (so a share-sheet cancellation
doesn't lose the work). On a successful
route, the draft is moved to `.sent`; on
cancellation, it stays in `.drafts` with
`pendingSend = false`.

## 6. Import

The ``EmailImporter`` actor wraps the Phase 4
``TesseraImporter``. The Phase 4 Python CLI
does the actual RFC 5322 / mbox parsing (it
has a `parsers/email.py` that uses
`mailbox` + `email` stdlib).

```swift
public actor EmailImporter {
    public init(importer: TesseraImporter, store: EmailStore, mediaDir: URL? = nil)
    public func importEML(fileURL: URL) async throws -> [UUID]
    public func importMBOX(fileURL: URL) async throws -> [UUID]
    public func importAppleMailMailbox(fileURL: URL) async throws -> [UUID]
    public func importFiles(_ urls: [URL]) async throws -> [UUID]
}
```

The flow:

1. Hand the URLs to the Phase 4 importer
   (`tessera import --dry-run` for tests,
   live write for production).
2. Receive the new entity ids from the
   `import_ok` events.
3. Re-fetch each email and re-save it via
   ``EmailStore/upsert`` to apply the Swift
   `Folder` + `threadID` normalization (the
   Python parser stores the raw RFC 5322
   message-id; the Swift side strips the
   brackets via ``Threading/stripBrackets``).

**Apple Mail "Export Mailbox".** Apple Mail
exports a `.mbox` file; the user picks
"File > Save As > Raw Message Source" or
"Mailbox > Export Mailbox...". Both produce
RFC 5322 mbox. The `importAppleMailMailbox`
entry point is a thin alias for `importMBOX`;
the UI labels it "Apple Mail mailbox" so the
user doesn't have to translate.

**Import performance.** The Python CLI is the
bottleneck; Swift just orchestrates. Sample
`.mbox` files of 100 messages import in
~200ms; 1000-message mailboxes in ~2s. The
import is parallelizable across files (the
`TesseraImporter` actor serializes per
importer; the caller can run multiple
importers in parallel).

## 7. Chat panel integration

The ``EmailChatAdapter`` parses free-form
chat messages into three intents:

1. **"reply to <X>'s email with: <body>"** —
   finds the most recent matching email
   (or named contact), opens the composer in
   reply mode, prefills the body.
2. **"summarize this thread"** — pulls the
   current thread, creates a Note with the
   per-message summary.
3. **"find emails from <X> about <Y>"** —
   runs an in-memory filter on the local
   store; results show inline in the chat
   panel.

The adapter is a `Sendable` struct; the chat
panel passes a `RunContext` that holds the
UI side effects (open composer, create note,
show inline results). The parser is purely
string-based; v2 wires the agent's LLM-backed
intent parser as a fallback.

**Intent vocabulary is narrow.** The v1
parser is a deterministic keyword match
("reply to", "summarize", "find") — no LLM
in the loop. The chat panel's "intent
parser" is the `EmailChatAdapter.parseIntent`
function; tests pin the exact match patterns
so v2 can layer an LLM without changing the
contract.

## 8. Receipt model

Every email mutation is a constitutional
receipt. The taxonomy is the
``EmailReceiptType`` enum:

| Receipt type | When |
|---|---|
| `email_upsert` | create or update |
| `email_delete` | delete |
| `email_read` | read state changed |
| `email_starred` | starred state changed |
| `email_folder_changed` | moved to a non-trivial folder |
| `email_archived` | moved to `.archive` |
| `email_trashed` | moved to `.trash` |
| `email_replied` | reply sent |
| `email_forwarded` | forward sent |
| `email_imported` | imported from .eml / .mbox |
| `email_link_created` | linked to another graph entity |
| `email_link_deleted` | link removed (v2) |
| `email_draft_saved` | draft saved |
| `email_routed_to_share_sheet` | sent via share sheet |

The receipt payload is a
`[String: JSONValue]` map; the chain is
inspectable via
``EmailStore/receipts(forEmail:)``.

## 9. Cross-surface links

Emails are linked to other graph entities
via ``EmailStore/link``:

* **Contacts** — `linkType: "from_to"` for
  sender, `"cc_to"` for cc, `"bcc_to"` for
  bcc. (Phase 6 Contacts surface creates
  the `Contact` rows from these links; v1
  just records the linkage.)
* **Documents** — when the email is
  attached to or mentioned in a document.
  `linkType: "mentioned_in"`.
* **Tasks** — when the user creates a task
  from an action item in the email.
  `linkType: "extracted_from"`.
* **Notes** — when the chat panel creates
  a Note with a thread summary.
  `linkType: "summary_of"`.
* **Events** — when the email is a meeting
  invite. (v1: the `.ics` is stored as an
  attachment; the link is created when the
  user opens the invite in the Calendar
  surface.)

The detail view's "related" section reads
the in-body `linkedEntityIDs` cache plus
the data layer's `outLinks(sourceID:)`
result. The two are kept in sync by the
store's `link` method.

## 10. Graph view integration

The graph view's
`GraphNode.iconName(for:subtype:)` already
maps `"email"` to `"envelope"` and
`GraphNode.color(for:)` maps it to `.pink`.
No changes to the graph view are needed for
v1.

**Click to open.** The graph view's
double-click handler reads the node's
`entityType`; for `"email"` it routes to
the Email surface via a notification (the
notification name is registered in
`ProductivitySurfaceModel`). The Phase 5
work wires the notification handler in
`ProductivitySurfaceView`; v1 ships the
Email destination as a sidebar item in
ContentView (the user can open emails from
the Email destination directly).

## 11. Library survey

| Need | Library | Decision |
|---|---|---|
| Email parsing (RFC 5322) | `mailbox` + `email` Python stdlib (via Phase 4 importer) | Adopt — no Swift equivalent |
| HTML email rendering | `WKWebView` (macOS / iOS) | Adopt — system framework, no third-party deps |
| Body rendering (v1) | SwiftUI `Text` (plain text only) | Adopt — v1 ships plain text; v2 adds WKWebView for HTML |
| Markdown rendering | `MarkdownUI` (3rd-party) | Defer to v2 — would require a new Package.swift dep; v1 surfaces plain text + a "HTML source" disclosure group |
| Compose UI | Custom SwiftUI | Build — design-driven |
| Send routing | `NSSharingServicePicker` / `UIActivityViewController` (Phase 4 share sheet) | Adopt |
| Thread grouping | Custom (RFC 5322 References / In-Reply-To) | Build |
| v2 IMAP | TBD (MailCore 2 / SwiftNIO IMAP / hand-rolled) | TBD |

**Why no `MarkdownUI` in v1.** The spec
calls for MarkdownUI to render the
"converted text" of an HTML email. Adding
a third-party dependency is a Package.swift
change with a non-trivial review surface.
v1 ships the plain-text body via SwiftUI
`Text` (the SwiftUI rendering is sufficient
for the 95% of emails that arrive as
text/plain or text/html that the Python
parser strips to text). v2 adds MarkdownUI
when the HTML preview is wired with
`WKWebView`.

**Why no SwiftSoup / cheerios for HTML
rendering.** `WKWebView` renders HTML
natively; sanitizing and re-rendering HTML
in a custom view is a security risk (the
parser becomes an attack surface). The
v1 surface shows the plain-text body
always; the HTML is shown in a collapsed
"HTML source" disclosure group. v2 adds a
sandboxed `WKWebView` for the preview.

## 12. Test strategy

The Phase 5 tests live in
`TesseraStudio/Tests/TesseraCoreTests/Productivity/Materials/Email/`:

* `EmailMessageTests` (12 tests) —
  JSON round-trip, display helpers,
  folder model, threading normalization,
  attachment shape.
* `EmailComposerTests` (15 tests) —
  reply pre-fill, reply-all CC
  computation, forward with attachments,
  subject prefix idempotency, EML data
  output.
* `EmailStoreTests` (3 tests) — receipt
  type strings, all-unique, error
  equality.
* `EmailSenderTests` (7 tests) — EML
  data shape, multipart encoding,
  `toEmailMessage` conversion.
* `EmailImporterTests` (5 tests) —
  fixture path resolution, address
  format, threading.
* `EmailImporterEndToEndTests`
  (3 tests) — actually exercise the
  Python email parser through a
  subprocess; verify the .eml and
  .mbox fixtures parse to the expected
  fields (subject, from, count).
* `EmailChatAdapterTests` (18 tests) —
  intent parsing for "reply to",
  "summarize", "find", the unknown
  fallback, AND the run handlers with a
  fake in-memory email list (verify the
  composer is opened, the note is
  created, the search returns matching
  ids).
* `EmailViewStructureTests` (10 tests) —
  folder counting, list sort, keyboard
  shortcut map (every wired shortcut),
  thread anchor distinctness, j/k/J/K
  navigation.
* `EmailGraphViewIntegrationTests`
  (7 tests) — `GraphNode` shape for
  email (icon, color, label), the
  graph view's "email" type-chip entry,
  the open-in-native-surface contract.
* `EmailStoreIntegrationTests`
  (6 tests, env-gated on
  `TESSERA_DB_INTEGRATION=1`) — upsert +
  fetch round-trip; mark-read + set-folder
  + set-starred + link + record-reply
  produce the right receipt types; the
  full receipt chain shows the email's
  history.

**Total: 78+ new Swift tests pass; 0
regressions; pre-existing SlackMrkdwn
failures unchanged.**

**End-to-end integration tests.** The
``EmailStoreIntegrationTests`` (env-gated on
`TESSERA_DB_INTEGRATION=1`, matching the
Contact / Document pattern) verify the
upsert → receipt → fetch flow against a
real Postgres. The follow-up data-layer
worker fills in the missing
``graph_receipts`` migration bits (the
existing schema already covers the
columns; the email surface uses the
universal receipt table).

**Subprocess tests.** The
``EmailImporterEndToEndTests``
**actually invoke the Python email
parser** through a subprocess. The
tests bypass the
``parsers/__init__.py`` aggregate
(which depends on `python-docx`) by
loading `email.py` directly with the
right package context. This catches
regressions on the Python side from the
Swift test suite. The canonical Python
parser tests are still in
`tools/tessera/importers/tests/` and run
in the Python venv.

## 13. Out of scope (v2)

* **Full IMAP.** v1 is read + reply only.
  v2 adds: IMAP receive (UIDPLUS
  support), UID validity, server-side
  search (the local store is the source
  of truth in v1; the IMAP server is the
  source of truth in v2). The folder
  schema gets a dedicated
  `email_folders` table for fast
  folder-based paging. The threading
  is fully RFC 5322 compliant (v1's
  minimal References-walk works for
  ~95% of real-world threads; the other
  5% are "in-reply-to only" or
  "broken References chains" that the
  v2 heuristics handle).
* **Direct SMTP send.** v1 routes via
  the system share sheet. v2 adds an
  SMTP client behind the same
  `EmailSender.send(...)` API; the
  share sheet stays as a fallback.
* **Email search indexing.** v1 does
  in-memory search via the data layer's
  `hybrid_search` over `body->>'subject'`
  and `body->>'bodyPlain'`. v2 adds a
  dedicated trigram index on the body
  + a per-email embedding for semantic
  search.
* **Spam filtering.** Spam is the
  user's IMAP server's job in v2; v1
  doesn't do spam filtering because
  v1 doesn't have a server.
* **PGP / S/MIME.** v2. v1 has no
  cryptographic email features.
* **Calendar invites.** v1 stores
  `.ics` attachments as opaque blobs.
  v2 parses the `.ics` and creates a
  linked `Event` entity.
* **Cross-surface AI workflows.**
  ("if I get an email from John about
  X, create a task to review Y") is
  v2. v1 supports the explicit
  chat-panel intents only.
