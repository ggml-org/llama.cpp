# Tessera Productivity Surface — Phase 3: Chat Panel + Receipts Drawer

**Status:** Design v1.0
**Date:** 2026-08-05
**Companion:** `docs/tessera-productivity-design.md` §6 (chat panel) + §7 (receipts)
**Sister specs:** `docs/tessera-productivity-foundations-design.md`, `docs/tessera-productivity-editor-design.md`
**Branch:** `feat/prod-chat-receipts` (off `feat/prod-editor`)

---

## 1. Problem

The productivity surface needs two control surfaces that wrap the editor and the constitutional-receipt backbone: the **chat panel** (per-document command queue) and the **receipts drawer** (audit-trail inspector). The data layer, the block AST, the mutation engine, the receipt signer, the `ReceiptUndoManager`, the `DocumentStore`, the `TesseraTextContentManager` (Phase 2 editor), and the `ChatQueueItem` / `ChatQueue` data model (Phase 1) are all on the `feat/prod-editor` branch. The SwiftUI bindings, the cross-surface coordination, and the user-facing interactions are not.

This spec defines those bindings. It also commits to the load-bearing interactions from the architect's review (spec §6.5–§6.11, §7.3–§7.5): the two-cursor model, the "Hold your horses" rename, the match-and-supersede-on-every-front-of-queue-add rule, the per-document queue model, and the three-tab receipts drawer.

---

## 2. Why this design

The architectural commitments that drove the shape:

| Commitment | Why it shapes the design |
|---|---|
| Chat panel is a per-document **command queue**, not a chat history | The data model is a state machine per `ChatQueueItem` (`pending → inProgress → applied` / `failed`), not a flat message list. The view is a queue view, not a chat view. |
| Match-and-supersede runs on every front-of-queue add | The state machine has a hook for the LLM call; it's not a "fire-and-forget" enqueue. The LLM result is cached and the user can drag-overrule. |
| "Take over" is renamed to "Hold your horses" | The semantics shifted from a binary pause to a co-editing session between user and agent. The button is always present. The agent enters a dialog. |
| Receipts are first-class; the chat panel itself is receipted | The state machine persists the queue to the data layer's `chat_queues` table. Every chat-panel mutation produces a `Receipt` (e.g., the act of reordering the queue is a `chat_queue_reorder` receipt, the act of marking an item failed is a `chat_queue_failure` receipt). The audit trail covers the chat surface itself. |
| The receipt drawer has three tabs | "This document" / "All documents" / "Export" — fixed by architect decision in the spec. |
| The on-device LLM is the default for the match-and-supersede check | The `MatchAndSupersedeEngine` accepts the existing `LLMProvider` protocol and the `TesseraLLMProviderFactory` picks the on-device model by default. |
| Cmd-2 / Cmd-Option-2 are fixed shortcuts | macOS: Cmd-2 focuses the chat panel (does NOT toggle), Cmd-Option-2 toggles the receipts drawer. |
| Phase 3 wires the SwiftUI surface; Phase 4-6 wire the per-surface wrappers | The `ChatPanelView` and `ReceiptsDrawerView` are general-purpose. The `Documents` / `Notes` / `Code` surface wrappers (Phase 5) wrap them. |

The deep UX rationale — the two-cursor model, the typing-pattern choices, the "always present" hold-your-horses button, the cross-document chip — lives in `docs/tessera-productivity-ux-research.md`. This spec is the source of truth; the UX doc is the evidence.

---

## 3. The chat panel layout (per spec §6.1)

**macOS:** always-visible sidebar on the right side of the window. The `ContentView` becomes a three-column `NavigationSplitView`: surfaces | editor | chat. Cmd-2 focuses the chat panel (does not toggle visibility). Cmd-Option-2 toggles the receipts drawer (a separate right-side inspector pane inside the chat panel's column). Both shortcuts are wired through `.focusedSceneValue` so `View > Focus Chat` and `View > Toggle Receipts` reach them from any menu.

**iOS:** bottom tab. The chat panel is one tab; the editor is another. Tabs are persistent (the chat tab is always reachable).

The chat panel has three regions (top to bottom):

```
+-------------------------------------+
|  HEADER                             |  (document title, working-in-background
|  Doc Title     [↶] [↷]  5 receipts  |   chip, undo/redo, receipt count)
+-------------------------------------+
|                                     |
|  QUEUE                              |  (pending / in-progress / applied /
|  ⏱ summarize section 2 (pending)    |   failed / superseded)
|  ⏱ use latest finance data          |
|  ✓ 3 paragraphs updated (applied)   |  drag-to-reorder; tap to interact
|  ⚠️ agent run failed (failed)       |
|                                     |
+-------------------------------------+
|  INPUT                              |  (text field + "Hold your horses")
|  [type a command...]   [Hold]       |
+-------------------------------------+
```

The receipts drawer is a sibling inspector pane that lives next to the chat panel (on macOS) or as a modal sheet (on iOS). The chat panel and the drawer share a `ReceiptsCoordinator` for navigation (tapping a chip in the chat opens the corresponding receipt in the drawer; tapping "Show in chat" in the drawer scrolls the chat to the corresponding item).

---

## 4. The state machine (per spec §6.2)

The state machine wraps the Phase 1 `ChatQueue` data model and persists it to the data layer's `chat_queues` table on every transition. It's an `actor` so concurrent reads from the SwiftUI view layer and the agent's `agentLoop` don't race on the queue.

```swift
public actor ChatPanelStateMachine {
    public init(
        documentID: UUID,
        documentStore: DocumentStore,
        dataLayer: TesseraDataLayer,
        supersedeEngine: MatchAndSupersedeEngine
    )

    public var queue: ChatQueue { get async }
    public var holdMode: HoldMode { get async }

    public func enqueue(message: String, sourceMutation: Mutation?) async throws -> ChatQueueItem
    public func startNextPending() async throws -> ChatQueueItem?
    public func markInProgress(itemID: UUID) async throws
    public func markApplied(itemID: UUID, receipt: Receipt) async throws
    public func markFailed(itemID: UUID, error: Error) async throws
    public func supersede(oldItemID: UUID, by newItemID: UUID) async throws
    public func cancelInProgress() async throws
    public func reorder(itemID: UUID, toNewIndex: Int) async throws
    public func delete(itemID: UUID) async throws
    public func holdYourHorses() async throws
    public func resume() async throws
    public func agentContext(limit: Int) async throws -> AgentContext
}
```

**Lifecycle:**

1. `enqueue` is the only entry point for new items. It performs the match-and-supersede check, persists the queue, and returns the new item.
2. `startNextPending` is called by the agent when it's idle. It transitions the head of the queue from `pending` to `inProgress`. Returns nil when the queue is empty or paused.
3. `markApplied` / `markFailed` close the lifecycle.
4. `supersede` is called by the `MatchAndSupersedeEngine` on every front-of-queue add.
5. `holdYourHorses` / `resume` pause and resume the queue.
6. `agentContext` builds the prompt-time context the agent's `LLMProvider.complete(...)` call sees.

**Receipts for chat-panel mutations.** Every state-machine method that mutates the queue persists it. The act of persisting the queue is not itself a "document" receipt (the chain is the document's audit trail; the queue is a side-table). The receipt for the *content* the queue caused (e.g., the receipt produced when an `inProgress` item finishes) is the standard document receipt — already produced by `DocumentStore.apply(...)`. The chat-queue persistence is silent from the chain's perspective.

**Persistence round-trip.** Every method that mutates the queue calls `documentStore.saveChatQueue(...)` after the in-memory mutation. The `enqueue` path also does the match-and-supersede check before the save. On document open, the state machine loads the queue from the data layer once (`loadChatQueue`).

**Idempotency.** The state machine is idempotent for repeated calls: re-enqueuing the same message is a no-op (the item is found by content hash). Re-marking applied is a no-op. The state machine never throws on a "wrong" transition; it logs and returns the current state.

**HoldMode.** The state machine has a `holdMode` enum that records the pause state: `running`, `holdRequested` (the dialog is up but the queue is still picking up the current in-progress item), `hold` (no new items will be picked up), and `resuming` (the user has clicked Resume and the agent is being told to pick up the new front).

**Cross-document continuity.** The `ChatPanelStateMachine` is per-document. The `CrossDocumentChatRegistry` (a separate `actor`) tracks the set of `ChatPanelStateMachine`s across all documents and exposes a `pauseAll` method for the "Working in background" chip. See §10.

---

## 5. Visual treatment per state (per spec §6.3)

Each `ChatQueueItem.State` has a visual treatment. The treatments are encoded in a single `ChatQueueItemStyle` enum (so they're unit-testable without a SwiftUI view tree) and the SwiftUI view consumes the style.

| State | Typography | Background | Icon | Animations |
|---|---|---|---|---|
| `pending` | italic, 60% opacity | none | `clock` (sf symbol) | none |
| `inProgress` | regular | subtle highlight (yellow 5% blend) | `circle.dotted` with `thinkingPulse(isActive: true)` | the thinking-pulse from Phase 2's animation primitives; "Hold your horses" button visible |
| `applied` | regular | none | `checkmark.circle.fill` (green) | none — receipt chip rendered inline |
| `failed` | regular | red flash (250ms ease-out) | `exclamationmark.triangle.fill` (red) | the red-flash animation; retry button visible |
| `superseded` | regular, 50% opacity | none | original icon dimmed; "replaces #N" badge | none |

The row is interactive: tap to edit (pending), tap to open receipt (applied), tap to retry (failed), tap to expand (superseded — shows the original + the supersession note). Drag-to-reorder is enabled on all non-applied rows (the chat panel can't reorder applied items because they're in the audit trail).

**Agent cursor highlight.** When an item is `inProgress`, the editor highlights the affected block(s) — the row fires a "highlight blocks" event that the editor consumes (the `TesseraTextContentManager` paints a subtle blue background on the block). The agent cursor (the small robot icon from `AgentCursorOverlay`, Phase 2) is positioned at the agent's current edit location.

---

## 6. The "hard suggestion" semantics (per spec §6.4)

When the user types a message and hits return:

1. The text becomes a `ChatQueueItem` with `state = .pending` and `order = 0`.
2. The match-and-supersede check runs (see §8).
3. The new item is inserted at the front of the queue.
4. The `agentContext` (the agent's prompt-time view) is rebuilt: `pending` includes the new item, `recentReceipts` includes any new receipts produced since the last context build.
5. The agent picks up the new front item when idle.

The `AgentContext` is the data the agent sees:

```swift
public struct AgentContext: Codable, Sendable, Hashable {
    public let documentID: UUID
    public let pending: [ChatQueueItem]    // newest first
    public let recentReceipts: [Receipt]   // last N, in chain order (oldest first)
    public let documentAST: DocumentAST
    public let builtAt: Date

    public init(
        documentID: UUID,
        pending: [ChatQueueItem],
        recentReceipts: [Receipt],
        documentAST: DocumentAST,
        builtAt: Date = Date()
    )
}
```

The agent's `LLMProvider.complete(...)` is called with a system prompt that serializes this context (e.g., "The user has 3 pending instructions: ...; the last 5 receipts were: ..."). The agent's response is parsed into tool calls (mutations).

---

## 7. The two-cursor model in the chat panel (per spec §6.5)

The chat panel tracks the user's chat input cursor (their text caret in the input field) and the agent's edit cursor (the agent's current `NSTextLocation` in the editor). The two cursors are independent:

- **User chat cursor** — standard text caret in the input field, managed by SwiftUI's `TextField`.
- **Agent edit cursor** — read from the editor's `EditorCursorState.agentCursor` (Phase 2). The chat panel's "in-progress" item displays a small "Agent is editing paragraph 3" caption with a tiny live preview of the affected block.

When the user clicks in the document, the editor updates `EditorCursorState.userCursor`; the chat panel doesn't react (the user is interacting with the editor, not the chat).

When the user adds a pending message, the new item goes to the front of the queue. The agent's in-flight work is re-prioritized: the agent finishes the current mutation if the user's edit doesn't affect the affected blocks, or rolls back and re-plans if it does. The "steering" behavior is the agent's responsibility; the state machine just exposes the new front.

---

## 8. Match-and-supersede (per spec §6.7)

When a new item is added to the front of the queue, the state machine calls the `MatchAndSupersedeEngine`:

```swift
public actor MatchAndSupersedeEngine {
    public init(llmProvider: TesseraLLMProvider, decisionCache: SupersedeDecisionCache? = nil)

    public func evaluate(
        newFront: ChatQueueItem,
        existingQueue: [ChatQueueItem]
    ) async throws -> SupersedeDecision
}

public struct SupersedeDecision: Codable, Sendable, Hashable {
    public var supersededItemIDs: [UUID]
    public var reasoning: String
}
```

The LLM call is a single prompt: "Given the new instruction X and the existing queue items [Y, Z, W], does X supersede any of Y, Z, W? If so, which? Respond with JSON." The on-device model is the default (`TesseraLLMProviderFactory.makeFromSettings()`); the engine falls back to a heuristic (lexical similarity) when the LLM is unavailable, so the chat panel still works in the empty-library case.

The result is cached: the `SupersedeDecisionCache` is a small in-memory `[UUID: SupersedeDecision]` keyed by the new-front item's id. Repeated enqueues of the same content skip the LLM call.

The state machine applies the decision by calling `supersede(oldItemID:by:)` for each superseded id. The user can drag-overrule the supersession (the new item's supersession of older items is reversed on drag).

---

## 9. "Hold your horses" (per spec §6.8)

The "Hold your horses" button is always present in the chat panel footer. Clicking it:

1. Sets `holdMode = .holdRequested`.
2. The agent's `startNextPending` returns nil for any subsequent call.
3. The chat panel shows a dialog (a SwiftUI `.sheet` or `.popover` on macOS, a `.sheet` on iOS) titled "Hold your horses" with the message "Is something wrong? Would you like me to reframe and approach things differently?"
4. The user can type a response; the response is enqueued as a pending item with `actor: .user(...)` and a special `holdResponse` flag.
5. The user can drag pending items to reorder (the agent can suggest reorderings via the LLM; the suggestion is added as a pending item with a `[suggested]` prefix in the message).
6. The user clicks "Resume" to set `holdMode = .resuming` then `.running`. The agent picks up the new front item.

The button's color is the system "pause" orange (`Color.orange` on macOS / iOS). While paused, the label changes to "Resume" and the chat panel gets a subtle paused-indicator stripe (a 4-pt orange bar at the top, animated in with `agentPausedBanner` from Phase 2's animation primitives).

---

## 10. Cross-document behavior (per spec §6.9)

Each document has its own `ChatPanelStateMachine` (per-document queue model, architect-confirmed). When the user switches from doc A to doc B, the agent's in-flight edit on doc A continues in the background. The chat panel of doc A shows a "Working in background" chip when the user switches back:

> Agent is editing 'Doc B' — [Switch to Doc B] [Pause all]

The chip is rendered by the `CrossDocumentChatRegistry` (a separate `actor`) that tracks the set of active `ChatPanelStateMachine`s across all documents. The registry exposes:

```swift
public actor CrossDocumentChatRegistry {
    public init()

    public func register(_ machine: ChatPanelStateMachine, for documentID: UUID, title: String)
    public func unregister(documentID: UUID)
    public func activeDocuments() async -> [ActiveDocumentInfo]
    public func pauseAll() async
}

public struct ActiveDocumentInfo: Sendable, Hashable {
    public let documentID: UUID
    public let title: String
    public let inFlightItemCount: Int
    public let isCurrent: Bool
}
```

"Pause all" calls `holdYourHorses()` on every registered state machine. The receipt chain serializes the work — even if two agent runs are happening on two docs, their receipts land in order, no conflicts.

---

## 11. Drag-to-reorder (per spec §6.10)

Pending items are draggable. On macOS, click-and-drag. On iOS, long-press to lift, then drag. VoiceOver rotor on both platforms (the `.accessibilityRotor("Pending items")` API on macOS, the `.accessibilityRotorEntry` on iOS).

The reorder is wired through SwiftUI's `.onDrag` / `.dropDestination` (or `.draggable` / `.dropDestination` for iOS 16+). The reorder updates the agent's context window (re-ordering the pending list in `AgentContext`).

Reordering during a "Hold your horses" pause is the primary way the user and agent co-edit the queue. The reorder is receipted: the state machine's `reorder(itemID:toNewIndex:)` method writes a `chat_queue_reorder` receipt to a separate audit table (`chat_queue_audit`, see Phase 1's migration `0002_productivity_receipts.sql`).

---

## 12. The receipt drawer (per spec §7.3, architect decision)

**macOS:** right-side inspector pane in `NavigationSplitView`. Always available; Cmd-Option-2 toggles; tapping a receipt chip in the chat panel opens that receipt in the drawer.

**iOS:** modal sheet with `.large` detent. Tap a chip to present the sheet.

The drawer has three tabs:

| Tab | Content |
|---|---|
| **This document** | The receipt chain for the open document, newest first. Each row: actor icon, timestamp, summary. Tapping a row opens the receipt detail. |
| **All documents** | The same chain view but across all documents, filterable by date (last 24h / 7d / 30d / all), by actor (user / agent), and by document title. |
| **Export** | The export UI (see §14). |

The drawer is a `View` that takes the document id, the data layer, the document store, and a `ReceiptsCoordinator` (for the cross-surface nav). The three tabs are themselves sub-views.

---

## 13. The receipt details view (per spec §7.4)

When the user taps a receipt in the drawer, the drawer shows the receipt detail (in a `NavigationLink` push on macOS, in a `NavigationStack` push on iOS). The detail view has five sections:

1. **Header** — actor (with model + prompt hash if agent), timestamp, receipt id.
2. **Mutations** — a list of the typed `Mutation` operations, each expandable to show its fields.
3. **Diff** — a before/after of the affected blocks, rendered as text with red strikethrough for deletions and green underline for additions. Uses the `preMutationSnapshot` field on the receipt (the receipt is self-contained for undo).
4. **Signature** — the ed25519 signature as a hex string, with a "Verify" button that re-runs the signature check against the embedded public key (from `ReceiptSigner.publicKey`). The result is shown inline (valid / invalid / voided).
5. **C2PA** — the manifest summary (assertions list), with a "View C2PA manifest" button that opens the full JSON in a sheet.

The detail view is built without a separate "show diff" toggle — the diff is always rendered for applied receipts. For receipts whose mutations don't touch block content (e.g., `setDocumentTitle`), the diff section shows "No content changes."

---

## 14. Receipt export (per spec §7.5)

The user can export the receipt chain as:

- **Signed JSON bundle** (default) — the full chain as a single JSON file, with the ed25519 signatures and C2PA manifests inline. Filename: `<document-name>-audit-<date>.json`.
- **Markdown summary** (opt-in) — human-readable Markdown file. Filename: `<document-name>-audit-<date>.md`.
- **C2PA-signed document** — the document itself, signed with the C2PA manifest embedded. Filename: `<document-name>-c2pa.<ext>` where `<ext>` is the document's export format (we use `.txt` for v1 — the document AST serialized to text).

The export goes through the `ReceiptExportService`, which:

1. Builds the bundle (JSON, Markdown, or C2PA-embedded).
2. Asks the user to confirm (a SwiftUI `.alert` or `.confirmationDialog`).
3. Saves the bundle to the data layer's `export_artifacts` table (so the export is durably logged).
4. Writes a `Receipt` with `receiptType: "export"` to the chain.
5. Returns the bundle (the SwiftUI view shows an "Export saved" toast and an "Open in Finder" button).

The egress policy is governed by the existing `TesseraEgressGuard`. The guard is invoked once per export: if the guard denies, the export is blocked and the user is told why. v1 denies exports to any location outside the encrypted volume; the guard can be extended later for opt-in remote destinations (iCloud Drive, local file picker, etc.).

```swift
public struct ReceiptExportService: Sendable {
    public init(
        documentStore: DocumentStore,
        dataLayer: TesseraDataLayer,
        signer: ReceiptSigner,
        egressGuard: TesseraEgressGuard
    )

    public func export(
        documentID: UUID,
        format: ReceiptExportFormat,
        userConfirmed: Bool
    ) async throws -> ExportArtifact
}

public enum ReceiptExportFormat: String, CaseIterable, Sendable, Codable {
    case signedJSON
    case markdown
    case c2paDocument
}

public struct ExportArtifact: Codable, Sendable, Hashable {
    public let id: UUID
    public let format: ReceiptExportFormat
    public let filename: String
    public let payload: Data
    public let receiptID: UUID
}
```

The export receipt is signed with the same key as the document receipts (per spec §7.2.1) and the export is logged as a chain entry with `receiptType: "export"`. The audit trail of an exported document includes the export receipt.

---

## 15. Chat panel ↔ receipt drawer coordination

The two surfaces share a `ReceiptsCoordinator` (an `actor` for the cross-surface state and an `ObservableObject` view-model for SwiftUI binding):

```swift
public actor ReceiptsCoordinator {
    public init()

    public func openReceiptInDrawer(_ receiptID: UUID, fromChatItem itemID: UUID?) async
    public func showInChat(receiptID: UUID) async -> UUID?  // returns chat item id
    public func showInGraph(entityID: UUID) async            // Phase 6 hook
    public func currentFocus() async -> ReceiptsFocus
}

public enum ReceiptsFocus: Sendable, Hashable {
    case none
    case receipt(UUID)
    case graphEntity(UUID)
}
```

The `ChatPanelView` calls `openReceiptInDrawer` when the user taps a chip. The `ReceiptDetailView` calls `showInChat` when the user taps "Show in chat" (the chat panel scrolls to the corresponding item and highlights it). The `ReceiptsDrawerView` calls `showInGraph` for the "Show in graph" button (the Graph surface takes over; Phase 6 dependency).

The `ReceiptsCoordinator` is also a SwiftUI `EnvironmentObject` so the views can observe the focus and re-render.

---

## 16. Library survey

| Need | Library | Decision |
|---|---|---|
| SwiftUI chat UI | `SwiftUI` (stdlib) | **Adopt** — build it natively |
| Drag-to-reorder | `SwiftUI` `.onDrag` / `.dropDestination` | **Adopt** — native APIs |
| Receipt drawer | `SwiftUI` `NavigationSplitView` (macOS), `NavigationStack` (iOS) | **Adopt** |
| C2PA manifest viewer | none | **Build** — render the JSON manifest in a sheet |
| Markdown rendering (for export preview) | `MarkdownUI` (gonzalezreal) | **Adopt** — already in Phase 2 |
| Export bundle builder | stdlib `JSONEncoder` + `Data` | **Adopt** |
| Egress gating | `TesseraEgressGuard` (existing on main) | **Adopt** — use the existing enum; v1 extension point for "save to encrypted volume" |
| Receipt signing | `ReceiptSigner` (Phase 1) | **Adopt** — same key path as document receipts |
| LLM for match-and-supersede | `LLMProvider` (existing) | **Adopt** — the on-device model is the default; falls back to a lexical heuristic |

**MarkdownUI** is already a Phase 2 dependency (it was added for the export preview). We use the same dep here for the markdown export preview.

**TesseraEgressGuard** is a static filter for training data. We extend it (via the `ReceiptExportService`) with a higher-level "user-approved export" gate: the user has to confirm via a SwiftUI dialog, the confirmation is recorded, and the export is logged as a receipt. The existing `TesseraEgressGuard` is consulted for any data that would leave the encrypted volume.

---

## 17. Test strategy

The Phase 3 tests live in `TesseraStudio/Tests/TesseraCoreTests/Productivity/Chat/` and `.../Productivity/Receipts/`. The 746 existing tests (Phase 2 baseline) must stay green.

**Unit tests:**

- `ChatPanelStateMachineTests` — state transitions, persistence, hold/resume, reorder, idempotency.
- `MatchAndSupersedeEngineTests` — LLM call (mocked), heuristic fallback, decision cache.
- `ChatQueueItemStyleTests` — the per-state visual treatment (the `ChatQueueItemStyle` enum is testable without a view tree).
- `CrossDocumentChatRegistryTests` — register/unregister, pauseAll, active documents.
- `ReceiptExportServiceTests` — JSON / Markdown / C2PA bundle output, user-confirmation gate, export-receipt logging.
- `ReceiptsCoordinatorTests` — openReceiptInDrawer, showInChat, showInGraph, currentFocus.

**View tests (snapshot-style with a SwiftUI renderer):**

- `ChatPanelViewTests` — the three regions render, the queue list reflects the state, the input field is wired.
- `ReceiptsDrawerViewTests` — the three tabs render, the receipt chain is in order, the all-documents tab is filterable.
- `ReceiptDetailViewTests` — the five sections render, the verify button runs the signature check.
- `HoldYourHorsesViewTests` — the dialog is shown on hold, the resume button is wired.

**Integration tests:**

- The state machine + the receipt chain — every `markApplied` writes a receipt to the chain; the chain is queryable by document id; the queue item's `producedReceiptID` matches the chain's last entry.
- Persistence round-trip — enqueue 1000 items, reload the queue from the data layer, assert the order and the state. The performance target is <50ms for 1000 items.

**Property tests:**

- The match-and-supersede decision is monotonic: re-evaluating with the same input returns the same decision (idempotent).
- Reorder is a permutation: the queue's items are unchanged in count and identity; only the order field differs.

**Performance tests:**

- 1000-item queue persistence round-trip <50ms.
- 1000-item chain export to JSON <200ms.
- 1000-item chain export to Markdown <500ms.

---

## 18. Out of scope

- **Phase 4:** importers / exporters (DOCX, XLSX, PPTX, PDF) — already on a separate branch.
- **Phase 5:** per-Materials-surface wrappers (Documents / Notes / Code) — the chat panel and drawer are general-purpose; Phase 5 wraps them with surface-specific toolbar, sidebar, and theme.
- **Phase 6:** Contacts + Graph viz — the `ReceiptsCoordinator.showInGraph` is a hook; the Graph view itself is a later phase.
- **Multi-device receipt sync (v2)** — the per-device signing key (spec §7.2.1) means receipts can't be verified on a different device. v2 may add a multi-device key-sync story.
- **Real-time collaboration (v2)** — the chat panel is single-user; v2 may add a multi-user model.
- **Per-receipt C2PA viewer** — the detail view shows the manifest as JSON. A full C2PA-aware viewer (with the spec's `c2patool` parity) is v2.
- **Inline LLM for chat replies** — the chat panel is a *command queue* (spec §6), not a chat history. The user types commands; the agent executes them. There is no "agent reply" text in the queue.

---

## Appendix A: API surface (consolidated)

```swift
// Core: state machine + engines
public actor ChatPanelStateMachine
public actor MatchAndSupersedeEngine
public actor CrossDocumentChatRegistry
public struct AgentContext
public struct SupersedeDecision
public struct ActiveDocumentInfo

// Core: export
public struct ReceiptExportService
public enum ReceiptExportFormat
public struct ExportArtifact

// Core: coordination
public actor ReceiptsCoordinator
public enum ReceiptsFocus

// Core: visual treatment (testable)
public enum ChatQueueItemStyle
public struct ChatQueueItemDisplay  // row data (state + label + icon + summary)

// Mac: views
public struct ChatPanelView
public struct ReceiptsDrawerView
public struct ReceiptDetailView
public struct ReceiptExportView
public struct HoldYourHorsesDialog
public struct ReceiptRowView
public struct ReceiptDiffView
public struct ReceiptSignatureSection
public struct C2PAManifestSheet

// iOS: views
public struct ChatPanelView_iOS
public struct ReceiptsDrawerSheet_iOS
```
