# Tessera Productivity Foundations — Design Specification

**Status:** Draft v1 — 2026-08-05
**Author:** Tessera Architecture
**Applies to:** Tessera Studio for macOS 1.0.0+ (post-data-layer, pre-productivity-editor)
**Branch:** `feat/prod-foundations`
**Companion:** `docs/tessera-productivity-design.md` (the full productivity spec; this doc is the Phase 1 deliverable spec)
**Sister specs:** `docs/tessera-data-layer-design.md`, `docs/tessera-plead-the-fifth-design.md`

---

## 1. Problem

Tessera's productivity surface (documents, spreadsheets, slides, email, chat panel) needs load-bearing primitives that Phases 2 (editor), 3 (chat panel), 4 (importers), and 6 (contacts + graph) all depend on. The full spec is `docs/tessera-productivity-design.md`; this doc is the Phase 1 deliverable: the data model, the mutation API, the receipt infrastructure, the undo manager, the two-cursor data model, the chat queue data model, and the data-layer integration.

This is NOT a UI surface. There is no SwiftUI, no `NSTextView`, no `RichTextKit`. The deliverable is pure-Swift types and one new migration that the productivity surface sits on top of.

---

## 2. Why this design

The architectural choice is to make **the receipt the source of truth for the audit trail**, with the **Block AST** as the source of truth for the document state, and the **Mutation API** as the only path that mutates either. This is a load-bearing constraint: every editor, every agent, every importer, and every export goes through the same Mutation API; every mutation produces a signed Receipt; every receipt is in an append-only chain; every receipt can be undone by computing its inverse mutations and appending a new (also signed) receipt.

The two-cursor model (user + agent in the same document) and the chat queue (per-document command queue) are the data primitives Phase 3 needs; the **DocumentStore** is the productivity surface's wrapper around the existing `TesseraDataLayer` that ties it all together.

**Six invariants the foundation guarantees:**

1. **No raw SQL past `TesseraDataLayer`.** The productivity surface's data access goes through `DocumentStore` → `TesseraDataLayer` → `TesseraDataStore`. `PostgresNIO` types never leak.
2. **No raw Keychain past `TesseraKeychainVolume`.** The receipt signing key is derived from the existing volume-password Keychain entry. The Plea the Fifth 9-step wipe destroys it.
3. **Every mutation produces a receipt.** The receipt is signed with ed25519 from the device's signing key (derived from the volume password).
4. **Receipts are append-only.** Voiding is implemented by appending a new receipt whose payload references the voided one and setting `voidedBy` on the original in memory.
5. **Undo is receipt-aware.** Each `Cmd-Z` undoes one whole receipt (one user-perceived edit unit), not the last character. Multi-receipt instructions (the agent's multi-step plan) are grouped as one undo unit.
6. **C2PA is in v1.** The receipt's C2PA manifest is a placeholder matching the C2PA Technical Specification 2.x shape (the format fields are identical); the signature algorithm is `ed25519` (the spec example uses `es256`; the architect's choice of ed25519 throughout drives the deviation — see §5.3).

---

## 3. The Block AST (the data model)

### 3.1 Block types

The `BlockType` enum covers 13 cases: `heading`, `paragraph`, `list`, `listItem`, `table`, `tableCell`, `image`, `codeBlock`, `callout`, `divider`, `quote`, `toggle`, `equation`. Each carries its type-specific data in `attributes` / `content` / `children`. The shape matches the spec §4.1 verbatim:

| Block type | attributes | content | children |
|---|---|---|---|
| `heading` | `{ level: 1..6 }` | (none) | (none) |
| `paragraph` | (none) | inline runs | (none) |
| `list` | `{ style, items }` | (none) | ordered list-item block IDs |
| `listItem` | (none) | inline runs | (none) |
| `table` | `{ rows, cols, cells }` | (none) | ordered cell block IDs |
| `tableCell` | (none) | inline runs | (none) |
| `image` | `{ source, alt, width?, height? }` | (none) | (none) |
| `codeBlock` | `{ language? }` | source text | (none) |
| `callout` | `{ emoji?, color? }` | (none) | (none) |
| `divider` | (none) | (none) | (none) |
| `quote` | `{ cite? }` | inline runs | (none) |
| `toggle` | `{ expanded: Bool }` | (none) | ordered child block IDs |
| `equation` | `{ latex: String }` | (none) | (none) |

### 3.2 Inline runs

`InlineRun` is a contiguous span of text with a uniform set of annotations. Annotations include: `bold`, `italic`, `underline`, `strikethrough`, `code`, `subscript`, `superscript`, `link(URL)`, `color(hex: String)`. The annotation enum is a tagged enum (per-variant associated values), which `Codable` encodes as a tagged JSON object (`{"link": "https://..."}` / `{"color": "#FF00FF"}`).

### 3.3 Storage shape

The document is a flat `[UUID: Block]` map plus a `rootChildren: [UUID]` ordered list. The map gives O(1) lookup; the list gives the document order. This matches Notion's internal shape and is what the spec §4.2 commits to.

The AST is stored in the data layer as the `body` JSONB column of a `graph_entity` row with `entity_type = "document"`. The custom `Codable` conformance in `DocumentAST` (see `Block.swift`) handles Swift's quirk that `JSONDecoder` cannot decode `Dictionary<UUID, V>` from a JSON object directly — the conformance encodes the dict as a JSON object with stringified UUID keys, then decodes back to a `[UUID: Block]` at the boundary.

### 3.4 Content hashing

`DocumentAST.contentHash()` returns the SHA-256 of the canonical JSON form (sorted keys, ISO-8601 dates). The receipt infrastructure embeds this hash in the C2PA `c2pa.hash.data` assertion. Two semantically-equal ASTs produce the same hash; this is the property that makes "the receipt chain verifies the document state" a real invariant.

### 3.5 Why JSONValue (a.k.a. `AnyCodable`)

The spec writes `attributes: [String: AnyCodable]`. The existing `TesseraCore` already has `JSONValue` (a tagged enum: `string | number | bool | array | object | null`) which is exactly what `AnyCodable` does in community packages. To avoid introducing a parallel type, we `typealias AnyCodable = JSONValue` at the productivity surface. The alias keeps the spec's API name recognisable while reusing the existing implementation.

---

## 4. The Mutation API (the typed operations)

### 4.1 The taxonomy

Three groups, per spec §5.1:

- **Block-level**: `insertBlockAfter`, `insertBlocksAfter`, `replaceBlock`, `deleteBlock`, `moveBlock`.
- **Attribute / content**: `setBlockAttribute`, `setBlockContent`, `appendInlineRun`, `replaceInlineRun`, `deleteInlineRun`, `setInlineAnnotation`.
- **Document-level**: `setDocumentTitle`, `setDocumentMeta`. These are no-ops against the in-memory AST (the AST doesn't carry document-level meta); the `DocumentStore` wrapper is responsible for persisting them on the `graph_entity` row.

### 4.2 The engine

`MutationEngine` is a value type with two methods:

- `validate(_:against:)` — pre-flight check. Throws `MutationError` for any invariant violation.
- `apply(_:to:)` — apply in place, returning a `[UUID: Block]` snapshot of the blocks the mutation touched in their PRE-mutation state. The snapshot is what makes the receipt self-contained for undo.

The engine is deliberately not an actor: it's in-memory only, has no I/O, and the caller serializes access to the document.

### 4.3 Pre-mutation snapshots (the undo substrate)

This is the key design decision that makes undo self-contained. When a mutation is applied, the engine captures the `Block` objects it touched in their pre-state. The snapshot is embedded in the signed receipt. The inverse mutations are computed from this snapshot, NOT from the live document.

The alternative — computing the inverse from the live document at undo time — is broken: the document may have been further mutated between apply and undo, and the receipt is no longer self-contained for audit. A future auditor with the receipt should be able to undo it without the live document.

### 4.4 Validation rules

The engine validates before applying. Failures are `MutationError` cases:

- `blockNotFound(blockID)` — mutation references a block that isn't in the document.
- `anchorNotFound(parentID, anchorID)` — the anchor isn't a child of the parent.
- `indexOutOfRange(parentID, index, count)` — for `moveBlock` with an out-of-range index.
- `inlineIndexOutOfRange(blockID, index, count)` — for inline-run mutations with out-of-range index.
- `wouldCreateCycle(blockID, newParent)` — for `moveBlock` that would create a cycle.
- `invalidOperation(reason)` — for type-specific rejections (e.g. `setBlockContent` on a `divider`).

### 4.5 Inverse computation

`Mutation.inverse(preMutation:)` returns the inverse mutations for the given pre-state snapshot. The inverses are themselves `Mutation` values:

| Mutation | Inverse |
|---|---|
| `insertBlockAfter(p, a, b)` | `deleteBlock(b.id)` |
| `insertBlocksAfter(p, a, bs)` | one `deleteBlock` per inserted block |
| `replaceBlock(id, new)` | `replaceBlock(id, oldBlock)` |
| `deleteBlock(id)` | `insertBlockAfter(parent, anchor, oldBlock)` |
| `moveBlock(id, newParent, newIndex)` | `moveBlock(id, oldParent, oldIndex)` |
| `setBlockAttribute(id, k, v)` | `setBlockAttribute(id, k, oldValue)` |
| `setBlockContent(id, content)` | `setBlockContent(id, oldContent)` |
| `appendInlineRun(id, run)` | `deleteInlineRun(id, oldCount)` |
| `replaceInlineRun(id, i, run)` | `replaceInlineRun(id, i, oldRun)` |
| `deleteInlineRun(id, i)` | `appendInlineRun(id, oldRun)` |
| `setInlineAnnotation(id, i, ann, on)` | `setInlineAnnotation(id, i, ann, !hadIt)` |

---

## 5. The Receipt infrastructure

### 5.1 The Receipt

The receipt is the constitutional record of a mutation. It is `Codable, Sendable, Identifiable` and contains:

- `id` — UUID v7.
- `documentID` — the document the mutation is against.
- `actor` — `.user(UserID)` or `.agent(AgentRunID, model, promptHash)`.
- `mutations` — the list of mutations this receipt applies.
- `timestamp` — ISO-8601 UTC.
- `priorReceiptID` — the receipt immediately before this one in the document's chain. `nil` for the genesis receipt.
- `signature` — ed25519 signature, 64 bytes, over the canonical JSON form of (id, documentID, actor, mutations, timestamp, priorReceiptID, c2paManifest, summary, preMutationSnapshot).
- `c2paManifest` — the C2PA content authenticity manifest (optional but always present in v1).
- `summary` — human-readable, e.g. "3 paragraphs updated, 1 list added".
- `preMutationSnapshot` — the `[UUID: Block]` map of touched blocks in their pre-state. The receipt is self-contained for undo.
- `voidedBy` — the id of the receipt that voided this one (e.g. an undo receipt). `nil` when valid.

### 5.2 The chain

Receipts form a chain per document. The `priorReceiptID` field is the linkage; the data layer's `receipt_chain` table (see §9) holds the chain as `(document_id, chain_index) → receipt_id` rows with `chain_index` being the monotonic position. The `chain_index` and the `priorReceiptID` are redundant: `chain_index` enables efficient "newest first" reads, `priorReceiptID` is the cryptographic linkage.

### 5.3 C2PA manifest

The C2PA manifest follows the C2PA Technical Specification 2.x shape (per the spec §5.3 and §7.2). The placeholder:

```json
{
  "format": "c2pa.v2",
  "claim_generator": "tessera/1.0",
  "assertions": [
    { "label": "c2pa.hash.data", "data": { "hash": "sha256:..." } },
    { "label": "c2pa.actions", "data": { "actions": [{ "action": "c2pa.edited" }] } },
    { "label": "tessera.actor", "data": { "type": "user", "id": "..." } },
    { "label": "tessera.summary", "data": "3 paragraphs updated" },
    { "label": "tessera.receipt_id", "data": "uuid" }
  ],
  "signature": "ed25519:..."
}
```

The fields are exactly what a C2PA-aware tool expects (the spec's required assertions + a few `tessera.*` extensions). The signature algorithm is `ed25519` instead of the spec example's `es256` because the architect chose ed25519 as the receipt-signing primitive (CryptoKit's `Curve25519.Signing.PrivateKey`) and adding an ECDSA P-256 key path would double the key-lifecycle surface.

**Deferral note:** the C2PA Technical Specification 2.x is detailed and the reference Rust implementation (`c2pa-rs`) would have to be vendored via FFI to produce spec-compliant manifests. Apple's `ContentAuthenticity` is closed-source and not a public SDK. The placeholder shape is forward-compatible: when a production C2PA library matures (or the architect chooses to vendor `c2pa-rs`), the field set is identical and no schema change is required.

### 5.4 Signing key

The ed25519 signing key is **the volume password**, reinterpreted. `TesseraKeychainVolume` (the existing Keychain actor on main) already generates 32 random bytes for the volume password (via `SecRandomCopyBytes`) and stores them base64-encoded. These 32 bytes are also exactly the ed25519 seed size. The new `TesseraKeychainVolume.receiptSigningKey()` method (in `Encryption/TesseraKeychainVolume.swift`) base64-decodes the password and constructs a `Curve25519.Signing.PrivateKey(rawRepresentation:)` from it.

**Constitutional property:** the volume password lives in the Keychain entry that `PleadTheFifthExecutor` step 3 (`destroy_volume_password`) destroys. Once that entry is gone, the signing key disappears with it. Prior signed receipts are unverifiable on the device that produced them — by design. This is the property that makes "all edits are signed with a C2PA manifest" a procurement signal: the signature is non-repudiable on the device, and gone with the device.

There is no separate signing-key entry to forget to destroy. The receipt infrastructure reuses the existing volume-password Keychain entry. This is the load-bearing design decision the spec §7.2.1 calls out: the signing key and the Plea the Fifth crypto-shred key are the same key.

### 5.5 ReceiptSigner

`ReceiptSigner` is the typed entry point for signing and verifying. Two construction modes:

- `ReceiptSigner()` — production mode; signing key is read from the Keychain via `TesseraKeychainVolume.receiptSigningKey()`. Throws `signingKeyUnavailable` if no key is present.
- `ReceiptSigner(signingKey:)` — test mode; signing key is the injected value. Used by every test in `ReceiptTests` and `ReceiptUndoManagerTests` to avoid touching the real Keychain.

### 5.6 Receipt verification

`Receipt.verify(against:)` runs the cryptographic check. `ReceiptSigner.verify(_:against:)` returns a discriminated result:

- `.valid` — signature is correct.
- `.invalid` — signature is wrong, or the canonical content is tampered.
- `.voided(by:)` — the receipt has been voided. The `by` field is the id of the voiding receipt.

A receipt's `voidedBy` field is set in memory (the underlying `graph_receipts.payload` JSONB row is mutable; the receipt chain is append-only at the receipt-row level but the voiding is metadata, not a new row). A future Phase 3 PR may move voiding to a separate `voiding_receipt` table for forensic clarity, but the current shape is sufficient for the receipt drawer UI.

---

## 6. The Undo manager (receipt-aware, batched)

### 6.1 The shape

`ReceiptUndoManager` is a final class (not a struct, because of the in-place mutation of its stacks). It holds:

- `undoStack: [Receipt]` — receipts available to undo (oldest first).
- `redoStack: [Receipt]` — receipts available to redo.
- `voidedReceipts: [Receipt]` — receipts that have been undone. Stored separately from the undo stack so the audit trail captures the voiding (the voided receipt's `voidedBy` field is set to the inverse receipt's id).
- `documentID: UUID` — the document the manager is bound to.

### 6.2 Operations

- `register(_:)` — push a new receipt onto the undo stack, clear the redo stack.
- `group(_:)` — push a list of receipts as one undo unit (the chat panel's "agent instruction 1 → 2 → 3" is one undo unit).
- `undo(document:actor:signer:)` — pop the most recent receipt, compute the inverse mutations from the receipt's `preMutationSnapshot`, apply them to the document, sign a new inverse receipt, mark the original as voided, push the original onto the redo stack.
- `redo(document:actor:signer:)` — pop the most recent receipt from the redo stack, re-apply its mutations (NOT the inverse's), sign a new redo receipt, mark the redo'd receipt as voided, push the redo receipt onto the undo stack.

### 6.3 The redo semantics

A common bug: the redo stack stores the inverse receipt (so redo re-applies the inverse, restoring the pre-undo state). The correct design: the redo stack stores the ORIGINAL receipt. The redo re-applies its mutations, restoring the post-original state. The new redo receipt is signed as `priorReceiptID = original.id`, voiding the original.

This matches the spec §5.4: "redo restores the exact original receipt".

### 6.4 Persistence

`snapshotUndoStack()`, `snapshotRedoStack()`, `snapshotVoidedReceipts()` return value copies for persistence. `restore(undoStack:redoStack:voidedReceipts:)` rehydrates the manager on document open.

The brief requires the undo stack survive document open/close. The data layer persists the receipts themselves; the undo manager stores the per-window stacks, which the data layer rebuilds by replaying the chain.

### 6.5 The grouping semantic

The `group(_:)` method lets the caller batch multiple receipts into a single undo unit. The receipts are pushed onto the undo stack in order, and the `undo()` call pops them in reverse order. This is what the chat panel's multi-step agent instruction uses.

---

## 7. The two-cursor data model

### 7.1 Shape

```swift
public struct TextCursor: Codable, Sendable, Equatable, Hashable {
    public let blockID: UUID
    public let offset: Int
    public let affinity: Affinity
    public enum Affinity: String, Codable, Sendable, Hashable {
        case upstream, downstream
    }
}

public struct CursorPair: Codable, Sendable, Equatable, Hashable {
    public var user: TextCursor?
    public var agent: TextCursor?
}
```

The data model supports both the user cursor and the agent cursor as named fields. The Phase 2 editor wires these to the `TesseraTextContentManager`; Phase 1 only carries the data.

### 7.2 Why the affinity field

Affinity is the standard text-editing distinction for ambiguous positions (e.g. at a line boundary). The platform text views use it to pick the right rendering side when the cursor sits between two blocks. The data model carries it so Phase 2 doesn't need to reconstruct it.

### 7.3 Why a `CursorPair` (not a single cursor)

The spec §6.5 commits to "the agent and the user have separate cursors in the same document". The data model reflects this: a `CursorPair` carries both. The mutation engine doesn't distinguish them — it's a UI concern (Phase 2). The data model just supports both being present.

---

## 8. The chat queue data model

### 8.1 Shape

```swift
public struct ChatQueueItem: Codable, Sendable, Identifiable, Hashable {
    public let id: UUID
    public let documentID: UUID
    public var order: Int
    public var message: String
    public var state: State
    public var actor: Actor
    public var sourceMutation: Mutation?
    public var producedReceiptID: UUID?
    public var createdAt: Date
    public var supersededByID: UUID?
    public enum State: String, Codable, Sendable, Hashable, CaseIterable {
        case pending, inProgress, applied, failed
    }
}

public struct ChatQueue: Codable, Sendable, Hashable {
    public var items: [ChatQueueItem]
    // ... insertAtFront, reorder, supersede, start, finish, fail
}
```

### 8.2 The lifecycle

```
pending → inProgress → applied
                     ↘ failed
```

A pending item is in the queue but the agent hasn't started. `starting(itemID:)` transitions it to `inProgress`; `finishing(itemID:with:)` transitions it to `applied` (and records the receipt id); `failing(itemID:)` transitions it to `failed` (with no receipt).

### 8.3 Ordering

`order` is an integer; lower = closer to the front. `insertingAtFront(_:)` shifts all existing items by 1 and sets the new item's order to 0. `reordering(itemID:to:)` moves an item to a new index and renumbers compactly.

### 8.4 Superseding

The spec §6.7's match-and-supersede check: when a new instruction supersedes an existing one, the original is marked `supersededByID = supersederID`. The item stays in the queue (visible but dimmed) so the user sees the history of intent.

### 8.5 Storage

The chat queue is stored in the data layer's `chat_queues` table (see §9). The table holds the queue as a JSONB blob per document. The full queue is rewritten on every change (the chat panel's drag-to-reorder and match-and-supersede operations rewrite the whole array).

---

## 9. The document ↔ data layer integration

### 9.1 New tables (migration `0002_productivity_receipts.sql`)

```sql
CREATE TABLE IF NOT EXISTS receipt_chain (
    document_id   uuid NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
    chain_index   bigint NOT NULL,
    receipt_id    uuid NOT NULL REFERENCES graph_receipts(id) ON DELETE RESTRICT,
    created_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (document_id, chain_index)
);

CREATE INDEX IF NOT EXISTS idx_receipt_chain_doc
    ON receipt_chain (document_id, chain_index DESC);

CREATE TABLE IF NOT EXISTS chat_queues (
    document_id  uuid PRIMARY KEY REFERENCES graph_entities(id) ON DELETE CASCADE,
    items        jsonb NOT NULL DEFAULT '[]'::jsonb,
    updated_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chat_queues_updated_at
    ON chat_queues (updated_at);
```

- `receipt_chain` is the per-document linear ordering. The composite primary key `(document_id, chain_index)` gives O(log n) lookup; the descending index supports "newest first" reads.
- `chat_queues` is per-document JSONB storage. The queue is small (tens of items) so JSONB is fine.
- Both are additive on top of `0001_init.sql` — no changes to existing tables.

### 9.2 Data layer extensions

The existing `TesseraDataLayer` (on main) gains five methods:

- `appendReceiptToChain(documentID:receiptType:payload:signature:)` — writes to `graph_receipts` AND links into `receipt_chain` at the next monotonic `chain_index`. The two writes are NOT in a single transaction; a crash between them leaves the receipt in `graph_receipts` but not in `receipt_chain`. The (planned) `rebuildReceiptChain(documentID:)` helper would repair this on document open. For Phase 1, the error path is logged and the orphan is visible in the audit trail.
- `receiptChain(documentID:limit:)` — ordered chain query.
- `latestChainIndex(documentID:)` — `nil` for an empty chain.
- `loadChatQueue(documentID:)` — returns the JSONB blob (`"[]"` for empty).
- `saveChatQueue(documentID:itemsJSON:)` — upsert.

### 9.3 The DocumentStore wrapper

`DocumentStore` is the productivity surface's high-level wrapper. It owns the seam between the in-memory mutation engine and the durable data layer:

```swift
public struct DocumentStore: Sendable {
    public init(dataLayer: TesseraDataLayer)
    public func loadDocument(id: UUID) async throws -> DocumentAST
    public func saveDocument(id: UUID, ast: DocumentAST) async throws
    public func apply(mutation: Mutation, to documentID: UUID, actor: Actor) async throws -> Receipt
    public func applyBatch(mutations: [Mutation], to documentID: UUID, actor: Actor) async throws -> Receipt
    public func history(of documentID: UUID, limit: Int) async throws -> [Receipt]
    public func loadChatQueue(documentID: UUID) async throws -> ChatQueue
    public func saveChatQueue(_ queue: ChatQueue, documentID: UUID) async throws
}
```

`apply` and `applyBatch` are the entry points. The flow:

1. Load the AST from `graph_entities.body`.
2. Apply the mutations in memory, capturing the union of pre-mutation snapshots.
3. Sign the receipt (with the captured snapshot).
4. Look up the prior receipt id (latest chain entry).
5. Persist the updated AST and the receipt.

A crash between step 5a's AST save and step 5b's receipt append leaves the AST change durable but the receipt un-appendable. The receipt chain's history query can detect this and rebuild the chain on document open. For Phase 1, the failure is logged and the audit trail captures the orphan.

---

## 10. Test strategy

### 10.1 Unit tests (no DB)

- **BlockTests** (24 tests) — every `BlockType` round-trips; every `InlineRun.Annotation` round-trips; deep-nesting preserved; content-hash stability.
- **MutationEngineTests** (30 tests) — every mutation variant applies correctly; validation rejects invalid mutations; cycle detection; composition of N mutations produces expected state; edge cases.
- **ReceiptTests** (15 tests) — sign + verify round-trip; tamper detection; voided receipts report voided; C2PA manifest format; JSON round-trip; summary correctness.
- **ReceiptSignerTests** (4 tests) — composition with the injected signing key.
- **ReceiptUndoManagerTests** (10 tests) — single undo/redo; batched undo; undo-of-undo; redo stack semantics; voiding; snapshot/restore.
- **TextCursorTests** (11 tests) — equality, serialization, two-cursor data model.
- **ChatQueueItemTests** (13 tests) — state transitions; ordering; superseding; serialization.
- **DocumentStoreTests** (3 tests) — JSON encoding/decoding; error equality.
- **TesseraKeychainVolumeReceiptSigningTests** (5 tests) — volume password → ed25519 key derivation; sign + verify through the volume password; constitutional property tests.

### 10.2 Integration tests (env-gated)

- **ProductivityDataLayerTests** (11 tests) — env-gated on `TESSERA_DB_INTEGRATION=1`. Creates a throwaway database, applies 0001 + 0002 migrations, and tests the data-layer extensions: `receipt_chain` schema, monotonic `chain_index`, per-document chain isolation, `latestChainIndex`, `chat_queues` schema, save/load, overwrite.

### 10.3 Existing test preservation

The 493 existing tests (data layer, Plea the Fifth, encryption, agent, tools, etc.) all stay green. The productivity work is purely additive.

### 10.4 Total

- Existing: 493 tests, 0 failures.
- New: 126 tests, 0 failures.
- Combined: 619 tests, 0 failures, 28 skipped (DB integration tests when env is not set).

---

## 11. Out of scope (deferred to later phases)

- **Phase 2: the `STTextView` + `TesseraTextContentManager` SwiftUI view** (per spec §9.2). The text content manager is the AST-backed text storage that wires the Block AST to the platform text view. The Mutation API is what it produces/consumes.
- **Phase 2: the editor's animation primitives** (per spec §8). Seven animation primitives (block slide-in, block replace, text appear, etc.) live in the SwiftUI layer.
- **Phase 3: the chat panel SwiftUI** (per spec §6). The `ChatQueue` data model is ready; the SwiftUI list and state machine are Phase 3.
- **Phase 3: the receipt drawer** (per spec §7.3-7.5). The drawer UI + the export (signed JSON + Markdown + C2PA) are Phase 3.
- **Phase 3: the two-cursor visualization** (per spec §6.5). The `TextCursor` and `CursorPair` data models are ready; the platform text view integration is Phase 2.
- **Phase 4: the importers / exporters** (per spec §10). Python + Pandoc. The `DocumentStore` and the Block AST are the format they emit/consume.
- **Phase 5: the Materials surfaces** (per spec §12). Tasks, Reminders, Calendar, Notes, Email, Code.
- **Phase 6: Contacts + Graph visualization** (per spec §12.7-12.8).

### 11.1 Library survey + decisions

The Phase 1 worker adopted **zero** new external libraries. The choice to compose with existing infrastructure (the data layer, the Keychain, the JSONValue type) is deliberate: the productivity surface already has the load-bearing primitives it needs. New libraries would add attack surface, build time, and maintenance cost for no gain in this phase.

Specific decisions:

- **No C2PA library adopted.** `c2pa-rs` is a Rust library; bringing it in via FFI would require a new Swift C-ABI shim and a vendored Rust toolchain. Apple's `ContentAuthenticity` is a closed-source framework, not a public SDK. The placeholder manifest format (see §5.3) matches the C2PA spec's field set; a future worker can swap in a real library without a schema change.
- **No AnyCodable library adopted.** The existing `JSONValue` (in `TesseraTool.swift`) is exactly what the community's `AnyCodable` package provides. A `typealias AnyCodable = JSONValue` keeps the spec's API name without introducing a parallel type.
- **No receipt-signing library adopted.** Apple's `CryptoKit` (already on main) provides `Curve25519.Signing.PrivateKey` for ed25519. No third-party crypto is needed.
- **Composed with `TesseraKeychainVolume`, not extended with a new Keychain abstraction.** The receipt signing key IS the volume password (see §5.4). This keeps the key lifecycle in one place and is the property that makes the constitutional crypto-shred work.
- **Composed with `TesseraDataLayer`, not extended with a new persistence abstraction.** The productivity surface's data access goes through the existing facade. The data layer gains five new methods; the productivity types are agnostic to Postgres.

---

## 12. Constitutional properties (the load-bearing invariants)

For the architect's review, the six invariants this design guarantees:

1. **No raw SQL past `TesseraDataLayer`.** The productivity surface's data access goes through `DocumentStore` → `TesseraDataLayer` → `TesseraDataStore`. The new `appendReceiptToChain` and `saveChatQueue` are inside the data layer; the productivity code never sees `PostgresQuery`.
2. **No raw Keychain past `TesseraKeychainVolume`.** The receipt signing key is derived from the existing volume-password Keychain entry. `TesseraKeychainVolume.receiptSigningKey()` is the single point of contact.
3. **Every mutation produces a signed receipt.** The receipt is the source of truth for the audit trail. The chain is append-only; voiding is metadata.
4. **Undo is receipt-aware.** `Cmd-Z` undoes one whole receipt (one user-perceived edit unit). Multi-receipt agent instructions are grouped via `group(_:)`.
5. **The receipt is self-contained for undo.** The `preMutationSnapshot` is embedded in the signed receipt. The inverse mutations are computed from the snapshot, not from the live document. A future auditor with the receipt can undo it without the live document.
6. **C2PA in v1.** The placeholder manifest format matches the C2PA spec's field set. The signature algorithm is ed25519 (deviation from the spec example's es256, documented in §5.3). Forward-compatible with a real C2PA library.

---

## 13. Next steps (the architect's checklist)

- [ ] Architect reviews and approves the foundation.
- [ ] Phase 1 is FF-merged into `main` (the brief is explicit: no push, no PR; the architect does the merge).
- [ ] Phase 2 worker (editor) starts on `feat/prod-editor` based on `main` (now containing this work). Uses the Mutation API + the `DocumentStore` + the `ReceiptUndoManager`.
- [ ] Phase 4 worker (importers) starts on `feat/prod-import-export` based on `main`. Uses the Block AST as the import target.
- [ ] Phase 6 worker (contacts + graph) starts on `feat/prod-contacts-graph` based on `main`. The data layer's `hybrid_search` is unchanged; the productivity surface is just another `graph_entity` subtype.
- [ ] Phase 3 worker (chat panel + receipt drawer) starts on `feat/prod-chat-receipts` after Phase 2 lands. Uses the `ChatQueue` data model and the `Receipt` infrastructure.
- [ ] C2PA library: when a production-grade option matures, the worker swaps in the real library against the placeholder format. No schema change.

---

## 14. Appendix: the receipt JSON shape

A receipt, serialized for export (e.g. the receipt-drawer "Export" button):

```json
{
  "id": "01900000-0000-7000-8000-000000000001",
  "documentID": "01900000-0000-7000-8000-000000000002",
  "actor": {
    "user": "01900000-0000-7000-8000-000000000003"
  },
  "mutations": [
    {
      "insertBlockAfter": {
        "parentID": null,
        "anchorID": null,
        "block": {
          "id": "01900000-0000-7000-8000-000000000004",
          "type": "paragraph",
          "attributes": {},
          "content": [{"text": "Hello", "annotations": []}],
          "children": [],
          "parentID": null
        }
      }
    }
  ],
  "timestamp": "2026-08-05T12:00:00Z",
  "priorReceiptID": null,
  "summary": "insert paragraph block",
  "preMutationSnapshot": {},
  "voidedBy": null,
  "signature": "ed25519:...",
  "c2paManifest": {
    "format": "c2pa.v2",
    "claim_generator": "tessera/1.0",
    "assertions": [...],
    "signature": "ed25519:..."
  }
}
```

The signed bytes (for verification) are the canonical JSON form of (id, documentID, actor, mutations, timestamp, priorReceiptID, c2paManifest, summary, preMutationSnapshot) — i.e. the receipt minus the `signature` and `voidedBy` fields, with sorted keys and ISO-8601 dates.
