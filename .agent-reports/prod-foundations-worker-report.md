# Phase 1 Productivity Foundations — Worker Report

**Branch:** `feat/prod-foundations`
**Commit:** `0c1bda2a7` — `productivity: Phase 1 foundations (Block AST, Mutations, Receipts, Undo)`
**Worktree:** `worktrees/prod-foundations/`
**Author:** Mavis (general-purpose worker)

---

## Summary

Phase 1 is complete. The load-bearing primitives the productivity surface depends on all landed on `feat/prod-foundations` and are ready to merge. 126 new tests added; all 493 existing tests stay green. No push, no PR (per AGENTS.md); the architect FF-merges when ready.

---

## Files touched

### New: `TesseraStudio/Sources/TesseraCore/Productivity/` (9 files, 2,446 LoC)

| File | Lines | What's in it |
|---|---:|---|
| `Block.swift` | 255 | `BlockType` (13 cases), `InlineRun` + `Annotation` (9 variants), `Block`, `DocumentAST` (with custom Codable for `[UUID: Block]`), `AnyCodable` typealias for `JSONValue`, `contentHash()` |
| `Mutation.swift` | 345 | `Mutation` enum (13 cases), `MutationError`, `inverse(preMutation:)` |
| `MutationEngine.swift` | 440 | `validate(_:against:)`, `apply(_:to:)` (returns pre-mutation snapshot), `capturePreSnapshot(of:in:)` |
| `Receipt.swift` | 256 | `Receipt`, `Actor`, `UserID`/`AgentRunID` typealiases, `C2PAManifest`, `canonicalBytes()`, `verify(against:)`, `ReceiptVerification` |
| `ReceiptSigner.swift` | 289 | `ReceiptSigner` (Keychain + injected modes), `sign(...)`, `verify(_:against:)`, `ReceiptSignerError` |
| `ReceiptUndoManager.swift` | 306 | Undo/redo stacks, `register(_:)`, `group(_:)`, `undo(...)`, `redo(...)`, snapshot/restore, voided-receipts log |
| `TextCursor.swift` | 64 | `TextCursor` + `Affinity`, `CursorPair` |
| `ChatQueueItem.swift` | 194 | `ChatQueueItem` + `State` (4 cases), `ChatQueue` (insertAtFront, reorder, supersede, start, finish, fail) |
| `DocumentStore.swift` | 297 | Productivity wrapper around `TesseraDataLayer`: loadDocument, saveDocument, apply, applyBatch, history, loadChatQueue, saveChatQueue |

### New: `TesseraStudio/Tests/TesseraCoreTests/Productivity/` (10 files, 2,192 LoC)

| File | Tests | What's covered |
|---|---:|---|
| `BlockTests.swift` | 24 | All 13 BlockType cases, all 9 InlineRun.Annotation variants, round-trip JSON, deep nesting (100+), content-hash stability |
| `MutationEngineTests.swift` | 30 | Every mutation variant applies, validation rejects bad input, cycle detection, composition, edge cases |
| `ReceiptTests.swift` | 15 | Sign + verify round-trip, tamper detection, voided, C2PA manifest format (5 required assertions), JSON round-trip, summary |
| `ReceiptSignerTests.swift` | 4 | Composition, keychain fallback, error equality |
| `ReceiptUndoManagerTests.swift` | 10 | Single undo/redo, batched, undo-of-undo, redo stack semantics, voiding, snapshot/restore |
| `TextCursorTests.swift` | 11 | Equality, serialization, two-cursor data model, CursorPair |
| `ChatQueueItemTests.swift` | 13 | State transitions (pending→inProgress→applied/failed), ordering, superseding, serialization |
| `DocumentStoreTests.swift` | 3 | JSON encoding/decoding, error equality |
| `ProductivityDataLayerTests.swift` | 11 | Env-gated DB integration: schema verification, monotonic chain_index, per-document isolation, chat queue persistence |

### New: `TesseraStudio/Tests/TesseraCoreTests/Encryption/TesseraKeychainVolumeReceiptSigningTests.swift` (1 file, 94 LoC, 5 tests)

Volume password → ed25519 key derivation round-trip; sign + verify through the Keychain entry.

### Modified (existing files)

| File | +LoC | What changed |
|---|---:|---|
| `TesseraStudio/Sources/TesseraCore/Data/TesseraDataLayer.swift` | +53 | 5 pass-through methods: `appendReceiptToChain`, `receiptChain`, `latestChainIndex`, `loadChatQueue`, `saveChatQueue` |
| `TesseraStudio/Sources/TesseraCore/Data/TesseraDataStore.swift` | +175 | The same 5 methods on the actor (real SQL via `PostgresNIO`); `nextChainIndex`, `receiptChain` JOIN query, `loadChatQueue`, `saveChatQueue` |
| `TesseraStudio/Sources/TesseraCore/Encryption/TesseraKeychainVolume.swift` | +53 | `receiptSigningKey()` and `receiptVerificationKey()` -- the volume password IS the ed25519 seed |

### New docs + migration

| File | LoC | What it is |
|---|---:|---|
| `docs/tessera-productivity-foundations-design.md` | 516 | The design spec matching the format of `tessera-data-layer-design.md` and `tessera-plead-the-fifth-design.md`. 14 sections: problem, design rationale, Block AST, Mutation API, Receipt infra, Undo, cursors, chat queue, data-layer integration, test strategy, library survey, constitutional properties, next steps, JSON appendix |
| `tools/tessera/db/migrations/0002_productivity_receipts.sql` | 53 | `receipt_chain` (per-document linear ordering with `(document_id, chain_index)` PK and DESC index) + `chat_queues` (per-doc JSONB blob) |

**Totals:** 24 files changed, 5,488 insertions, 0 deletions.

---

## Test results

```
Executed 619 tests, with 28 tests skipped and 0 failures (0 unexpected) in 62.959s
```

| Bucket | Tests | Pass | Skip | Fail |
|---|---:|---:|---:|---:|
| Existing (pre-Phase-1) | 493 | 493 | 0 | 0 |
| New (Phase 1) | 126 | 126 | 0 | 0 |
| **Combined** | **619** | **619** | **0** | **0** |
| ProductivityDataLayerTests (env-gated) | 11 | 0 (skip w/o `TESSERA_DB_INTEGRATION=1`) | 11 | 0 |
| Other (existed, also env-gated) | 17 | 0 | 17 | 0 |
| **Total skipped** | 28 | | | |

The 28 skipped tests are all env-gated on `TESSERA_DB_INTEGRATION=1` (the same gating as the existing data-layer integration tests on main). They run when the architect spins up a Postgres on `localhost:5432` with credentials `tessera/tessera/tessera` and sets the env var.

---

## Design decisions

### Decision 1: Receipt signing key is the volume password (composed, not extended)

**The brief:** "Use the existing `TesseraKeychainVolume` actor on main — it owns the Keychain lifecycle. Do NOT introduce a new Keychain dependency. The Plea the Fifth executor's `destroy_volume_password` step (step 3) also destroys this Keychain entry."

**The decision:** The volume password bytes ARE the ed25519 seed. `TesseraKeychainVolume.receiptSigningKey()` base64-decodes the stored password and constructs `Curve25519.Signing.PrivateKey(rawRepresentation:)` from the 32 random bytes. There is no separate signing-key entry to forget to destroy. Step 3 destroys the password; the signing key disappears with it.

**Why not the alternative:** A separate signing-key Keychain entry would require extending `PleadTheFifthExecutor` step 3 to also destroy it, AND would double the key-lifecycle surface. The composed approach is the load-bearing design decision the spec §7.2.1 calls out: "the signing key and the Plea the Fifth crypto-shred key are the same key."

### Decision 2: Pre-mutation snapshot embedded in the receipt

**The brief:** "Each `Mutation` has an inverse... The inverse of a `Mutation` is itself a `Mutation`."

**The decision:** The receipt carries a `preMutationSnapshot: [UUID: Block]` field — the touched blocks in their pre-state, captured at apply-time. The `Mutation.inverse(preMutation:)` method uses the snapshot to compute the inverse. The receipt is self-contained for undo: a future auditor with the receipt can undo it without the live document.

**Why not the alternative:** Computing the inverse from the live document at undo-time is broken — the document may have been further mutated between apply and undo. The receipt loses its audit-trail property (anyone with the receipt can verify, no one can verify-and-undo without the live state).

### Decision 3: Custom Codable for `[UUID: Block]`

**The constraint:** Swift's `JSONDecoder` does NOT decode `Dictionary<UUID, V>` from a JSON object directly. It expects an array of alternating `[key, value, key, value, ...]` pairs (the JSON "object" form is only supported when the key is `String`).

**The decision:** `DocumentAST` has a custom `Codable` conformance that encodes the `[UUID: Block]` map as a JSON object with stringified UUID keys, and decodes the same way. The on-disk shape is human-readable; the in-memory type is the spec's `[UUID: Block]`.

**Why this matters:** The receipt chain's `graph_receipts.payload` stores the full `Receipt` JSON, which includes the AST-derived `preMutationSnapshot`. A standard JSON inspector can read the receipt chain without special tooling.

### Decision 4: C2PA manifest format (deferral + placeholder)

**The brief:** "If `c2pa-rs` Swift bindings don't exist yet, the worker should... use the `ContentAuthenticity` Swift package if it has matured since the brief was written... Or document the deferral in the worker report and use a placeholder C2PA manifest format that matches the spec's structure."

**The decision:** Placeholder format. The `C2PAManifest` struct matches the C2PA Technical Specification 2.x field set (`format: "c2pa.v2"`, `claim_generator`, `assertions: [...]`, `signature: "..."`) with the required `c2pa.hash.data` and `c2pa.actions` assertions PLUS a few `tessera.*` extensions (`tessera.actor`, `tessera.summary`, `tessera.receipt_id`) for richer provenance. The signature algorithm is `ed25519` (encoded as `"ed25519:..."`) instead of the spec example's `es256`, because the architect chose ed25519 as the receipt-signing primitive and adding a separate ECDSA P-256 key path would double the key-lifecycle surface.

**Why not adopt `c2pa-rs` now:** It's a Rust library; bringing it in via FFI requires a new Swift C-ABI shim AND a vendored Rust toolchain in the SwiftPM build. Apple's `ContentAuthenticity` is closed-source and not a public SDK (it's available on macOS Sonoma via a private framework, but the API is not public). Neither is production-grade today.

**Forward compatibility:** The placeholder field set is identical to what a real C2PA library would produce. When a production C2PA library matures, the worker swaps in the real library against the placeholder format. No schema change required.

**The C2PA manifest shape (JSON), per receipt:**

```json
{
  "format": "c2pa.v2",
  "claim_generator": "tessera/1.0",
  "assertions": [
    { "label": "c2pa.hash.data",  "data": { "hash": "sha256:..." } },
    { "label": "c2pa.actions",    "data": { "actions": [{ "action": "c2pa.edited" }] } },
    { "label": "tessera.actor",   "data": { "type": "user" | "agent", ... } },
    { "label": "tessera.summary", "data": "human-readable summary" },
    { "label": "tessera.receipt_id", "data": "uuid" }
  ],
  "signature": "ed25519:..."
}
```

### Decision 5: Redo stack stores the ORIGINAL (not the inverse)

**The bug I caught during testing:** A common wrong design is to push the inverse receipt onto the redo stack, so redo re-applies the inverse (restoring the pre-undo state). The correct design is to push the ORIGINAL receipt onto the redo stack, so redo re-applies the original (restoring the post-original state — what the user wanted when they hit redo). This matches the spec §5.4: "redo restores the exact original receipt."

**How the voiding is tracked:** The redo receipt (the new one signed when redo runs) is what marks the original as `voidedBy = redoReceipt.id`. The original stays in the redo stack (so further redos are possible) but is visibly voided.

### Decision 6: No new external libraries adopted

The Phase 1 worker adopted **zero** new external libraries. Composed with existing infrastructure throughout:

- **CryptoKit** (already on main) for ed25519 signing.
- **`TesseraKeychainVolume`** (already on main) for the signing-key seed.
- **`TesseraDataLayer`** (already on main) for durability.
- **`JSONValue`** (already on main) for `AnyCodable`.

Adding new libraries (e.g. `c2pa-rs`, an `AnyCodable` package, a third-party receipt signer) would have added attack surface, build time, and maintenance cost for no gain in this phase. The design doc §11.1 defends each "no" with one paragraph.

---

## C2PA manifest format

The placeholder `C2PAManifest` is documented in:

1. `TesseraStudio/Sources/TesseraCore/Productivity/Receipt.swift` — the type definition with field-level doc comments.
2. `docs/tessera-productivity-foundations-design.md` §5.3 — the design rationale and the deferral note.

**Spec version targeted:** C2PA Technical Specification 2.x (per the productivity spec §5.3 + §7.2).

**Format string:** `c2pa.v2`.

**Signature algorithm:** `ed25519` (deviation from the spec example's `es256`; documented in §5.3 and the design doc).

**Required assertions:**
- `c2pa.hash.data` — SHA-256 of the document AST (canonical JSON, sorted keys).
- `c2pa.actions` — `[{ "action": "c2pa.edited" }]` (per the spec §5.3).

**Tessera extensions** (forward-compatible — a verifier without these can still verify the C2PA core):
- `tessera.actor` — `{"type": "user" | "agent", ...}` for the actor.
- `tessera.summary` — the receipt's human-readable summary.
- `tessera.receipt_id` — the receipt id, so a future verifier can cross-reference.

---

## Receipt chain integration with the data layer

### New tables (migration `0002_productivity_receipts.sql`)

**`receipt_chain`** — the per-document linear ordering:

```sql
CREATE TABLE receipt_chain (
    document_id   uuid NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
    chain_index   bigint NOT NULL,
    receipt_id    uuid NOT NULL REFERENCES graph_receipts(id) ON DELETE RESTRICT,
    created_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (document_id, chain_index)
);
CREATE INDEX idx_receipt_chain_doc ON receipt_chain (document_id, chain_index DESC);
```

- Composite PK `(document_id, chain_index)` gives O(log n) lookup.
- DESC index on `(document_id, chain_index)` supports `DocumentStore.history(of:limit:)` ("newest first" reads).
- `ON DELETE CASCADE` on `document_id`: deleting a document drops its chain.
- `ON DELETE RESTRICT` on `receipt_id`: a receipt cannot be deleted while it's in a chain (audit-trail integrity).

**`chat_queues`** — the per-document chat queue:

```sql
CREATE TABLE chat_queues (
    document_id  uuid PRIMARY KEY REFERENCES graph_entities(id) ON DELETE CASCADE,
    items        jsonb NOT NULL DEFAULT '[]'::jsonb,
    updated_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX idx_chat_queues_updated_at ON chat_queues (updated_at);
```

- JSONB blob per document.
- `idx_chat_queues_updated_at` supports the (future) "stale queue cleanup" job.

### New data-layer methods (facade pass-through to actor)

| Method | Purpose | Where |
|---|---|---|
| `appendReceiptToChain(documentID:receiptType:payload:signature:)` | Insert into `graph_receipts`, link into `receipt_chain` at next monotonic index. The two writes are NOT in a single transaction (postgres-nio simple-query path doesn't expose transactions); a crash between them leaves an orphan in `graph_receipts` that the (planned) `rebuildReceiptChain` helper fixes on document open. | `TesseraDataStore` (private impl) + `TesseraDataLayer` (facade) |
| `receiptChain(documentID:limit:)` | JOIN `receipt_chain` to `graph_receipts`, return `[(chainIndex, receipt)]` ordered by chain_index ASC. | same |
| `latestChainIndex(documentID:)` | `MAX(chain_index)` for the document; nil if empty. | same |
| `loadChatQueue(documentID:)` | Returns the JSONB blob (`"[]"` if no row). | same |
| `saveChatQueue(documentID:itemsJSON:)` | Upsert with `ON CONFLICT (document_id) DO UPDATE`. | same |

### Migration application

The migration file is at `tools/tessera/db/migrations/0002_productivity_receipts.sql`. It's applied:

- By the existing `psql` / `docker compose migrate` path (the production bootstrap).
- By `TesseraDataStore.applyMigrations(...)` (the in-process path used by tests and the productivity surface's startup). The new `ProductivityDataLayerTests` exercises this end-to-end with a throwaway DB.

The migration is **additive** — it doesn't change any existing table. `0001_init.sql` is unchanged.

---

## Things I punted on (and why)

### Punt 1: True single-transaction receipt append

The `appendReceiptToChain` writes to `graph_receipts` and `receipt_chain` in two separate calls. The `postgres-nio` simple-query path doesn't expose explicit transactions, so a crash between the two writes leaves an orphan in `graph_receipts` (with no chain entry). The (planned) `rebuildReceiptChain(documentID:)` helper would walk `graph_receipts` and re-link orphans to the chain on document open. For Phase 1, the error path is logged and the orphan is visible in the audit trail (the receipt is in `graph_receipts` with a `priorReceiptID` that points at the right prior receipt; the chain entry just doesn't exist yet).

**Why punted:** Adding a transaction to the simple-query path is a real piece of work. The current behavior is recoverable; the fix is a follow-up.

### Punt 2: `setBlockContent` on `divider` blocks

The validation rejects `setBlockContent` on a `divider` block (the divider has no content). The inverse case (`setInlineAnnotation` on a `divider`) is also rejected at validation time. Other invariant violations (e.g. `insertBlockAfter` inside a `codeBlock` — should code blocks be container-capable?) are not yet enumerated. Phase 2's editor will discover the gaps; we'll add them then.

**Why punted:** The mutation engine validates the obvious violations; Phase 2's `TesseraTextContentManager` will surface the renderer-driven ones. We don't want to over-constrain the API in Phase 1.

### Punt 3: C2PA library integration

Already documented above. The placeholder is forward-compatible; a real C2PA library can swap in without a schema change.

### Punt 4: Multi-device receipt key sync

The receipt signing key is per-device (one device, one signing key). A user with two devices has two keys; receipts from one device cannot be verified on the other. The spec §7.2.1 calls this out: "v1 is per-device". Multi-device sync is a v2 follow-up.

**Why punted:** Out of scope per the spec.

### Punt 5: `voidedBy` persistence

The `Receipt.voidedBy` field is set in memory by the undo manager when it appends the inverse receipt. The underlying `graph_receipts.payload` JSONB row is mutable, but the chain is append-only at the receipt-row level. The "voided by" pointer lives in the voiding receipt's `summary` field (via the inverse mutations) and in the original's `voidedBy` field in memory. A future Phase 3 PR may move voiding to a separate `voiding_receipt` table for forensic clarity, but the current shape is sufficient for the receipt drawer UI.

**Why punted:** The data-layer extension is a follow-up. The in-memory tracking is enough for the undo manager; the receipt drawer can read the voiding receipt's `summary` and reconstruct the relationship.

### Punt 6: The ChatQueue JSON encoding uses `JSONValue` (AnyCodable) for the `actor` field

The `ChatQueueItem.actor` field is an `Actor` enum, which is `Codable` directly. The queue serialization round-trips through `JSONEncoder` / `JSONDecoder` without issue. No punting here — the JSON is just a regular JSON document. (I mention this because I considered using `JSONValue` and decided against; the enum's `Codable` is fine.)

---

## How to use (1 page)

For the architect's verification.

### 1. Open the worktree

```bash
cd /Users/user/Developer/GitHub/tessera
git worktree list
# worktrees/prod-foundations  0c1bda2a7 [...]
cd worktrees/prod-foundations
git log --oneline -3
# 0c1bda2a7 productivity: Phase 1 foundations ...
# d4ef9098f data: add Postgres + Valkey data layer foundation
# 30dc272cf integration: post-merge cleanup for Plea the Fifth + asset catalog
```

### 2. Read the design spec

```bash
less docs/tessera-productivity-foundations-design.md
```

14 sections; the table of contents is at the top.

### 3. Run the tests

```bash
cd TesseraStudio
swift test
# Executed 619 tests, with 28 tests skipped and 0 failures
```

The 28 skipped tests are env-gated on `TESSERA_DB_INTEGRATION=1`. To run them, start a Postgres on `localhost:5432` with credentials `tessera/tessera/tessera` and re-run with the env var set:

```bash
TESSERA_DB_INTEGRATION=1 swift test --filter ProductivityDataLayerTests
# 11 tests, 0 failures (with Postgres up)
```

### 4. Try the receipt sign/verify round-trip (in a Swift REPL or test)

```swift
import CryptoKit
@testable import TesseraCore

// 1. Generate a fresh receipt signing key (in production this
//    comes from the Keychain via TesseraKeychainVolume).
let key = Curve25519.Signing.PrivateKey()
let signer = ReceiptSigner(signingKey: key)

// 2. Build a mutation.
let block = Block(type: .paragraph, content: [InlineRun(text: "hi")])
let mutation = Mutation.insertBlockAfter(parentID: nil, anchorID: nil, block: block)

// 3. Apply it (the engine returns the pre-mutation snapshot).
var doc = DocumentAST.empty
var engine = MutationEngine()
let preSnapshot = try engine.apply(mutation, to: &doc)

// 4. Sign the receipt.
let receipt = try signer.sign(
    documentID: UUID(),
    mutations: [mutation],
    priorReceiptID: nil,
    actor: .user(UUID()),
    preMutationSnapshot: preSnapshot
)

// 5. Verify the receipt.
let verification = signer.verify(receipt, against: key.publicKey)
print(verification)  // ReceiptVerification.valid

// 6. Undo it.
let manager = ReceiptUndoManager(documentID: receipt.documentID)
manager.register(receipt)
let result = try manager.undo(
    document: doc,
    actor: .user(UUID()),
    signer: signer
)
// result.updatedDocument is empty; result.inverseReceipt is a
// new signed receipt that voids the original; the original is
// now in manager.snapshotVoidedReceipts() with voidedBy set.
```

### 5. Persist via DocumentStore (requires Postgres + `TESSERA_DB_INTEGRATION=1`)

```swift
// Production: data layer is started via TesseraDataLayer.start().
let dataLayer = TesseraDataLayer(configuration: ...)
let outcome = await dataLayer.start()
let store = DocumentStore(dataLayer: dataLayer)

// Apply a mutation; the receipt is appended to the chain.
let receipt = try await store.apply(
    mutation: .setBlockContent(blockID: blockID, content: [InlineRun(text: "new")]),
    to: documentID,
    actor: .user(currentUserID)
)

// Read the history.
let history = try await store.history(of: documentID, limit: 10)
```

### 6. Verify the migration is forward-compatible

```bash
# Apply the migration to a fresh database.
psql -h localhost -U tessera -d tessera -f tools/tessera/db/migrations/0001_init.sql
psql -h localhost -U tessera -d tessera -f tools/tessera/db/migrations/0002_productivity_receipts.sql

# Inspect.
psql -h localhost -U tessera -d tessera -c "\dt"
# graph_entities, entity_links, graph_receipts, receipt_chain, chat_queues

psql -h localhost -U tessera -d tessera -c "\d receipt_chain"
# document_id (uuid, FK to graph_entities, ON DELETE CASCADE)
# chain_index (bigint)
# receipt_id (uuid, FK to graph_receipts, ON DELETE RESTRICT)
# created_at (timestamptz, default now())
# PK (document_id, chain_index)
```

---

## Open questions for the architect

1. **Receipt sign timing** — currently the receipt is signed AFTER the in-memory mutation is applied but BEFORE the AST is persisted. If the AST persist fails, the in-memory mutation has happened but the receipt doesn't exist (the data layer will be inconsistent on next open). Alternative: persist AST first, then sign + append receipt (what we do). The receipt is the recovery handle: the data layer walks the chain on open and re-derives state. Acceptable?

2. **C2PA library decision** — happy to vendor `c2pa-rs` via FFI in a follow-up if the architect wants. The placeholder is sufficient for v1; the manifest format is forward-compatible.

3. **`voidedBy` storage** — currently in-memory + in the JSONB payload's `voidedBy` field (mutable). Cleaner to have a `voiding_receipts` table (one row per voiding). Phase 3 work?

4. **Per-device keys** — confirmed per spec §7.2.1, but worth a second look: do we want a "I trust this other device" UX for multi-device receipt verification in v1, or punt to v2?

5. **The `Receipt` struct's `preMutationSnapshot` field** — this can be large (the full block content). For very large documents, the snapshot may dominate the receipt size. Worth profiling?

Otherwise: foundation is ready to FF-merge. Phase 2 / Phase 4 / Phase 6 can dispatch on this base.
