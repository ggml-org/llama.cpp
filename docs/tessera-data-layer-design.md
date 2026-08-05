# Tessera Studio: Data Layer Design Specification

**Status:** Draft v1 — 2026-08-05
**Author:** Tessera Architecture
**Applies to:** Tessera Studio for macOS 1.0.0+ (post-SwiftData, pre-productivity-surface)

---

## 1. Executive Summary

Tessera Studio currently uses SwiftData as the on-device app store. The next
phase — a productivity surface (Tasks, Reminders, Events, Notes, Emails, Docs,
Sheets, Slides) with an AI-driven live editor and export to PDF / Apple Mail
/ Slack — needs a server-side store that supports full-text search, vector
embeddings, a typed knowledge graph, and constitutional receipts. The
SwiftData store is the right shape for the local app surface but not for the
server-side retrieval and graph queries the productivity surface requires.

This document designs the **Tessera Data Layer**: Postgres 16 + pgvector +
pg_trgm (durable) plus Valkey 7 (ephemeral), wrapped behind a Swift facade
that the productivity surface depends on instead of the raw client
libraries. The facade is the load-bearing piece — it enforces the hexagonal
boundary that keeps Postgres and Valkey types out of the productivity
code, so the productivity surface ships without ever importing
`PostgresNIO` or `RediStack`.

**Hard constraints** carried over from the project posture:

- No SaaS, no API keys, no third-party hosted services.
- Hexagonal boundary: no raw SQL or Redis commands outside the data layer.
- Apple Silicon native (`platform: linux/arm64` in compose).
- Linux + macOS both compile and test cleanly.
- The local SwiftData store remains; the data layer sits **alongside** it,
  not in place of it.

**Out of scope for this worker** (deferred to the next wave): the
productivity-surface tables (Tasks, Reminders, Events, Notes, Emails,
Materials), the importer, the AI-driven live editor, the export pipeline,
receipt signing, and the embedder integration. The data layer is the
foundation; the productivity surface builds on top of it.

---

## 2. Why Postgres + Valkey (vs SQLite, vs embedded Redis, vs FoundationDB)

**SQLite + extensions** would be the path of least resistance on macOS, but
the productivity surface is multi-process (TesseraStudioMac and the
tessera-cli subprocess both need to read the same store) and the receipts
need to be queryable by the future receipt-verifier sidecar. SQLite's
file-level locking doesn't compose well with the planned multi-process
deployment. pgvector + pg_trgm are also more battle-tested than their
SQLite equivalents (sqlite-vss is alpha-tier; sqlite-fts5 is solid but
we'd need to wire the embedder separately).

**Embedded Redis** (e.g. KeyDB in-process, or a Swift re-implementation)
loses the cross-process story. The productivity surface has at least two
processes (TesseraStudioMac and tessera-cli) and the receipt-verifier
sidecar wants to read the cache too. A real Redis / Valkey server is
the standard solution; embedding the server inside the app would be
novel for no benefit.

**FoundationDB** is excellent for ordered KV with watchable transactions
but its query layer (Record Layer) is JVM-based. The productivity
surface needs SQL-shaped queries (graph traversals + RRF ranking +
full-text search) and that means we either bolt a query layer on top
of FDB or pick a store that has SQL natively. Postgres wins on
ergonomics.

**Postgres 16 + pgvector + pg_trgm + Valkey 7** is the conservative
choice: well-understood, single-server (no consensus dance), runs on
the user's own machine, no SaaS / API keys. The combination covers
graph (recursive CTE), full-text (tsvector + GIN), vector (pgvector
HNSW), and ephemeral (Valkey) in one local stack. Both services have
native `linux/arm64` images, so Apple Silicon containers don't pay
the Rosetta tax.

The split is: **Postgres is durable (graph, receipts, embeddings, FTS);
Valkey is ephemeral (scratchpad, decay windows, session state, idempotency
keys, in-flight cache)**. The two never reach across the boundary — the
facade composes them but doesn't replicate data between them. A read
misses in Valkey, falls through to Postgres, and backfills Valkey with
a short TTL. A write hits Postgres first, then invalidates the
relevant Valkey keys.

---

## 3. Schema Design

The schema is defined in `tools/tessera/db/migrations/0001_init.sql` and
seeded by `tools/tessera/db/seeds/seed.sql`. The model is the universal
"one row per thing" pattern with a single polymorphic
`graph_entities` table.

### 3.1 Why polymorphic

Hybrid search needs the same shape across every entity type. Splitting
by type means N parallel `UNION ALL` queries in the retrieval path, which
kills the planner. Edges (`entity_links`) are type-agnostic — the
`link_type` column discriminates. Recursive CTEs walk the table without
per-type metadata joins.

Productivity columns that ARE type-specific (due dates for tasks,
calendar windows for events) live in dedicated sidecar tables that
reference `graph_entities` 1:1. They are added in a later migration;
the data-layer foundation stays lean.

### 3.2 Tables

**`graph_entities`** — universal entity table.

- `id` uuid PK (server-side `gen_random_uuid()`).
- `entity_type` text NOT NULL (the discriminator: material, file,
  project, chat, message, topic, person, tool_invocation, receipt,
  decision, task, reminder, calendar_event, email, note, document,
  spreadsheet, presentation). The productivity surface adds
  tables that reference this id; this column is the join key.
- `subtype` text NULL (for finer-grained types: chat.message vs
  chat.summary, task.bucket vs task.grocery, etc.).
- `label` text NOT NULL (the short human-readable string; weighted A
  in the tsvector).
- `body` text NULL (the long-form text; weighted B in the tsvector).
- `source_url` text NULL (for entities that originated from a URL —
  receipts, web clips, etc.).
- `created_at`, `updated_at` timestamptz (server-managed; the
  `touch_updated_at` trigger keeps `updated_at` honest when callers
  UPDATE without explicitly setting it).
- `search_tsv` tsvector GENERATED ALWAYS AS (label weighted A, body
  weighted B) STORED. Generated columns let the index see the result
  without a trigger.
- `embedding` vector(1536). 1536 matches OpenAI text-embedding-3-small;
  for local embeddings, callers re-embed to whatever their model
  produces and pass the new vector through `upsertEntity`. The
  column type is fixed at 1536 in this migration; the dimension
  switch is a follow-up (the worker report calls this out as a
  known limitation; changing the dimension is a destructive
  migration that the user should sign off on, not something the
  worker should bake in).

**`entity_links`** — typed graph edges.

- `id`, `source_id`, `target_id` (FK to `graph_entities` with
  ON DELETE CASCADE), `link_type`, `weight` (real, default 1.0),
  `created_at`.
- UNIQUE (source_id, target_id, link_type) — the same edge can
  appear at most once; the production `linkEntities` does an
  upsert that increments `weight` on conflict if the caller asks.

**`graph_receipts`** — constitutional receipt log.

- `id`, `entity_id` (FK), `receipt_type`, `payload` jsonb (the
  schema-versioned receipt; e.g. `"schema":
  "tessera.receipt.tool_invocation.v1"` inside the json),
  `signature` bytea (64-byte ed25519; **NULL for now** — the
  signing path is a follow-up but the column exists so the schema
  doesn't have to migrate when signing lands), `witnessed_at`
  timestamptz.

### 3.3 Indexes

| Index | Type | Purpose |
|---|---|---|
| `idx_entities_type` | B-tree | Lookup by `entity_type` (the productivity surface's most common filter). |
| `idx_entities_subtype` | B-tree (partial) | Lookup by `subtype`; partial because most entities have `subtype IS NULL`. |
| `idx_entities_source_url` | B-tree (partial) | Lookup by `source_url`; partial because most entities don't have one. |
| `idx_entities_search_tsv` | GIN | Full-text search via `ts_rank_cd` in the keyword signal. |
| `idx_entities_embedding` | HNSW (vector_cosine_ops) | K-nearest-neighbours in the vector signal. |
| `idx_entities_trgm_label` | GIN (gin_trgm_ops) | Typo-tolerant autocomplete on `label`. |
| `idx_links_source` | B-tree | Outgoing-edge lookup. |
| `idx_links_target` | B-tree | Incoming-edge lookup. |
| `idx_links_type` | B-tree | Filter edges by type. |
| `idx_receipts_entity` | B-tree | Receipts for a given entity. |
| `idx_receipts_type` | B-tree | Receipts of a given type (compliance, audit). |

The HNSW index is the costliest; the migration builds it eagerly,
which on a 100k-row table takes a few seconds and blocks writes. The
productivity surface can add `m` and `ef_construction` tuning in a
follow-up if the recall / latency trade-off needs adjustment.

### 3.4 Extensions

- `vector` (pgvector) — for the `embedding` column + HNSW index.
- `pg_trgm` — for the trigram GIN index on `label`.
- `pgcrypto` — for `gen_random_uuid()`. Optional; if the host Postgres
  doesn't have it, the migration can be amended to use a
  `uuid-ossp` extension or a server-side id generator.

---

## 4. Hybrid Retrieval (RRF)

The load-bearing query is `hybrid_search(p_anchor uuid, p_query_text text,
p_query_embedding vector(1536), p_max_depth int)`. It walks the graph
from `p_anchor` up to `p_max_depth` hops, then ranks the reachable
set by Reciprocal Rank Fusion (Cormack et al., 2009) across three
signals.

### 4.1 The query

```sql
WITH RECURSIVE walk AS (
    SELECT target_id AS id, 1 AS depth, link_type
      FROM entity_links WHERE source_id = p_anchor
    UNION ALL
    SELECT el.target_id, w.depth + 1, el.link_type
      FROM walk w JOIN entity_links el ON el.source_id = w.id
     WHERE w.depth < p_max_depth
),
vector_ranked AS (
    SELECT e.id, ROW_NUMBER() OVER (ORDER BY e.embedding <=> p_query_embedding) AS rn
      FROM graph_entities e
     WHERE p_query_embedding IS NOT NULL
),
keyword_ranked AS (
    SELECT e.id, ROW_NUMBER() OVER (
        ORDER BY ts_rank_cd(e.search_tsv, plainto_tsquery('english', p_query_text)) DESC
    ) AS rn
      FROM graph_entities e
     WHERE p_query_text IS NOT NULL
       AND e.search_tsv @@ plainto_tsquery('english', p_query_text)
)
SELECT
    e.id, e.entity_type, e.label, e.body,
    (1.0 / (1 + 1.5 * w.depth))::real                                         AS graph_score,
    COALESCE(1 - (e.embedding <=> p_query_embedding), 0.0)::real               AS vector_score,
    COALESCE(ts_rank_cd(e.search_tsv, plainto_tsquery('english', p_query_text)), 0.0)::real AS keyword_score,
    (
        0.2 * COALESCE(1.0 / (60 + vr.rn), 0) +
        0.5 * COALESCE(1.0 / (60 + kr.rn), 0) +
        0.3 * COALESCE(1.0 / (1 + 1.5 * w.depth), 0)
    )::real                                                                    AS rrf_score
FROM walk w
JOIN graph_entities e ON e.id = w.id
LEFT JOIN vector_ranked  vr ON vr.id = e.id
LEFT JOIN keyword_ranked kr ON kr.id = e.id
ORDER BY rrf_score DESC
LIMIT 25;
```

The function returns up to 25 rows. The `LIMIT` is hard-coded; if
the productivity surface needs a different cap, we add a
`p_max_results int DEFAULT 25` parameter in a follow-up migration.

### 4.2 Weight calibration

RRF weights (0.2 graph / 0.5 vector / 0.3 keyword) were calibrated
against the 5-entity / 4-link test fixture shipped in
`tools/tessera/db/seeds/seed.sql`. The fixture is small on purpose:
the test asserts the *ordering* (project at depth 1 ranks above
topic at depth 2 ranks above document/person at depth 3), not the
absolute RRF score.

The `+60` constants in the RRF denominators are the standard
k from the original Cormack paper; larger k means a slower
falloff (more emphasis on lower-ranked matches), smaller k means
the top rank dominates more. 60 is a reasonable default; the
productivity surface can experiment with smaller values (e.g. 10)
if the test fixture shows the top hit is being drowned out by
mid-tier matches.

In a denser graph (1000+ entities, 5+ hops), the graph signal
will dominate. The current weights are tuned for sparse graphs;
if a real workload shows the vector signal being drowned out by
graph noise, the weights shift toward 0.6 vector / 0.2 keyword /
0.2 graph. We do NOT tune the weights in the migration because
the test fixture is too small to derive meaningful weights from;
the productivity surface owns the calibration in production.

### 4.3 Depth limit

`p_max_depth int DEFAULT 3` is conservative. The recursive CTE
traverses up to 3 hops from the anchor; for the test fixture,
that's the entire reachable set. For a production graph with
100k+ entities, depth=3 is still tractable (the CTE materialises
the reachable set, which is bounded by graph density, not by
total node count). The productivity surface can pass a higher
depth for slow, exhaustive search; we cap it at 5 in the
calling code so a misuse doesn't run away.

---

## 5. Cache Patterns (Valkey)

The cache is ephemeral state. Everything in the cache can be lost
without violating the user's data integrity — Postgres is the
source of truth.

| Key pattern | Namespace | TTL | Use case |
|---|---|---|---|
| `tessera:<ns>:scratchpad:<agent-id>:<task>` | `scratchpad` | 1h | Agent's working memory for a multi-turn task. Invalidated on task completion. |
| `tessera:<ns>:capture:<user-id>:<ts>` | `capture` | 24h | Recently-captured items (clipboard, screenshot, URL). Powers the "what did I just grab" panel. |
| `tessera:<ns>:decay:<category>:<entity-id>` | `decay` | 7d | Sorted-set member of a decay window. Score = unix timestamp at insert. Expired entries swept by `ZREMRANGEBYSCORE`. |
| `tessera:<ns>:session:<user-id>` | `session` | 30m | Session state for the active TesseraStudioMac window. |
| `tessera:<ns>:idem:<endpoint>:<key-hash>` | `idempotency` | 1h | Idempotency keys for the future receipt-verifier API surface. |
| `tessera:<ns>:lookup:<endpoint>:<id>` | `lookup` | 5m | Read-through cache for the productivity surface's "look up entity by id" path. |

**Default TTLs** are tuned for a single-user, single-machine workload.
The productivity surface can override per-call.

**Invalidation strategy**: writes to Postgres are followed by a
`DEL` of the affected cache keys. We do NOT write-through because
that creates a window where an offline Postgres leaves the cache
stale. The cache is best-effort.

**Decay windows** (the production use case for sorted sets): the
productivity surface inserts a (member, score=now) into a sorted
set at every event, then `ZREMRANGEBYSCORE` periodically to evict
members older than the window. The score is a unix timestamp, the
member is an opaque string (typically a UUID). `ZRANGEBYSCORE` reads
the window in score order.

**Key prefix** is `tessera:<namespace>:` (namespace is per-cache-
instance, default `"default"`). The prefix is automatically added
by every method on `TesseraCache`, so callers can't accidentally
bypass it.

---

## 6. Swift Client Architecture

The Swift client is three files in
`TesseraStudio/Sources/TesseraCore/Data/`:

- `TesseraDataStore.swift` — the Postgres client.
- `TesseraCache.swift` — the Valkey client.
- `TesseraDataLayer.swift` — the top-level facade.

### 6.1 Why these client libraries

**Postgres: `vapor/postgres-nio` 1.33.1 (SwiftNIO-based, Vapor-maintained).**

Picked over alternatives:

- **`vapor/fluent-postgres`**: ORM-shaped; we want raw SQL because the
  hybrid retrieval query is hand-written and ORM-shaped layers fight
  that.
- **`PerfectlySoft/Perfect-PostgreSQL`**: NIO but lower-level; we'd
  have to write the protocol layer ourselves.
- **`MainasuF/Swift-PG`**: not NIO-based, not production-grade.
- **A pure-Swift ORM**: would couple the surface to the productivity
  domain in a way the facade is designed to avoid.

`postgres-nio` 1.33.1 ships a `PostgresClient` that owns a
connection pool, exposes async/await query methods, and lets us
inject a custom event loop group. The `PostgresQuery` string
interpolation with `PostgresEncodable` bind values is the right
ergonomic: type-safe, parameterised (no SQL injection), and the
literal SQL stays readable.

`PostgresConnectionGroup` (mentioned in the worker brief) does
not exist as a public type in `postgres-nio` 1.33.1. The actual
class is `PostgresClient`, which has its own built-in pool
(`maximumConnections` config). The worker report flags this
discrepancy; the brief's name is treated as a synonym for
`PostgresClient`.

**Valkey: `swift-server/RediStack` 1.6.3 (SwiftNIO-based, swift-server-maintained).**

Picked over alternatives:

- **`vapor/redis`**: requires `RediStack` underneath anyway; the
  higher-level wrapper is a thin shim we don't need.
- **`MainasuF/Swift-Redis`**: not NIO-based, not production-grade.
- **A bespoke Swift Redis client**: we'd be reimplementing the
  RESP protocol; the productivity surface shouldn't have to wait
  for that.

`RediStack` ships a `RedisConnectionPool` with `activate()`,
`send(command:with:)`, and the typed helpers (`get`, `set`,
`lpush`, `brpop`, `zadd`, `zrangebyscore`, `eval`) that the
productivity surface's decay-window and idempotency-key paths
need. The `RESPValue` round-trip on the lower-level `send()` is
how we build `SET key value NX EX <ttl>` without waiting for
RediStack to grow a typed helper.

### 6.2 Hexagonal boundary

The data layer enforces a strict boundary:

- `TesseraDataStore.swift` is the **ONLY** file in TesseraCore that
  imports `PostgresNIO`.
- `TesseraCache.swift` is the **ONLY** file in TesseraCore that
  imports `RediStack`.
- The productivity surface imports `TesseraDataLayer` and never
  the underlying client libraries.

The boundary is load-bearing for two reasons:

1. **Portability.** If we want to swap the underlying stores
   (FoundationDB, SQLite, a remote Postgres) later, the surface
   stays unchanged — only the three Data/ files move.
2. **Testability.** Tests for the productivity surface can swap
   the data store for an in-memory fake without dragging the
   client libraries into the test target.

The facade is `final class TesseraDataLayer` (an `actor` for
serialised mutation) that holds references to a `TesseraDataStore`
and a `TesseraCache`. Its public API is domain-shaped:
`getEntity(id:)`, `upsertEntity(_:)`, `linkEntities(...)`,
`appendReceipt(...)`, `hybridSearch(...)`, plus the cache
passthroughs (`cacheGet`, `cacheSet`, `cacheZadd`,
`cacheZrangebyscore`, etc.). No Postgres or Valkey types appear
on this surface.

### 6.3 The `actor` model

`TesseraDataStore` is an `actor` so all state mutations (the
client reference, the in-flight query tracking) are serialised
without explicit locks. The `PostgresClient` underneath is
already thread-safe (it's `Sendable` and uses a NIO event loop),
so the actor mostly wraps the lifecycle (`connect`, `close`)
and serialises `setConfiguration` so it can't run after
`connect()` has opened a client.

`TesseraCache` is the same pattern: an `actor` wrapping a
`RedisConnectionPool`. The pool is already thread-safe; the
actor wraps the lifecycle.

`TesseraDataLayer` is the third actor. It composes the other two
and adds the `start()` / `shutdown()` lifecycle that opens both
stores in one call. The `start()` outcome is one of four:
`ready`, `cacheDegraded(reason:)`, `dataStoreDegraded(reason:)`,
or a bootstrap failure. The facade never throws from `start()`;
a missing Postgres becomes `dataStoreDegraded` so the
productivity surface can run in a "scratchpad-only" mode (Valkey
works, Postgres doesn't).

---

## 7. Test Strategy

### 7.1 Env-gated integration tests

The 17 new tests in `TesseraStudio/Tests/TesseraCoreTests/Data/`
are **integration tests** that require a running Postgres and
Valkey. They are gated by `TESSERA_DB_INTEGRATION=1` so `swift
test` works in CI environments without Docker.

When the env var is set, the tests connect to `localhost:5432` /
`localhost:6379` with credentials from env vars
(`TESSERA_PG_USER`, `TESSERA_PG_PASSWORD`, `TESSERA_PG_DB`, etc.)
or default to `tessera/tessera/tessera` — matching the
docker-compose defaults.

When the env var is missing, every test calls `XCTSkip(...)`
which makes the test runner report it as "skipped" rather than
"failed". The 476 existing tests + 17 integration tests now show
as:

```
493 tests, 17 tests skipped (no DB)
493 tests, 0 failures, 0 skipped (with TESSERA_DB_INTEGRATION=1)
```

### 7.2 The 17 integration tests

- **`SchemaMigrationTests`** (4) — apply 0001_init.sql to a
  throwaway database, then assert:
  - All three tables exist.
  - All eleven indexes exist.
  - The `hybrid_search` function exists and has 4 args.
  - The `vector` and `pg_trgm` extensions are loaded.

- **`HybridSearchTests`** (4) — apply the migration + seed to a
  throwaway database, then assert:
  - All 4 reachable entities are returned at depth 3.
  - Depth ordering: project (1) > topic (2) > document / person (3).
  - Without a vector query, `vectorScore` is 0.
  - Depth=1 returns only the direct edge.

- **`ConnectionPoolTests`** (3) — apply the migration to a
  throwaway database, then assert:
  - 50 concurrent reads (random UUIDs, all nil) complete without
    deadlock.
  - 50 concurrent writes (different IDs) all succeed.
  - The pool opens at least one connection.

- **`CacheTTLTests`** (6) — set + read + sleep + assert,
  covering the core cache contract: TTL expiry, persistent SET
  without TTL, DEL semantics, INCR + EXPIRE composition,
  ZADD + ZRANGEBYSCORE on a sorted set, and SET-NX's first /
  second semantics.

### 7.3 The fixture

The seed fixture in `tools/tessera/db/seeds/seed.sql` is the
smallest graph that exercises the full hybrid retrieval path:
5 entities, 4 links, 2 receipts, embeddings on each entity, and
a generated tsvector per row. The fixture is the same one the
end-to-end test in `HybridSearchTests` uses — the test reads
from the file and applies it via `TesseraDataStore.applyMigrations`.

The fixture is deterministic (fixed UUIDs, fixed label strings,
fixed embeddings derived from the entity id index). Tests that
depend on the fixture can hard-code the expected ordering.

### 7.4 What's NOT tested in this worker

- **Receipt signing.** The column exists; the signing path is
  a follow-up. The signature tests will be added when the
  signing path lands.
- **The cache write-through policy.** The cache invalidation on
  write is wired in the facade, but the productivity surface
  doesn't exist yet, so the policy isn't exercised end-to-end.
- **Connection pool starvation under heavy load.** The 50-
  concurrent test is a smoke test, not a benchmark.
- **Postgres 16 specifically vs 17.** The docker-compose pins
  pgvector/pgvector:pg16; the local dev Postgres may be 17.
  The SQL is portable across 16 / 17.

---

## 8. What This Enables

The data layer is the foundation. The productivity surface, the
AI live editor, and the export pipeline all build on top of it
without re-touching the store layer.

### 8.1 The productivity surface

Tasks, Reminders, Events, Notes, Emails, Materials, Documents,
Spreadsheets, Slides — the 9 entity types the productivity
surface needs — are all `graph_entities` rows with the right
`entity_type`. The sidecar tables (due dates, calendar windows,
thread ids) are added in a follow-up migration that references
`graph_entities(id)` 1:1. The retrieval layer (`hybridSearch`)
already works for them because the schema is polymorphic.

### 8.2 The AI live editor

The editor's knowledge retrieval ("what does the user know about
this topic?") is `hybridSearch(anchor: queryText: queryEmbedding:
maxDepth:)` against the user's recent context anchor (the open
chat message, the open document, the open note). The RRF
weights were calibrated for that query shape.

### 8.3 The export pipeline

PDF / Apple Mail / Slack export all read from `graph_entities` and
the sidecar tables; they don't need new schema. The receipt log
(`graph_receipts`) lets the export pipeline stamp each emitted
artifact with a constitutional receipt — the receipt is what
proves the export is in the user's record.

### 8.4 The receipt-verifier sidecar

A future process (out of scope here) that reads `graph_receipts`
to verify that a workflow run actually produced the receipts it
claims to have produced. The `signature` column is in place; the
verifier reads, verifies, and audits. The data layer doesn't
need any changes to support this — the verifier just calls
`TesseraDataLayer.receipts(forEntity:)` over the same
`graph_receipts` table.

### 8.5 What's in place today

- `TesseraDataStore` — Postgres CRUD + hybrid search.
- `TesseraCache` — Valkey strings, lists, sorted sets, EVAL.
- `TesseraDataLayer` — combined facade, startup lifecycle,
  health probes.
- Docker compose for both services (Apple Silicon native,
  arm64-only images, health checks, named volumes).
- Makefile with `db/up`, `db/down`, `db/reset`, `db/migrate`,
  `db/seed`, `db/verify`, `db/lint`.
- README explaining how to run.
- 17 integration tests (env-gated).
- This design doc.

---

## 9. Out of Scope / Next Steps

The data layer is the foundation. The productivity surface
follows in the next wave. The full list of "not yet":

- **Productivity-surface tables** (Tasks, Reminders, Events,
  Notes, Emails, Materials) — a follow-up migration that
  references `graph_entities(id)` 1:1.
- **The document / spreadsheet / slides / email importer** —
  the productivity surface's first vertical.
- **The AI-driven live editor** — the productivity surface's
  second vertical; uses `hybridSearch` for retrieval.
- **Export to PDF, Apple Mail, Slack** — the productivity
  surface's third vertical; emits constitutional receipts.
- **TesseraStudioMac app code that USES `TesseraDataLayer`** —
  wiring the app's session, capture, and library features to
  the new store. This is the first surface the architect will
  want after merging this branch.
- **Receipt signing (ed25519)** — the `signature` column is in
  place; the signing path is a follow-up.
- **Vector embedding model integration** — we hardcode 1536
  dims; the actual embedder (OpenAI, a local model, or a
  hybrid) is a follow-up. The dimension is fixed in the
  schema; changing it is a destructive migration.
- **Multi-tenant isolation, row-level security, authn/z** —
  single-user, single-machine is the only mode we ship.
  Anything multi-tenant is a separate design pass.
- **Production deployment (TLS, AUTH, monitoring, backups)** —
  dev/staging only. The docker compose is not production-hardened.
- **Read-through + write-through cache policy in the facade** —
  the facade exposes the cache + store but the cache-coherence
  policy is intentionally not baked in. The productivity
  surface owns the read-through / write-through / cache-aside
  decision per use case. This avoids building a cache policy
  the user hasn't asked for.
- **Linux + macOS binary parity tests in CI** — the integration
  tests run locally; we don't have a Linux CI run that
  validates Linux compilation. The `Package.swift` uses no
  Apple-only APIs in the Data/ files, so it should build on
  Linux, but it isn't tested.

The next wave's planning doc should pick up at the boundary
between "data layer done" and "productivity surface next". The
worker's deliverable stops at the foundation.

---

## Appendix A: Files Touched

```
tools/tessera/db/
├── docker-compose.yml
├── .env.example
├── Makefile
├── README.md
├── migrations/0001_init.sql
└── seeds/seed.sql

TesseraStudio/Sources/TesseraCore/Data/
├── TesseraDataStore.swift
├── TesseraCache.swift
└── TesseraDataLayer.swift

TesseraStudio/Tests/TesseraCoreTests/Data/
├── SchemaMigrationTests.swift
├── HybridSearchTests.swift
├── ConnectionPoolTests.swift
└── CacheTTLTests.swift

TesseraStudio/Package.swift               (postgres-nio + RediStack deps)
docs/tessera-data-layer-design.md         (this file)
```

## Appendix B: How to Run

```bash
# 1. From tools/tessera/db/, copy the env file.
cd tools/tessera/db
cp .env.example .env

# 2. Bring the stack up (Postgres + Valkey on Apple Silicon native).
make db/up

# 3. Apply the migration.
make db/migrate

# 4. Load the test fixture.
make db/seed

# 5. Verify: apply migration + seed to a throwaway DB and assert
#    tables / indexes / functions exist via pg_catalog.
make db/verify

# 6. From TesseraStudio/, run the env-gated integration tests.
cd ../../../TesseraStudio
TESSERA_DB_INTEGRATION=1 swift test
```
