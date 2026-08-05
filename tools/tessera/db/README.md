# Tessera Studio: Data Layer (Postgres + Valkey)

The Tessera Studio data layer is a server-side store that sits **alongside** SwiftData
(it's not a replacement). The local SwiftData store remains the on-device app store;
Postgres + Valkey provide:

- **Postgres 16 + pgvector + pg_trgm** — durable knowledge graph, constitutional
  receipts, materials slices, full-text search, vector embeddings.
- **Valkey 7** — ephemeral cache: agent scratchpad, capture cache, decay windows,
  session state, idempotency keys.

The design is privacy-first: no SaaS, no API keys, no third-party hosted services.
Both services run locally and are Apple Silicon native (`platform: linux/arm64`).

This directory hosts the dev/staging infra for the data layer. The full design is
in [`docs/tessera-data-layer-design.md`](../../../docs/tessera-data-layer-design.md).

## Quick start

```bash
# 1. From tools/tessera/db/, copy the env file (or let db/up do it for you).
cp .env.example .env

# 2. Bring the stack up.
make db/up

# 3. Apply migrations (creates extensions, tables, indexes, hybrid_search function).
make db/migrate

# 4. Load the 5-entity / 4-link test fixture.
make db/seed

# 5. (Optional) Run a one-shot verify pass: migrations + seed into a throwaway DB
#    and assert tables/indexes/functions exist via pg_catalog.
make db/verify
```

To bring it down (keeping the data volumes):

```bash
make db/down
```

To nuke the volumes and start over:

```bash
make db/reset     # DESTRUCTIVE
```

## Layout

```
tools/tessera/db/
├── docker-compose.yml     -- Postgres 16 + Valkey 7 (arm64)
├── .env.example           -- dev credentials
├── Makefile               -- db/up | db/down | db/reset | db/migrate | db/seed | db/verify
├── migrations/
│   └── 0001_init.sql      -- extensions, tables, indexes, hybrid_search function
├── seeds/
│   └── seed.sql           -- 5 entities, 4 links, 2 receipts (deterministic fixture)
└── README.md              -- this file
```

## Schema (0001_init.sql)

Three tables, no SaaS-side extensions other than `vector`, `pg_trgm`, and `pgcrypto`:

- `graph_entities` — universal "one row per thing" table. The `entity_type` column
  discriminates material / file / project / chat / message / topic / person /
  tool_invocation / receipt / decision / task / reminder / calendar_event / email /
  note / document / spreadsheet / presentation. Productivity-surface-specific
  columns (due dates, calendar windows, etc.) live in dedicated tables that
  reference `graph_entities` 1:1; the polymorphic table here stays lean.
- `entity_links` — typed graph edges. The recursive CTE in `hybrid_search` walks
  this table.
- `graph_receipts` — constitutional receipt log. The `entity_id` points at the
  thing the receipt is about. `payload` is jsonb (schema-versioned in the json).
  `signature` is a nullable bytea (64-byte ed25519); NULL for now because the
  signing path is a follow-up, but the column exists so the schema does not
  need to migrate when signing lands.

Indexes: B-tree on `entity_type`, `subtype`, `source_url`; GIN on the generated
`search_tsv`; **HNSW** on the `embedding` column (pgvector `vector_cosine_ops`);
GIN trigram on `label` for typo-tolerant autocomplete.

## Hybrid search (RRF)

`hybrid_search(p_anchor, p_query_text, p_query_embedding, p_max_depth)` is the
load-bearing query for the productivity surface. It walks the graph from
`p_anchor` up to `p_max_depth` hops, then ranks the reachable set by Reciprocal
Rank Fusion across three signals:

| Signal | Weight | Function |
|---|---|---|
| Graph (depth) | 0.2 | `1 / (1 + 1.5 * depth)` |
| Vector (cosine) | 0.5 | `1 - (embedding <=> query)` |
| Keyword (ts_rank) | 0.3 | `ts_rank_cd(search_tsv, plainto_tsquery('english', q))` |

Weights are calibrated against the 5-entity / 4-link test fixture. With a
sparser graph, the graph weight can be increased; see
[`docs/tessera-data-layer-design.md` §4.2](../../../docs/tessera-data-layer-design.md).

## Hard rules

- The `embedding` column is fixed at 1536 dims. Re-embedding for other models
  is a follow-up.
- No raw SQL or Redis commands live outside `TesseraDataStore.swift` /
  `TesseraCache.swift` in the Swift client. The rest of the app depends on
  `TesseraDataLayer`, not on the client libraries.
- All integration tests are env-gated (`TESSERA_DB_INTEGRATION=1`) so
  `swift test` works without a running DB.

## Troubleshooting

- **`pg_isready` fails on `db/migrate`**: Postgres is still booting. The
  Makefile retries for ~30s. If it still fails, `make db/down && make db/up`
  to recreate.
- **Connection refused on localhost:5432**: the compose stack is not up.
  `make db/up` brings it up.
- **pgvector not found in migration**: the `pgvector/pgvector:pg16` image
  ships with vector + pg_trgm. If you're running outside the compose stack
  (e.g. a Homebrew Postgres), `brew install pgvector` separately.
