-- Tessera Studio: data-layer schema (migration 0001).
--
-- The universal "one row per thing" pattern: a single `graph_entities`
-- table covers every knowledge-graph node (material, file, project,
-- chat, message, topic, person, tool_invocation, receipt, decision,
-- task, reminder, calendar_event, email, note, document, spreadsheet,
-- presentation). Productivity-surface tables (tasks/reminders/...) are
-- added in a later migration; this one focuses on the data-layer +
-- retrieval foundation the productivity surface depends on.
--
-- Why a single polymorphic table:
--   * Hybrid search (RRF over graph + vector + keyword) needs the same
--     shape across all entity types. Splitting by type means N parallel
--     UNION ALLs in the retrieval query, which kills the planner.
--   * Edges (`entity_links`) are type-agnostic; the link_type column
--     discriminates. Recursive CTEs walk the table without joins to
--     per-type metadata.
--   * Productivity columns that ARE type-specific live on a sidecar
--     jsonb column or in dedicated tables that reference graph_entities
--     1:1; we deliberately do not duplicate here.
--
-- Extensions: vector (pgvector) and pg_trgm must be loaded BEFORE the
-- tables that depend on them. pg_trgm provides the GIN operator class
-- `gin_trgm_ops` used by `idx_entities_trgm_label`. The `vector` type
-- is what backs the `embedding` column + the HNSW index.
--
-- Idempotency: `CREATE EXTENSION IF NOT EXISTS` and `CREATE TABLE IF
-- NOT EXISTS` make this migration safe to re-apply on a partially
-- migrated DB. The `hybrid_search` function is `CREATE OR REPLACE`.
--
-- We do NOT wrap this file in BEGIN/COMMIT. Each DDL statement
-- auto-commits in Postgres, and the explicit transaction wrapper
-- breaks the in-process migration path (postgres-nio's query()
-- only allows one statement per call). The `psql` / `docker compose
-- migrate` paths run the file via psql which accepts multi-statement
-- input; the in-process path runs each statement separately (see
-- TesseraDataStore.applyMigrations), which is also fine.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid()

-- graph_entities: the universal "one row per thing" table.
CREATE TABLE IF NOT EXISTS graph_entities (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type     text NOT NULL,
    subtype         text,
    label           text NOT NULL,
    body            text,
    source_url      text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),
    -- Generated tsvector: label weighted A (high), body weighted B.
    -- Weights feed into ts_rank_cd in hybrid_search (keyword_score).
    search_tsv      tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(label, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(body, '')),  'B')
    ) STORED,
    -- 1536 dims matches OpenAI text-embedding-3-small; for local
    -- embeddings we re-embed to whatever the user's model produces.
    -- The column is fixed at 1536 in this migration; the dimension
    -- switch is a follow-up (the worker report calls this out).
    embedding       vector(1536)
);

-- entity_links: typed graph edges. The recursive CTE in hybrid_search
-- walks this table.
CREATE TABLE IF NOT EXISTS entity_links (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id       uuid NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
    target_id       uuid NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
    link_type       text NOT NULL,
    weight          real NOT NULL DEFAULT 1.0,
    created_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (source_id, target_id, link_type)
);

-- graph_receipts: constitutional receipt log. The entity_id points at
-- the thing the receipt is about (could be a tool_invocation, a
-- decision, a chat message, ...). `payload` is jsonb (schema-versioned
-- INSIDE the json, e.g. "schema": "tessera.receipt.tool_invocation.v1").
-- `signature` is a 64-byte ed25519 signature; ABSENT (NULL) for now --
-- signing is a follow-up but the column exists so the schema doesn't
-- need to migrate when signing lands.
CREATE TABLE IF NOT EXISTS graph_receipts (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id       uuid NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
    receipt_type    text NOT NULL,
    payload         jsonb NOT NULL,
    signature       bytea,
    witnessed_at    timestamptz NOT NULL DEFAULT now()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_entities_type        ON graph_entities (entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_subtype     ON graph_entities (subtype) WHERE subtype IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_entities_source_url  ON graph_entities (source_url) WHERE source_url IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_entities_search_tsv  ON graph_entities USING GIN (search_tsv);
CREATE INDEX IF NOT EXISTS idx_entities_embedding   ON graph_entities USING HNSW (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_entities_trgm_label  ON graph_entities USING GIN (label gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_links_source         ON entity_links (source_id);
CREATE INDEX IF NOT EXISTS idx_links_target         ON entity_links (target_id);
CREATE INDEX IF NOT EXISTS idx_links_type           ON entity_links (link_type);

CREATE INDEX IF NOT EXISTS idx_receipts_entity      ON graph_receipts (entity_id);
CREATE INDEX IF NOT EXISTS idx_receipts_type        ON graph_receipts (receipt_type);

-- updated_at trigger (cheap, fire-and-forget; keeps the column honest
-- when callers UPDATE without explicitly setting it).
CREATE OR REPLACE FUNCTION touch_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_entities_touch_updated_at ON graph_entities;
CREATE TRIGGER trg_entities_touch_updated_at
    BEFORE UPDATE ON graph_entities
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

-- hybrid_search: the load-bearing query for the productivity surface.
-- Returns the top 25 entities reachable from `p_anchor` within
-- `p_max_depth` graph hops, ranked by Reciprocal Rank Fusion (RRF)
-- over three signals:
--
--   * graph_score:  1 / (1 + 1.5 * depth)         -- closer = higher
--   * vector_score: 1 - cosine distance           -- (e <=> emb)
--   * keyword_score: ts_rank_cd on the tsvector   -- english config
--
-- RRF weights (calibrated against the 5-entity / 4-link test fixture
-- in seeds/seed.sql -- see docs/tessera-data-layer-design.md §4.2):
--
--   graph:  0.2  -- weak in small fixtures; bump for dense graphs
--   vector: 0.5  -- dominant signal for natural-language retrieval
--   keyword: 0.3 -- narrows recall; protects against embedding drift
--
-- The +60 constants match the standard RRF offset (k in Cormack et
-- al. 2009); larger k means a slower falloff, smaller k means the top
-- rank dominates more. The RRF formula is
--   rrf(d) = sum_i w_i / (k + rank_i(d))
-- where rank_i(d) is the per-signal rank (1 = top) or NULL if the
-- signal is not present for d.
--
-- Stability: the function is STABLE (read-only, deterministic for the
-- same input set within a single statement). The recursive CTE's
-- result is independent of the order of entity_links rows because
-- depth is the only graph-signal feature we read here.
CREATE OR REPLACE FUNCTION hybrid_search(
    p_anchor         uuid,
    p_query_text     text,
    p_query_embedding vector(1536),
    p_max_depth      int DEFAULT 3
) RETURNS TABLE (
    entity_id        uuid,
    entity_type      text,
    label            text,
    body             text,
    graph_score      real,
    vector_score     real,
    keyword_score    real,
    rrf_score        real
) AS $$
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
    COALESCE(ts_rank_cd(e.search_tsv, plainto_tsquery('english', p_query_text)), 0.0)::real
                                                                               AS keyword_score,
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
$$ LANGUAGE sql STABLE;
