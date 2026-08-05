-- Tessera Studio: data-layer test fixture.
--
-- Designed to be applied by `make db/seed`. Wipes the three tables
-- (cascade is enough), then inserts 5 entities + 4 links + 2 receipts
-- forming a small knowledge graph anchored on a "Project Atlas"
-- chat message. The embeddings are deterministic 1536-dim vectors
-- derived from the entity id (NOT real model embeddings -- the goal
-- is reproducibility, not semantic accuracy). The keyword tsvectors
-- are real (set by the GENERATED column from the label/body text).
--
-- The anchor for hybrid_search tests is `b0000000-0000-0000-0000-
-- 000000000001` (the chat message). The expected reachable set within
-- depth 3 is the full graph; the expected top-1 by vector score is
-- the entity whose embedding is the closest in cosine distance to
-- the query vector used in HybridSearchTests.
--
-- This file is idempotent: TRUNCATE then INSERT. Safe to re-seed.

BEGIN;

TRUNCATE TABLE graph_receipts, entity_links, graph_entities RESTART IDENTITY CASCADE;

-- ---------- graph_entities ----------
-- Entity 1: the chat message (anchor for hybrid_search).
INSERT INTO graph_entities (id, entity_type, subtype, label, body, source_url, embedding)
VALUES (
    'b0000000-0000-0000-0000-000000000001',
    'chat', 'message',
    'Atlas: kickoff message',
    'Project Atlas kickoff. Goals: build a portable knowledge graph, run hybrid search.',
    NULL,
    -- 1536 dims; index 0 = 1.0, rest = 0.0
    (SELECT array_agg(CASE WHEN i = 0 THEN 1.0 ELSE 0.0 END)::vector(1536)
       FROM generate_series(0, 1535) AS i)
);

-- Entity 2: the project (one hop from anchor).
INSERT INTO graph_entities (id, entity_type, subtype, label, body, source_url, embedding)
VALUES (
    'b0000000-0000-0000-0000-000000000002',
    'project', NULL,
    'Project Atlas',
    'Long-running internal initiative: portable, local-first knowledge graph with hybrid retrieval.',
    'https://example.invalid/projects/atlas',
    (SELECT array_agg(CASE WHEN i = 1 THEN 1.0 ELSE 0.0 END)::vector(1536)
       FROM generate_series(0, 1535) AS i)
);

-- Entity 3: the topic (two hops from anchor via project).
INSERT INTO graph_entities (id, entity_type, subtype, label, body, source_url, embedding)
VALUES (
    'b0000000-0000-0000-0000-000000000003',
    'topic', NULL,
    'Hybrid retrieval',
    'Reciprocal rank fusion over graph, vector, and keyword signals. Weights tuned on dev fixtures.',
    NULL,
    (SELECT array_agg(CASE WHEN i = 2 THEN 1.0 ELSE 0.0 END)::vector(1536)
       FROM generate_series(0, 1535) AS i)
);

-- Entity 4: a person (two hops via project -> topic; not directly reachable in 2 hops
-- unless we add a link; this one IS reachable in 3 hops via the document link).
INSERT INTO graph_entities (id, entity_type, subtype, label, body, source_url, embedding)
VALUES (
    'b0000000-0000-0000-0000-000000000004',
    'person', NULL,
    'Dr. Lin',
    'Research lead. Owns the hybrid retrieval research thread and the dev fixture corpus.',
    NULL,
    (SELECT array_agg(CASE WHEN i = 3 THEN 1.0 ELSE 0.0 END)::vector(1536)
       FROM generate_series(0, 1535) AS i)
);

-- Entity 5: a document (one hop from the topic). With depth=3, the doc
-- is reachable from the anchor.
INSERT INTO graph_entities (id, entity_type, subtype, label, body, source_url, embedding)
VALUES (
    'b0000000-0000-0000-0000-000000000005',
    'document', NULL,
    'Atlas hybrid retrieval design',
    'Design doc covering RRF weights, depth limits, and the test fixture used to calibrate them.',
    'https://example.invalid/docs/atlas-hybrid.md',
    (SELECT array_agg(CASE WHEN i = 4 THEN 1.0 ELSE 0.0 END)::vector(1536)
       FROM generate_series(0, 1535) AS i)
);

-- ---------- entity_links ----------
-- 1 -> 2: chat references project
INSERT INTO entity_links (source_id, target_id, link_type, weight)
VALUES ('b0000000-0000-0000-0000-000000000001', 'b0000000-0000-0000-0000-000000000002', 'references', 1.0);

-- 2 -> 3: project has topic
INSERT INTO entity_links (source_id, target_id, link_type, weight)
VALUES ('b0000000-0000-0000-0000-000000000002', 'b0000000-0000-0000-0000-000000000003', 'has_topic', 1.0);

-- 3 -> 4: topic involves person
INSERT INTO entity_links (source_id, target_id, link_type, weight)
VALUES ('b0000000-0000-0000-0000-000000000003', 'b0000000-0000-0000-0000-000000000004', 'involves', 1.0);

-- 3 -> 5: topic has document
INSERT INTO entity_links (source_id, target_id, link_type, weight)
VALUES ('b0000000-0000-0000-0000-000000000003', 'b0000000-0000-0000-0000-000000000005', 'has_document', 1.0);

-- ---------- graph_receipts ----------
-- Two receipts on the chat message (one for the message itself, one
-- for a tool invocation). The signature column is NULL because the
-- signing path is a follow-up; the column exists so the schema does
-- not need to migrate when signing lands.
INSERT INTO graph_receipts (entity_id, receipt_type, payload, signature)
VALUES (
    'b0000000-0000-0000-0000-000000000001',
    'chat_message',
    jsonb_build_object(
        'schema', 'tessera.receipt.chat_message.v1',
        'author', 'Dr. Lin',
        'channel', 'atlas',
        'sent_at', '2026-08-05T09:00:00Z'
    ),
    NULL
);

INSERT INTO graph_receipts (entity_id, receipt_type, payload, signature)
VALUES (
    'b0000000-0000-0000-0000-000000000002',
    'project_created',
    jsonb_build_object(
        'schema', 'tessera.receipt.project_created.v1',
        'created_by', 'Dr. Lin',
        'tags', jsonb_build_array('hybrid-retrieval', 'knowledge-graph', 'local-first')
    ),
    NULL
);

COMMIT;
