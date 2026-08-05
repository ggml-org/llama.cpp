-- Tessera Studio: productivity-surface schema (migration 0002).
--
-- Adds two tables on top of 0001_init.sql:
--
--   * receipt_chain: the per-document linear ordering of receipts.
--     The (document_id, chain_index) primary key gives O(log n)
--     lookups by position; the receipt_id FK ties each chain
--     entry to the constitutional receipt in graph_receipts.
--
--   * chat_queues: the per-document chat-queue state (one JSONB
--     blob per document). The queue is queryable as a single
--     row keyed by document_id; the agent's hybrid_search picks
--     up the queue when assembling context.
--
-- Both tables are additive: 0001 is unchanged. The migration
-- follows the 0001 conventions (IF NOT EXISTS, no transaction
-- wrapper, idempotent re-apply).
--
-- Indexes:
--   * idx_receipt_chain_doc is the descending index that
--     `DocumentStore.history(of:limit:)` uses to fetch the
--     most recent N receipts for a document.
--   * idx_chat_queues_updated_at is unused in v1 but supports
--     the future "stale queue" cleanup job.

-- receipt_chain: per-document ordering of receipts.
-- chain_index is the monotonic position; prior_receipt_id is
-- denormalized on the receipt payload (graph_receipts.payload)
-- for forward walks. We index DESC for "newest first" reads.
CREATE TABLE IF NOT EXISTS receipt_chain (
    document_id   uuid NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
    chain_index   bigint NOT NULL,
    receipt_id    uuid NOT NULL REFERENCES graph_receipts(id) ON DELETE RESTRICT,
    created_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (document_id, chain_index)
);

CREATE INDEX IF NOT EXISTS idx_receipt_chain_doc
    ON receipt_chain (document_id, chain_index DESC);

-- chat_queues: per-document chat-panel queue.
-- Items is a JSONB array of ChatQueueItem records. The chat
-- panel's drag-to-reorder and match-and-supersede operations
-- rewrite the whole array (Phase 3 -- out of scope for the
-- Phase 1 worker, which only persists the data model).
CREATE TABLE IF NOT EXISTS chat_queues (
    document_id  uuid PRIMARY KEY REFERENCES graph_entities(id) ON DELETE CASCADE,
    items        jsonb NOT NULL DEFAULT '[]'::jsonb,
    updated_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chat_queues_updated_at
    ON chat_queues (updated_at);
