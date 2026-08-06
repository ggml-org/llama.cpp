-- Tessera Studio: email material (migration 0008).
--
-- Adds two partial B-tree indexes for the email surface:
--
--   * idx_entities_email_received: B-tree on
--     (entity_type, body->>'receivedAt' DESC) where
--     entity_type = 'email'. The email list view sorts by
--     receivedAt DESC; with thousands of emails the index
--     makes the list a cheap index scan instead of a
--     seq scan over all graph_entities. The partial
--     predicate keeps the index narrow (only email
--     rows participate) so writes are cheap.
--
--   * idx_entities_email_thread: B-tree on
--     (entity_type, body->>'threadID') where
--     entity_type = 'email'. The thread grouping query
--     (phase 5's EmailStore.threads()) filters by
--     threadID; the index makes the bucket-collect
--     O(matches) instead of O(total emails).
--
-- We index body->>'receivedAt' and body->>'threadID' (the
-- JSON path extractor) rather than the top-level
-- `label` because emails are heterogeneous (subject line
-- in label, full RFC 5322 fields in body). The `label`
-- index from migration 0001/0003 still serves the
-- cross-cutting search.
--
-- Migration follows the 0001 conventions (IF NOT EXISTS,
-- no transaction wrapper, idempotent re-apply). 0001 +
-- 0002 + 0003 are unchanged.

CREATE INDEX IF NOT EXISTS idx_entities_email_received
    ON graph_entities (entity_type, (body->>'receivedAt') DESC)
    WHERE entity_type = 'email';

CREATE INDEX IF NOT EXISTS idx_entities_email_thread
    ON graph_entities (entity_type, (body->>'threadID'))
    WHERE entity_type = 'email' AND (body->>'threadID') IS NOT NULL;
