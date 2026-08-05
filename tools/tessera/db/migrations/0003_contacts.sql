-- Tessera Studio: contacts material + graph view (migration 0003).
--
-- Adds:
--   * idx_entities_contact_name: partial B-tree index on
--     (entity_type, label) for the contact rows. The contact
--     store's `search(matching:)` query does a case-insensitive
--     prefix match on `label`; with 10k+ contacts the index
--     makes that lookup O(log n) instead of a full scan. The
--     `WHERE entity_type = 'contact'` predicate keeps the
--     index narrow and write-cheap.
--
--   * idx_entities_entity_type: general-purpose index on
--     entity_type. The graph view's "load every entity" path
--     and the contact list query both filter by entity_type;
--     this index supports both. (The data layer's `hybrid_search`
--     doesn't benefit because it does a graph walk; the
--     contact list query does.)
--
--   * idx_entity_links_source / idx_entity_links_target:
--     indexes on the link endpoints. The graph view's
--     adjacency list walks these to build its in-memory
--     graph; without the indexes a `WHERE source_id = ...`
--     or `WHERE target_id = ...` does a sequential scan.
--
-- Migration follows the 0001 conventions (IF NOT EXISTS, no
-- transaction wrapper, idempotent re-apply). 0001 + 0002 are
-- unchanged.

CREATE INDEX IF NOT EXISTS idx_entities_contact_name
    ON graph_entities (entity_type, label)
    WHERE entity_type = 'contact';

CREATE INDEX IF NOT EXISTS idx_entities_entity_type
    ON graph_entities (entity_type);

CREATE INDEX IF NOT EXISTS idx_entity_links_source
    ON entity_links (source_id);

CREATE INDEX IF NOT EXISTS idx_entity_links_target
    ON entity_links (target_id);
