-- Tessera Studio: Code material surface (migration 0009).
--
-- Adds two partial B-tree indexes for the `code` entity
-- type. The Code surface (per
-- `docs/tessera-productivity-materials-code-design.md` §3)
-- stores source files as `graph_entity` rows with
-- `entity_type = 'code'`, `subtype` = the language tag,
-- and `body->>'path'` / `body->>'language'` for the
-- per-language filter and the "files in this directory"
-- query.
--
--   * idx_entities_code_path: (entity_type, body->>'path')
--     WHERE entity_type = 'code'. The CodeFileWatcher's
--     "is this path in the index?" check is O(log n)
--     instead of a sequential scan; the sidebar's
--     path-substring filter uses the same index.
--
--   * idx_entities_code_language: (entity_type, body->>'language')
--     WHERE entity_type = 'code'. The language-filter
--     dropdown ("show only Python files") hits the
--     index. The data layer's `hybrid_search` doesn't
--     benefit (it does a graph walk), but the Code
--     surface's per-language list view does.
--
-- The indexes are partial (the `WHERE entity_type = 'code'`
-- clause keeps them narrow and write-cheap) and follow
-- the 0001 / 0003 conventions (IF NOT EXISTS, no
-- transaction wrapper, idempotent re-apply).

CREATE INDEX IF NOT EXISTS idx_entities_code_path
    ON graph_entities (entity_type, (body->>'path'))
    WHERE entity_type = 'code';

CREATE INDEX IF NOT EXISTS idx_entities_code_language
    ON graph_entities (entity_type, (body->>'language'))
    WHERE entity_type = 'code';
