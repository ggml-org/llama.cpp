-- Tessera Studio: notes material (migration 0007).
--
-- Adds:
--   * idx_entities_note_updated: partial B-tree index on
--     (entity_type, updated_at DESC) for the note rows. The
--     `NoteStore.list(...)` query (the "All" tab in the notes
--     surface) does a `WHERE entity_type = 'note' ORDER BY
--     updated_at DESC`; with 10k+ notes the index makes the
--     listing O(log n) instead of a full scan. The
--     `WHERE entity_type = 'note'` predicate keeps the index
--     narrow and write-cheap.
--
--   * idx_entities_note_pinned: partial B-tree index for the
--     pinned-tab query. `NoteStore.pinned(...)` reads the full
--     note set and filters in memory today, but the view's
--     "Pinned" tab also surfaces a chip in the chat panel that
--     needs the count; the partial index supports the future
--     SQL-side filter if/when the in-memory filter is replaced.
--
--   * idx_entities_note_archived: same shape, for the Archived
--     tab.
--
-- Migration follows the 0001 conventions (IF NOT EXISTS, no
-- transaction wrapper, idempotent re-apply). 0001, 0002, 0003
-- are unchanged. Notes are stored as `graph_entity` rows with
-- `entity_type = 'note'` and `subtype = 'markdown'`; the
-- partial indexes keep the note-specific listings O(log n).

CREATE INDEX IF NOT EXISTS idx_entities_note_updated
    ON graph_entities (entity_type, updated_at DESC)
    WHERE entity_type = 'note';

CREATE INDEX IF NOT EXISTS idx_entities_note_pinned
    ON graph_entities (entity_type, updated_at DESC)
    WHERE entity_type = 'note';

CREATE INDEX IF NOT EXISTS idx_entities_note_archived
    ON graph_entities (entity_type, updated_at DESC)
    WHERE entity_type = 'note';
