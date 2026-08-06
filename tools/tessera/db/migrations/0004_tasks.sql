-- Tessera Studio: tasks material (migration 0004).
--
-- Adds the partial B-tree indexes that back the Tasks surface
-- (§12.2 of docs/tessera-productivity-design.md):
--
--   * idx_entities_task_due: index over
--     (entity_type, body->>'dueAt') for the task rows. The
--     Today / Upcoming list filters use dueAt as the primary
--     axis. The `WHERE entity_type = 'task'` predicate keeps
--     the index narrow and write-cheap.
--
--   * idx_entities_task_list: index over
--     (entity_type, body->>'list') for the task rows. The
--     Inbox / Today / Upcoming / Anytime / Someday lists
--     filter by the materialized list name. The user can
--     move a task between lists (the receipt chain records
--     the move), so the index has to be cheap to update.
--
-- The data layer stores tasks as `graph_entity` rows with
-- `entity_type = 'task'` and `subtype` = the current list
-- name. The body JSONB carries the Task struct (title,
-- notes, dueAt, completedAt, priority, tags, linkedEntityIDs,
-- sourceURL). We index on `body->>'list'` in addition to
-- `subtype` so we can survive the moment between the list
-- being written to the body and the entity's `subtype`
-- being updated (the two are kept in sync by `TaskStore`).
--
-- Migration follows the 0001 conventions (IF NOT EXISTS, no
-- transaction wrapper, idempotent re-apply). 0001 + 0002 +
-- 0003 are unchanged.

CREATE INDEX IF NOT EXISTS idx_entities_task_due
    ON graph_entities ((body->>'dueAt'))
    WHERE entity_type = 'task';

CREATE INDEX IF NOT EXISTS idx_entities_task_list
    ON graph_entities ((body->>'list'))
    WHERE entity_type = 'task';

CREATE INDEX IF NOT EXISTS idx_entities_task_subtype
    ON graph_entities (entity_type, subtype)
    WHERE entity_type = 'task';
