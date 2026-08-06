-- Tessera Studio: calendar material (migration 0006).
--
-- Adds:
--   * idx_entities_event_start: partial B-tree index on
--     (entity_type, body->>'startAt') for the calendar
--     event rows. The event body encodes `startAt` as an
--     ISO-8601 UTC string, so the text index sorts in
--     chronological order and supports the server-side
--     range scan the data layer facade will use for
--     "events between A and B" (the v1 CalendarStore
--     loads + filters client-side; the index is in place
--     for the facade-level range query). The
--     `WHERE entity_type = 'calendar_event'` predicate
--     keeps the index narrow and write-cheap, matching
--     the 0003 contacts convention.
--
-- Migration follows the 0001 conventions (IF NOT EXISTS,
-- no transaction wrapper, idempotent re-apply). 0001 -
-- 0005 are unchanged.

CREATE INDEX IF NOT EXISTS idx_entities_event_start
    ON graph_entities (entity_type, body->>'startAt')
    WHERE entity_type = 'calendar_event';
