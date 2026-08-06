-- Tessera Studio: reminders material (migration 0005).
--
-- Adds an index that supports the Reminders surface's two hot
-- read paths:
--
--   * "Upcoming" filter — the list view that shows reminders
--     whose trigger time is now <= t <= now + 24h, sorted by
--     triggerAt ascending. The list view's SQL is
--        SELECT ... FROM graph_entities
--         WHERE entity_type = 'reminder'
--           AND (body->>'triggerAt')::timestamptz BETWEEN now() AND now() + interval '24 hours'
--         ORDER BY (body->>'triggerAt')::timestamptz ASC
--     The expression index on (entity_type, body->>'triggerAt')
--     with the partial WHERE keeps the index narrow and turns
--     the range scan into an O(log n + k) seek.
--
--   * Snoozed filter — reminders whose `snoozedUntil` is set
--     and in the future. The view computes this in code (not
--     SQL), so the index is only the triggerAt one; the snoozed
--     filter is a small set after the in-memory bucketing.
--
-- Migration follows the 0001 conventions (IF NOT EXISTS, no
-- transaction wrapper, idempotent re-apply). 0001-0004 are
-- unchanged; 0004 is reserved for a future phase (likely the
-- tasks / calendar material that reminders link to).

CREATE INDEX IF NOT EXISTS idx_entities_reminder_trigger
    ON graph_entities (entity_type, body->>'triggerAt')
    WHERE entity_type = 'reminder';
