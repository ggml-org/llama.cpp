import type { ApiStreamSession } from '$lib/types';

/**
 * Pick the running session to splice into when discoverActiveStream lists candidates for
 * a conversation. Finalized sessions are not candidates: their final content was already
 * written to the DB by the original onComplete handler, so attaching to them would replay
 * a buffer that may not match what the DB holds. In particular a continue session's buffer
 * holds only the appended deltas, not the pre continue prefix, so replaying it as a fresh
 * generation would erase the original assistant content.
 *
 * Among running sessions we tie break on the most recent started_at, which covers the
 * pathological case of multiple inferences left running on the same conversation (eg user
 * spawned two tabs).
 *
 * Returns null when no running session exists or the input is empty.
 */
export function selectActiveStream(
	sessions: ApiStreamSession[] | null | undefined
): ApiStreamSession | null {
	if (!Array.isArray(sessions) || sessions.length === 0) {
		return null;
	}
	const running = sessions.filter((s) => !s.is_done);
	if (running.length === 0) {
		return null;
	}
	return running.reduce((best, cur) => (cur.started_at > best.started_at ? cur : best));
}
