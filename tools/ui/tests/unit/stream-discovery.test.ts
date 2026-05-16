import { describe, expect, it } from 'vitest';
import { selectActiveStream } from '$lib/services/stream-discovery.service';
import type { ApiStreamSession } from '$lib/types';

function makeSession(overrides: Partial<ApiStreamSession>): ApiStreamSession {
	return {
		conversation_id: 'conv',
		is_done: true,
		total_bytes: 0,
		started_at: 0,
		completed_at: 0,
		...overrides
	};
}

describe('selectActiveStream', () => {
	it('returns null on empty input', () => {
		expect(selectActiveStream([])).toBeNull();
	});

	it('returns null on null or undefined input', () => {
		expect(selectActiveStream(null)).toBeNull();
		expect(selectActiveStream(undefined)).toBeNull();
	});

	it('returns the single session when it is running', () => {
		const s = makeSession({ conversation_id: 'only', is_done: false, started_at: 42 });
		expect(selectActiveStream([s])).toBe(s);
	});

	it('returns null when the single session is finalized', () => {
		const s = makeSession({ conversation_id: 'only', is_done: true, started_at: 42 });
		expect(selectActiveStream([s])).toBeNull();
	});

	it('prefers a still running session over a finalized one regardless of started_at', () => {
		const finalized = makeSession({ conversation_id: 'old', is_done: true, started_at: 1000 });
		const running = makeSession({ conversation_id: 'new', is_done: false, started_at: 10 });
		expect(selectActiveStream([finalized, running])?.conversation_id).toBe('new');
		expect(selectActiveStream([running, finalized])?.conversation_id).toBe('new');
	});

	it('among running sessions, picks the most recently started one', () => {
		const a = makeSession({ conversation_id: 'a', is_done: false, started_at: 100 });
		const b = makeSession({ conversation_id: 'b', is_done: false, started_at: 200 });
		const c = makeSession({ conversation_id: 'c', is_done: false, started_at: 150 });
		expect(selectActiveStream([a, b, c])?.conversation_id).toBe('b');
		expect(selectActiveStream([c, a, b])?.conversation_id).toBe('b');
	});

	it('returns null when all sessions are finalized, the DB already holds the content', () => {
		const a = makeSession({ conversation_id: 'a', is_done: true, started_at: 10 });
		const b = makeSession({ conversation_id: 'b', is_done: true, started_at: 30 });
		const c = makeSession({ conversation_id: 'c', is_done: true, started_at: 20 });
		expect(selectActiveStream([a, b, c])).toBeNull();
	});

	it('keeps the first match on ties when both are running with identical started_at', () => {
		// reduce visits left to right, the initial accumulator stays unless a strictly greater value appears
		const a = makeSession({ conversation_id: 'first', is_done: false, started_at: 50 });
		const b = makeSession({ conversation_id: 'second', is_done: false, started_at: 50 });
		expect(selectActiveStream([a, b])?.conversation_id).toBe('first');
	});

	it('handles a typical realistic mix: two finalized old, one freshly running, one freshly finalized', () => {
		const old1 = makeSession({ conversation_id: 'old1', is_done: true, started_at: 100 });
		const old2 = makeSession({ conversation_id: 'old2', is_done: true, started_at: 200 });
		const freshFin = makeSession({ conversation_id: 'freshFin', is_done: true, started_at: 500 });
		const running = makeSession({ conversation_id: 'running', is_done: false, started_at: 400 });
		expect(selectActiveStream([old1, old2, freshFin, running])?.conversation_id).toBe('running');
	});
});
