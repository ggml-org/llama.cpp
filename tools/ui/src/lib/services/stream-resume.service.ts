/**
 * Stream resume persistence and reconnection helper.
 *
 * Tracks the running byte count for an in flight streaming generation per
 * conversation_id, so a later visit can resume the SSE replay at the right
 * offset. The conversation_id is the session identity end to end (server map,
 * client localStorage, /v1/stream/<conv_id> routes), no extra opaque token.
 */

import { streamIdentity } from '$lib/utils/stream-identity';
import { getAuthHeaders } from '$lib/utils/api-headers';

interface ResumableStreamState {
	bytesReceived: number;
	updatedAt: number;

	// model frozen at POST time, lets a reload rebuild the exact conv::model identity the
	// server keyed the session under. null when the POST carried no explicit model
	model?: string | null;
}

const STORAGE_PREFIX = 'llamacpp.stream.resume.';

function storageKey(conversationId: string): string {
	return STORAGE_PREFIX + conversationId;
}

export function saveStreamState(
	conversationId: string,
	bytesReceived: number,
	model?: string | null
): void {
	if (!conversationId) return;
	try {
		const state: ResumableStreamState = {
			bytesReceived,
			updatedAt: Date.now(),
			model: model ?? null
		};
		localStorage.setItem(storageKey(conversationId), JSON.stringify(state));
	} catch {
		// localStorage may be full or disabled, silently ignore
	}
}

export function getStreamState(conversationId: string): ResumableStreamState | null {
	if (!conversationId) return null;
	try {
		const raw = localStorage.getItem(storageKey(conversationId));
		if (!raw) return null;
		const parsed = JSON.parse(raw) as ResumableStreamState;
		if (!parsed || typeof parsed.bytesReceived !== 'number') return null;
		return parsed;
	} catch {
		return null;
	}
}

export function clearStreamState(conversationId: string): void {
	if (!conversationId) return;
	try {
		localStorage.removeItem(storageKey(conversationId));
	} catch {
		// nothing to do
	}
}

/**
 * Rebuild the stream identity for a resume. The model persisted at POST time wins, including a
 * stored null which means the POST carried no explicit model so the identity stays the bare
 * conv id. Only fall back to the caller supplied current model when nothing was persisted, e.g.
 * a state written before the model field existed.
 */
export function resumeStreamIdentity(
	conversationId: string,
	state: ResumableStreamState | null,
	fallbackModel: string | null
): string {
	const model = state && state.model !== undefined ? state.model : fallbackModel;
	return streamIdentity(conversationId, model);
}

/**
 * Reconnect to an interrupted stream for this conversation. Returns the fetch
 * Response so the existing SSE parser can drain it just like a fresh stream.
 * The caller is expected to feed the running byte count back via
 * saveStreamState as more data flows.
 *
 * The server returns 200 with text/event-stream on success, 404 if no session
 * exists for the conv_id (already evicted or never created), and 400 if the
 * requested offset is below the dropped prefix (buffer cap was hit and head
 * bytes were lost).
 */
export async function resumeStream(
	conversationId: string,
	signal?: AbortSignal,
	model?: string | null
): Promise<Response | null> {
	if (!conversationId) return null;
	const state = getStreamState(conversationId);
	const from = state?.bytesReceived ?? 0;
	const id = streamIdentity(conversationId, model);
	const url = `./v1/stream/${encodeURIComponent(id)}?from=${from}`;
	return await fetch(url, { method: 'GET', signal, headers: getAuthHeaders() });
}
