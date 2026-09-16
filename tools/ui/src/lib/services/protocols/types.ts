/**
 * Backend protocol adapters.
 *
 * A backend speaks one wire protocol. ChatService owns the transport (fetch,
 * SSE framing, resume offsets) and delegates the parts that differ per
 * protocol here: credential headers, request shaping and stream decoding.
 *
 * Decoding is per-stream: a reader keeps per-stream state, so it must not be
 * shared between concurrent requests.
 */

import type { Backend } from '$lib/types';
import type { ApiChatCompletionToolCallDelta, ApiChatCompletionUsage } from '$lib/types/api';
import type { ChatMessagePromptProgress, ChatMessageTimings } from '$lib/types/chat';

/** One canonical delta decoded from a backend's stream payload. */
export type ChatStreamEvent =
	| { type: 'done' }
	| { type: 'error'; message: string }
	| { type: 'id'; id: string }
	| { type: 'model'; model: string }
	| { type: 'prompt_progress'; progress: ChatMessagePromptProgress }
	| { type: 'text'; text: string }
	| { type: 'thinking'; text: string }
	| { type: 'timings'; timings: ChatMessageTimings; promptProgress?: ChatMessagePromptProgress }
	| { type: 'tool_calls'; deltas: ApiChatCompletionToolCallDelta[] }
	| { type: 'usage'; usage: ApiChatCompletionUsage };

/** Decodes one stream's payloads. Create one per request. */
export interface ChatStreamReader {
	/** Canonical deltas for one parsed SSE payload. */
	readChunk(payload: unknown): ChatStreamEvent[];
}

/** Wire mapping for one protocol. */
export interface ChatProtocolAdapter {
	/** Credential and protocol-required headers for a backend. */
	authHeaders(backend: Backend): Record<string, string>;
	/** Rewrite the canonical request body into this protocol's wire body. */
	buildChatRequest(body: Record<string, unknown>, backend: Backend): Record<string, unknown>;
	createStreamReader(): ChatStreamReader;
}
