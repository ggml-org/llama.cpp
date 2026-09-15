/**
 * OpenAI-compatible protocol: llama-server, hosted OpenAI endpoints and any
 * compatible gateway. llama-server accepts a superset of the wire format, so
 * its only specialization is that nothing gets stripped.
 */

import type { ChatProtocolAdapter, ChatStreamEvent, ChatStreamReader } from './types';
import { HEADERS } from '$lib/constants';
import type { Backend } from '$lib/types';
import type { ApiChatCompletionStreamChunk } from '$lib/types/api';
import { getBackendCompat } from '$lib/utils/backend';

/**
 * llama.cpp-only chat request fields. Strict OpenAI-compatible endpoints
 * reject unknown parameters, so they are dropped for those backends.
 */
const COMPAT_ONLY_OMIT_REQUEST_FIELDS = [
	'add_generation_prompt',
	'backend_sampling',
	'cache_prompt',
	'chat_template_kwargs',
	'continue_final_message',
	'dry_allowed_length',
	'dry_base',
	'dry_multiplier',
	'dry_penalty_last_n',
	'dynatemp_exponent',
	'dynatemp_range',
	'id_slot',
	'min_p',
	'n_keep',
	'n_predict',
	'reasoning_control',
	'reasoning_format',
	'repeat_last_n',
	'repeat_penalty',
	'return_progress',
	'samplers',
	'sse_ping_interval',
	'thinking_budget_tokens',
	'timings_per_token',
	'top_k',
	'typ_p',
	'xtc_probability',
	'xtc_threshold'
];

function authHeaders(backend: Backend): Record<string, string> {
	const headers: Record<string, string> = { ...(backend.headers ?? {}) };
	const apiKey = backend.apiKey?.trim();

	if (apiKey) {
		headers[HEADERS.AUTHORIZATION] = `${HEADERS.BEARER}${apiKey}`;
	}

	return headers;
}

function buildChatRequest(
	body: Record<string, unknown>,
	backend: Backend
): Record<string, unknown> {
	if (backend.protocol === 'llama.cpp') return body;

	const compat = getBackendCompat(backend);
	const request: Record<string, unknown> = { ...body };

	for (const field of COMPAT_ONLY_OMIT_REQUEST_FIELDS) {
		delete request[field];
	}

	// compatible endpoints reject the reasoning_content message extension
	const messages = request.messages as { reasoning_content?: string }[] | undefined;

	for (const message of messages ?? []) {
		delete message.reasoning_content;
	}

	// -1 is llama.cpp's "no limit" sentinel; compatible endpoints reject it
	if (typeof request.max_tokens === 'number' && request.max_tokens <= 0) {
		delete request.max_tokens;
	}

	// newer OpenAI models require max_completion_tokens, most compatible
	// endpoints only understand max_tokens
	if (compat.maxTokensField === 'max_completion_tokens' && request.max_tokens !== undefined) {
		request.max_completion_tokens = request.max_tokens;
		delete request.max_tokens;
	}

	// a final usage chunk is what the client side timing fallback reads
	if (request.stream && compat.supportsUsageInStreaming && request.stream_options === undefined) {
		request.stream_options = { include_usage: true };
	}

	return request;
}

/**
 * Model name a payload reports. Streaming chunks carry it on the delta, final
 * responses on the message, and some gateways on metadata or the choice itself.
 */
export function extractModelName(data: unknown): string | undefined {
	const asRecord = (value: unknown): Record<string, unknown> | undefined =>
		typeof value === 'object' && value !== null ? (value as Record<string, unknown>) : undefined;
	const getTrimmedString = (value: unknown): string | undefined =>
		typeof value === 'string' && value.trim() ? value.trim() : undefined;
	const root = asRecord(data);

	if (!root) return undefined;

	const rootModel = getTrimmedString(root.model);

	if (rootModel) return rootModel;

	const firstChoice = Array.isArray(root.choices) ? asRecord(root.choices[0]) : undefined;

	if (!firstChoice) return undefined;

	const metadataModel = getTrimmedString(asRecord(firstChoice.metadata)?.model);

	if (metadataModel) return metadataModel;

	const deltaModel = getTrimmedString(asRecord(firstChoice.delta)?.model);

	if (deltaModel) return deltaModel;

	const messageModel = getTrimmedString(asRecord(firstChoice.message)?.model);

	if (messageModel) return messageModel;

	return getTrimmedString(firstChoice.model);
}

function readChunk(payload: unknown): ChatStreamEvent[] {
	if (!payload || typeof payload !== 'object') return [];

	const chunk = payload as ApiChatCompletionStreamChunk;
	const events: ChatStreamEvent[] = [];
	const model = extractModelName(chunk);

	if (chunk.id) events.push({ id: chunk.id, type: 'id' });

	if (model) events.push({ model, type: 'model' });

	if (chunk.prompt_progress) {
		events.push({ progress: chunk.prompt_progress, type: 'prompt_progress' });
	}

	if (chunk.timings) {
		events.push({
			promptProgress: chunk.prompt_progress,
			timings: chunk.timings,
			type: 'timings'
		});
	}

	if (chunk.usage) events.push({ type: 'usage', usage: chunk.usage });

	const choice = chunk.choices?.[0];
	const delta = choice?.delta;

	if (!delta) return events;

	if (delta.content) events.push({ text: delta.content, type: 'text' });

	// endpoints disagree on the reasoning field name; take the first one set
	const reasoning = delta.reasoning_content ?? delta.reasoning ?? delta.reasoning_text;

	if (reasoning) events.push({ text: reasoning, type: 'thinking' });

	if (delta.tool_calls) events.push({ deltas: delta.tool_calls, type: 'tool_calls' });

	return events;
}

export const openaiAdapter: ChatProtocolAdapter = {
	authHeaders,
	buildChatRequest,
	createStreamReader(): ChatStreamReader {
		return { readChunk };
	}
};
