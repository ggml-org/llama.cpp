/**
 * Anthropic Messages protocol.
 *
 * The Messages API differs from the OpenAI shape in three ways that matter
 * here: the system prompt is a top level field, messages carry typed content
 * blocks, and tool calls are content blocks rather than a message field.
 */

import type { ChatProtocolAdapter, ChatStreamEvent, ChatStreamReader } from './types';
import { ANTHROPIC_API_VERSION, HEADERS } from '$lib/constants';
import { ContentPartType, MessageRole } from '$lib/enums';
import type { Backend } from '$lib/types';
import type {
	ApiChatCompletionTool,
	ApiChatCompletionUsage,
	ApiChatMessageContentPart,
	ApiChatMessageData
} from '$lib/types/api';

/** Output cap sent when the user did not configure one; the field is required. */
const DEFAULT_MAX_TOKENS = 4096;

interface AnthropicBlock {
	type: string;
	[key: string]: unknown;
}

interface AnthropicTool {
	description?: string;
	input_schema: Record<string, unknown>;
	name: string;
}

interface AnthropicStreamEvent {
	type?: string;
	index?: number;
	content_block?: { type?: string; id?: string; name?: string };
	delta?: { type?: string; text?: string; thinking?: string; partial_json?: string };
	error?: { message?: string };
	message?: {
		id?: string;
		model?: string;
		usage?: Record<string, unknown>;
	};
	usage?: Record<string, unknown>;
}

function authHeaders(backend: Backend): Record<string, string> {
	const headers: Record<string, string> = {
		...(backend.headers ?? {}),
		[HEADERS.ANTHROPIC_BROWSER_ACCESS]: 'true',
		[HEADERS.ANTHROPIC_VERSION]: ANTHROPIC_API_VERSION
	};
	const apiKey = backend.apiKey?.trim();

	if (apiKey) {
		headers[HEADERS.ANTHROPIC_API_KEY] = apiKey;
	}

	return headers;
}

function textOf(content: string | ApiChatMessageContentPart[]): string {
	if (typeof content === 'string') return content;

	return content
		.filter((part) => part.type === ContentPartType.TEXT)
		.map((part) => part.text ?? '')
		.join('');
}

function imageBlock(url: string): AnthropicBlock {
	const dataUrl = url.match(/^data:([^;,]+);base64,(.*)$/s);

	if (dataUrl) {
		return { source: { data: dataUrl[2], media_type: dataUrl[1], type: 'base64' }, type: 'image' };
	}

	return { source: { type: 'url', url }, type: 'image' };
}

function userBlocks(content: string | ApiChatMessageContentPart[]): AnthropicBlock[] {
	if (typeof content === 'string') {
		return content ? [{ text: content, type: 'text' }] : [];
	}

	return content.flatMap((part): AnthropicBlock[] => {
		if (part.type === ContentPartType.TEXT) {
			return part.text ? [{ text: part.text, type: 'text' }] : [];
		}

		if (part.type === ContentPartType.IMAGE_URL && part.image_url?.url) {
			return [imageBlock(part.image_url.url)];
		}

		// audio and video have no Messages API equivalent
		return [];
	});
}

function assistantBlocks(message: ApiChatMessageData): AnthropicBlock[] {
	const blocks: AnthropicBlock[] = [];
	const text = textOf(message.content);

	if (text) blocks.push({ text, type: 'text' });

	for (const call of message.tool_calls ?? []) {
		let input: unknown = {};

		try {
			input = call.function?.arguments ? JSON.parse(call.function.arguments) : {};
		} catch {
			input = {};
		}

		blocks.push({ id: call.id, input, name: call.function?.name, type: 'tool_use' });
	}

	return blocks;
}

function convertTools(tools: unknown): AnthropicTool[] | undefined {
	if (!Array.isArray(tools) || tools.length === 0) return undefined;

	return (tools as ApiChatCompletionTool[]).map((tool) => ({
		description: tool.function?.description,
		input_schema: tool.function?.parameters ?? { properties: {}, type: 'object' },
		name: tool.function?.name
	}));
}

function buildChatRequest(
	body: Record<string, unknown>,
	_backend: Backend
): Record<string, unknown> {
	const messages = Array.isArray(body.messages) ? (body.messages as ApiChatMessageData[]) : [];
	const configuredMaxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : 0;
	const maxTokens = configuredMaxTokens > 0 ? configuredMaxTokens : DEFAULT_MAX_TOKENS;
	const system: string[] = [];
	const converted: { content: AnthropicBlock[]; role: 'assistant' | 'user' }[] = [];
	// the Messages API requires strictly alternating roles, so adjacent blocks
	// of the same role (tool results, split user turns) are merged
	const push = (role: 'assistant' | 'user', content: AnthropicBlock[]): void => {
		if (content.length === 0) return;

		const last = converted.at(-1);

		if (last?.role === role) {
			last.content.push(...content);
		} else {
			converted.push({ content, role });
		}
	};

	for (const message of messages) {
		if (message.role === MessageRole.SYSTEM) {
			const text = textOf(message.content);

			if (text) system.push(text);

			continue;
		}

		if (message.role === MessageRole.TOOL) {
			push('user', [
				{
					content: textOf(message.content),
					tool_use_id: message.tool_call_id,
					type: 'tool_result'
				}
			]);

			continue;
		}

		if (message.role === MessageRole.ASSISTANT) {
			push('assistant', assistantBlocks(message));

			continue;
		}

		push('user', userBlocks(message.content));
	}

	const request: Record<string, unknown> = {
		max_tokens: maxTokens,
		messages: converted,
		model: body.model
	};

	if (system.length > 0) request.system = system.join('\n\n');

	if (body.stream) request.stream = true;

	if (typeof body.temperature === 'number') request.temperature = body.temperature;

	if (typeof body.top_p === 'number') request.top_p = body.top_p;

	const tools = convertTools(body.tools);

	if (tools) request.tools = tools;

	const chatTemplateKwargs = body.chat_template_kwargs as Record<string, unknown> | undefined;
	const budgetTokens =
		typeof body.thinking_budget_tokens === 'number' ? body.thinking_budget_tokens : 0;

	// extended thinking needs a budget below max_tokens, and Anthropic rejects
	// temperature and top_p while thinking is enabled
	if (
		chatTemplateKwargs?.enable_thinking === true &&
		budgetTokens > 0 &&
		budgetTokens < maxTokens
	) {
		request.thinking = { budget_tokens: budgetTokens, type: 'enabled' };
		delete request.temperature;
		delete request.top_p;
	}

	return request;
}

/** Map an Anthropic usage object onto the compatible fields, defined keys only. */
function usageOf(raw: Record<string, unknown> | undefined): ApiChatCompletionUsage | undefined {
	if (!raw) return undefined;

	const usage: ApiChatCompletionUsage = {};

	if (typeof raw.input_tokens === 'number') usage.input_tokens = raw.input_tokens;

	if (typeof raw.output_tokens === 'number') usage.output_tokens = raw.output_tokens;

	return Object.keys(usage).length > 0 ? usage : undefined;
}

function createStreamReader(): ChatStreamReader {
	// Anthropic indexes content blocks across text and tool_use, while the
	// canonical tool call deltas are indexed among tool calls only
	const toolIndexes = new Map<number, number>();

	let toolCount = 0;

	const toolIndex = (blockIndex: number | undefined): number => {
		if (typeof blockIndex !== 'number') return toolCount++;

		const known = toolIndexes.get(blockIndex);

		if (known !== undefined) return known;

		const index = toolCount++;

		toolIndexes.set(blockIndex, index);

		return index;
	};

	return {
		readChunk(payload: unknown): ChatStreamEvent[] {
			if (!payload || typeof payload !== 'object') return [];

			const event = payload as AnthropicStreamEvent;
			const events: ChatStreamEvent[] = [];

			switch (event.type) {
				case 'message_start': {
					if (event.message?.id) events.push({ id: event.message.id, type: 'id' });

					if (event.message?.model) events.push({ model: event.message.model, type: 'model' });

					const usage = usageOf(event.message?.usage);

					if (usage) events.push({ type: 'usage', usage });

					break;
				}

				case 'content_block_start': {
					if (event.content_block?.type === 'tool_use') {
						events.push({
							deltas: [
								{
									function: { name: event.content_block.name },
									id: event.content_block.id,
									index: toolIndex(event.index),
									type: 'function'
								}
							],
							type: 'tool_calls'
						});
					}

					break;
				}

				case 'content_block_delta': {
					const delta = event.delta;

					if (delta?.type === 'text_delta' && delta.text) {
						events.push({ text: delta.text, type: 'text' });
					} else if (delta?.type === 'thinking_delta' && delta.thinking) {
						events.push({ text: delta.thinking, type: 'thinking' });
					} else if (delta?.type === 'input_json_delta' && delta.partial_json) {
						events.push({
							deltas: [
								{
									function: { arguments: delta.partial_json },
									index: toolIndex(event.index)
								}
							],
							type: 'tool_calls'
						});
					}

					break;
				}

				case 'message_delta': {
					const usage = usageOf(event.usage);

					if (usage) events.push({ type: 'usage', usage });

					break;
				}

				case 'message_stop':
					events.push({ type: 'done' });

					break;

				case 'error':
					events.push({
						message: event.error?.message ?? 'Anthropic stream error',
						type: 'error'
					});

					break;
			}

			return events;
		}
	};
}

export const anthropicAdapter: ChatProtocolAdapter = {
	authHeaders,
	buildChatRequest,
	createStreamReader
};
