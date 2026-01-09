import type {
	ApiChatCompletionToolCall,
	ApiChatCompletionToolCallDelta,
	ApiChatCompletionStreamChunk
} from '$lib/types/api';
import type { ChatMessagePromptProgress, ChatMessageTimings } from '$lib/types/chat';
import { mergeToolCallDeltas, extractModelName } from '$lib/utils/chat-stream';
import type { AgenticChatCompletionRequest } from '$lib/types/agentic';

export type OpenAISseCallbacks = {
	onChunk?: (chunk: string) => void;
	onReasoningChunk?: (chunk: string) => void;
	onToolCallChunk?: (serializedToolCalls: string) => void;
	onModel?: (model: string) => void;
	onFirstValidChunk?: () => void;
	onProcessingUpdate?: (timings?: ChatMessageTimings, progress?: ChatMessagePromptProgress) => void;
};

export type OpenAISseTurnResult = {
	content: string;
	reasoningContent?: string;
	toolCalls: ApiChatCompletionToolCall[];
	finishReason?: string | null;
	timings?: ChatMessageTimings;
};

export type OpenAISseClientOptions = {
	url: string;
	buildHeaders?: () => Record<string, string>;
};

export class OpenAISseClient {
	constructor(private readonly options: OpenAISseClientOptions) {}

	async stream(
		request: AgenticChatCompletionRequest,
		callbacks: OpenAISseCallbacks = {},
		abortSignal?: AbortSignal
	): Promise<OpenAISseTurnResult> {
		const response = await fetch(this.options.url, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				...(this.options.buildHeaders?.() ?? {})
			},
			body: JSON.stringify(request),
			signal: abortSignal
		});

		if (!response.ok) {
			const errorText = await response.text();
			throw new Error(errorText || `LLM request failed (${response.status})`);
		}

		const reader = response.body?.getReader();
		if (!reader) {
			throw new Error('LLM response stream is not available');
		}

		return this.consumeStream(reader, callbacks, abortSignal);
	}

	private async consumeStream(
		reader: ReadableStreamDefaultReader<Uint8Array>,
		callbacks: OpenAISseCallbacks,
		abortSignal?: AbortSignal
	): Promise<OpenAISseTurnResult> {
		const decoder = new TextDecoder();
		let buffer = '';
		let aggregatedContent = '';
		let aggregatedReasoning = '';
		let aggregatedToolCalls: ApiChatCompletionToolCall[] = [];
		let hasOpenToolCallBatch = false;
		let toolCallIndexOffset = 0;
		let finishReason: string | null | undefined;
		let lastTimings: ChatMessageTimings | undefined;
		let modelEmitted = false;
		let firstValidChunkEmitted = false;

		const finalizeToolCallBatch = () => {
			if (!hasOpenToolCallBatch) return;
			toolCallIndexOffset = aggregatedToolCalls.length;
			hasOpenToolCallBatch = false;
		};

		const processToolCalls = (toolCalls?: ApiChatCompletionToolCallDelta[]) => {
			if (!toolCalls || toolCalls.length === 0) {
				return;
			}
			aggregatedToolCalls = mergeToolCallDeltas(
				aggregatedToolCalls,
				toolCalls,
				toolCallIndexOffset
			);
			if (aggregatedToolCalls.length === 0) {
				return;
			}
			hasOpenToolCallBatch = true;
		};

		try {
			while (true) {
				if (abortSignal?.aborted) {
					throw new DOMException('Aborted', 'AbortError');
				}

				const { done, value } = await reader.read();
				if (done) break;

				buffer += decoder.decode(value, { stream: true });
				const lines = buffer.split('\n');
				buffer = lines.pop() ?? '';

				for (const line of lines) {
					if (!line.startsWith('data: ')) {
						continue;
					}

					const payload = line.slice(6);
					if (payload === '[DONE]' || payload.trim().length === 0) {
						continue;
					}

					let chunk: ApiChatCompletionStreamChunk;
					try {
						chunk = JSON.parse(payload) as ApiChatCompletionStreamChunk;
					} catch (error) {
						console.error('[Agentic][SSE] Failed to parse chunk:', error);
						continue;
					}

					if (!firstValidChunkEmitted && chunk.object === 'chat.completion.chunk') {
						firstValidChunkEmitted = true;
						callbacks.onFirstValidChunk?.();
					}

					const choice = chunk.choices?.[0];
					const delta = choice?.delta;
					finishReason = choice?.finish_reason ?? finishReason;

					if (!modelEmitted) {
						const chunkModel = extractModelName(chunk);
						if (chunkModel) {
							modelEmitted = true;
							callbacks.onModel?.(chunkModel);
						}
					}

					if (chunk.timings || chunk.prompt_progress) {
						callbacks.onProcessingUpdate?.(chunk.timings, chunk.prompt_progress);
						if (chunk.timings) {
							lastTimings = chunk.timings;
						}
					}

					if (delta?.content) {
						finalizeToolCallBatch();
						aggregatedContent += delta.content;
						callbacks.onChunk?.(delta.content);
					}

					if (delta?.reasoning_content) {
						finalizeToolCallBatch();
						aggregatedReasoning += delta.reasoning_content;
						callbacks.onReasoningChunk?.(delta.reasoning_content);
					}

					processToolCalls(delta?.tool_calls);
				}
			}

			finalizeToolCallBatch();
		} catch (error) {
			if ((error as Error).name === 'AbortError') {
				throw error;
			}
			throw error instanceof Error ? error : new Error('LLM stream error');
		} finally {
			reader.releaseLock();
		}

		return {
			content: aggregatedContent,
			reasoningContent: aggregatedReasoning || undefined,
			toolCalls: aggregatedToolCalls,
			finishReason,
			timings: lastTimings
		};
	}
}
