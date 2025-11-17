import type {
	ApiChatCompletionRequest,
	ApiChatMessageData,
	ApiChatCompletionToolCall
} from '$lib/types/api';
import type { ChatMessagePromptProgress, ChatMessageTimings } from '$lib/types/chat';
import type { MCPToolCall } from '$lib/mcp';
import { MCPClient } from '$lib/mcp';
import { OpenAISseClient, type OpenAISseTurnResult } from './openai-sse-client';
import type { AgenticChatCompletionRequest, AgenticMessage, AgenticToolCallList } from './types';
import { toAgenticMessages } from './types';

export type AgenticOrchestratorCallbacks = {
	onChunk?: (chunk: string) => void;
	onReasoningChunk?: (chunk: string) => void;
	onToolCallChunk?: (serializedToolCalls: string) => void;
	onModel?: (model: string) => void;
	onFirstValidChunk?: () => void;
	onComplete?: () => void;
	onError?: (error: Error) => void;
};

export type AgenticRunParams = {
	initialMessages: ApiChatMessageData[];
	requestTemplate: ApiChatCompletionRequest;
	callbacks: AgenticOrchestratorCallbacks;
	abortSignal?: AbortSignal;
	onProcessingUpdate?: (timings?: ChatMessageTimings, progress?: ChatMessagePromptProgress) => void;
	maxTurns?: number;
	filterReasoningAfterFirstTurn?: boolean;
};

export type AgenticOrchestratorOptions = {
	mcpClient: MCPClient;
	llmClient: OpenAISseClient;
	maxTurns: number;
	maxToolPreviewLines: number;
};

export class AgenticOrchestrator {
	private readonly mcpClient: MCPClient;
	private readonly llmClient: OpenAISseClient;
	private readonly maxTurns: number;
	private readonly maxToolPreviewLines: number;

	constructor(options: AgenticOrchestratorOptions) {
		this.mcpClient = options.mcpClient;
		this.llmClient = options.llmClient;
		this.maxTurns = options.maxTurns;
		this.maxToolPreviewLines = options.maxToolPreviewLines;
	}

	async run(params: AgenticRunParams): Promise<void> {
		const baseMessages = toAgenticMessages(params.initialMessages);
		const sessionMessages: AgenticMessage[] = [...baseMessages];
		const tools = await this.mcpClient.getToolsDefinition();

		const requestWithoutMessages = { ...params.requestTemplate };
		delete (requestWithoutMessages as Partial<ApiChatCompletionRequest>).messages;
		const requestBase: AgenticChatCompletionRequest = {
			...(requestWithoutMessages as Omit<ApiChatCompletionRequest, 'messages'>),
			stream: true,
			messages: []
		};

		const maxTurns = params.maxTurns ?? this.maxTurns;

		// Accumulate tool_calls across all turns (not per-turn)
		const allToolCalls: ApiChatCompletionToolCall[] = [];

		for (let turn = 0; turn < maxTurns; turn++) {
			if (params.abortSignal?.aborted) {
				params.callbacks.onComplete?.();
				return;
			}

			const llmRequest: AgenticChatCompletionRequest = {
				...requestBase,
				messages: sessionMessages,
				tools: tools.length > 0 ? tools : undefined
			};

			const shouldFilterReasoningChunks = params.filterReasoningAfterFirstTurn === true && turn > 0;

			let turnResult: OpenAISseTurnResult;
			try {
				turnResult = await this.llmClient.stream(
					llmRequest,
					{
						onChunk: params.callbacks.onChunk,
						onReasoningChunk: shouldFilterReasoningChunks
							? undefined
							: params.callbacks.onReasoningChunk,
						onModel: params.callbacks.onModel,
						onFirstValidChunk: params.callbacks.onFirstValidChunk,
						onProcessingUpdate: (timings, progress) =>
							params.onProcessingUpdate?.(timings, progress)
					},
					params.abortSignal
				);
			} catch (error) {
				// Check if error is due to abort signal (stop button)
				if (params.abortSignal?.aborted) {
					params.callbacks.onComplete?.();
					return;
				}

				const normalizedError = error instanceof Error ? error : new Error('LLM stream error');
				params.callbacks.onError?.(normalizedError);
				const errorChunk = `\n\n\`\`\`\nUpstream LLM error:\n${normalizedError.message}\n\`\`\`\n`;
				params.callbacks.onChunk?.(errorChunk);
				params.callbacks.onComplete?.();
				return;
			}

			if (
				turnResult.toolCalls.length === 0 ||
				(turnResult.finishReason && turnResult.finishReason !== 'tool_calls')
			) {
				params.callbacks.onComplete?.();
				return;
			}

			const normalizedCalls = this.normalizeToolCalls(turnResult.toolCalls);
			if (normalizedCalls.length === 0) {
				params.callbacks.onComplete?.();
				return;
			}

			// Accumulate tool_calls from this turn
			for (const call of normalizedCalls) {
				allToolCalls.push({
					id: call.id,
					type: call.type,
					function: call.function ? { ...call.function } : undefined
				});
			}

			// Forward the complete accumulated list
			params.callbacks.onToolCallChunk?.(JSON.stringify(allToolCalls));

			sessionMessages.push({
				role: 'assistant',
				content: turnResult.content || undefined,
				tool_calls: normalizedCalls
			});

			for (const toolCall of normalizedCalls) {
				if (params.abortSignal?.aborted) {
					params.callbacks.onComplete?.();
					return;
				}

				const result = await this.executeTool(toolCall, params.abortSignal).catch(
					(error: Error) => {
						// Don't show error for AbortError
						if (error.name !== 'AbortError') {
							params.callbacks.onError?.(error);
						}
						return `Error: ${error.message}`;
					}
				);

				// Stop silently if aborted during tool execution
				if (params.abortSignal?.aborted) {
					params.callbacks.onComplete?.();
					return;
				}

				this.emitToolPreview(result, params.callbacks.onChunk);

				const contextValue = this.sanitizeToolContent(result);
				sessionMessages.push({
					role: 'tool',
					tool_call_id: toolCall.id,
					content: contextValue
				});
			}
		}

		params.callbacks.onChunk?.('\n\n```\nTurn limit reached\n```\n');
		params.callbacks.onComplete?.();
	}

	private normalizeToolCalls(toolCalls: ApiChatCompletionToolCall[]): AgenticToolCallList {
		if (!toolCalls) {
			return [];
		}

		return toolCalls.map((call, index) => ({
			id: call?.id ?? `tool_${index}`,
			type: (call?.type as 'function') ?? 'function',
			function: {
				name: call?.function?.name ?? '',
				arguments: call?.function?.arguments ?? ''
			}
		}));
	}

	private async executeTool(
		toolCall: AgenticToolCallList[number],
		abortSignal?: AbortSignal
	): Promise<string> {
		const mcpCall: MCPToolCall = {
			id: toolCall.id,
			function: {
				name: toolCall.function.name,
				arguments: toolCall.function.arguments
			}
		};

		const result = await this.mcpClient.execute(mcpCall, abortSignal);
		return result;
	}

	private emitToolPreview(result: string, emit?: (chunk: string) => void): void {
		if (!emit) return;
		const preview = this.createPreview(result);
		emit(preview);
	}

	private createPreview(result: string): string {
		if (this.isBase64Image(result)) {
			return `\n![tool-image](${result.trim()})\n`;
		}

		const lines = result.split('\n');
		const trimmedLines =
			lines.length > this.maxToolPreviewLines ? lines.slice(-this.maxToolPreviewLines) : lines;
		const preview = trimmedLines.join('\n');
		return `\n\`\`\`\n${preview}\n\`\`\`\n`;
	}

	private sanitizeToolContent(result: string): string {
		if (this.isBase64Image(result)) {
			return '[Image displayed to user]';
		}
		return result;
	}

	private isBase64Image(content: string): boolean {
		const trimmed = content.trim();
		if (!trimmed.startsWith('data:image/')) {
			return false;
		}

		const match = trimmed.match(/^data:image\/(png|jpe?g|gif|webp);base64,([A-Za-z0-9+/]+=*)$/);
		if (!match) {
			return false;
		}

		const base64Payload = match[2];
		return base64Payload.length > 0 && base64Payload.length % 4 === 0;
	}
}
