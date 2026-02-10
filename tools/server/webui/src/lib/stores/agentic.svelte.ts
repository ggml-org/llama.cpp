/**
 * agenticStore - Reactive State Store for Agentic Loop Orchestration
 *
 * Manages multi-turn agentic loop with MCP tools:
 * - LLM streaming with tool call detection
 * - Tool execution via mcpStore
 * - Session state management
 * - Turn limit enforcement
 *
 * **Architecture & Relationships:**
 * - **ChatService**: Stateless API layer (sendMessage, streaming)
 * - **mcpStore**: MCP connection management and tool execution
 * - **agenticStore** (this): Reactive state + business logic
 *
 * @see ChatService in services/chat.service.ts for API operations
 * @see mcpStore in stores/mcp.svelte.ts for MCP operations
 */

import { SvelteMap } from 'svelte/reactivity';
import { ChatService } from '$lib/services';
import { config } from '$lib/stores/settings.svelte';
import { mcpStore } from '$lib/stores/mcp.svelte';
import { modelsStore } from '$lib/stores/models.svelte';
import { isAbortError } from '$lib/utils';
import { DEFAULT_AGENTIC_CONFIG, AGENTIC_TAGS } from '$lib/constants/agentic';
import { IMAGE_MIME_TO_EXTENSION } from '$lib/constants/mcp-resource';
import { AttachmentType, ContentPartType, MessageRole } from '$lib/enums';
import type {
	AgenticFlowParams,
	AgenticFlowResult,
	AgenticSession,
	AgenticConfig,
	SettingsConfigType,
	McpServerOverride,
	MCPToolCall
} from '$lib/types';
import type {
	AgenticMessage,
	AgenticToolCallList,
	AgenticFlowCallbacks,
	AgenticFlowOptions
} from '$lib/types/agentic';
import type {
	ApiChatCompletionToolCall,
	ApiChatMessageData,
	ApiChatMessageContentPart
} from '$lib/types/api';
import type {
	ChatMessagePromptProgress,
	ChatMessageTimings,
	ChatMessageAgenticTimings,
	ChatMessageToolCallTiming,
	ChatMessageAgenticTurnStats
} from '$lib/types/chat';
import type {
	DatabaseMessage,
	DatabaseMessageExtra,
	DatabaseMessageExtraImageFile
} from '$lib/types/database';

function createDefaultSession(): AgenticSession {
	return {
		isRunning: false,
		currentTurn: 0,
		totalToolCalls: 0,
		lastError: null,
		streamingToolCall: null
	};
}

function toAgenticMessages(messages: ApiChatMessageData[]): AgenticMessage[] {
	return messages.map((message) => {
		if (
			message.role === MessageRole.ASSISTANT &&
			message.tool_calls &&
			message.tool_calls.length > 0
		) {
			return {
				role: MessageRole.ASSISTANT,
				content: message.content,
				tool_calls: message.tool_calls.map((call, index) => ({
					id: call.id ?? `call_${index}`,
					type: (call.type as 'function') ?? 'function',
					function: { name: call.function?.name ?? '', arguments: call.function?.arguments ?? '' }
				}))
			} satisfies AgenticMessage;
		}
		if (message.role === MessageRole.TOOL && message.tool_call_id) {
			return {
				role: MessageRole.TOOL,
				tool_call_id: message.tool_call_id,
				content: typeof message.content === 'string' ? message.content : ''
			} satisfies AgenticMessage;
		}
		return {
			role: message.role as MessageRole.SYSTEM | MessageRole.USER,
			content: message.content
		} satisfies AgenticMessage;
	});
}

class AgenticStore {
	private _sessions = $state<Map<string, AgenticSession>>(new Map());

	get isReady(): boolean {
		return true;
	}
	get isAnyRunning(): boolean {
		for (const session of this._sessions.values()) {
			if (session.isRunning) return true;
		}
		return false;
	}

	getSession(conversationId: string): AgenticSession {
		let session = this._sessions.get(conversationId);
		if (!session) {
			session = createDefaultSession();
			this._sessions.set(conversationId, session);
		}
		return session;
	}

	private updateSession(conversationId: string, update: Partial<AgenticSession>): void {
		const session = this.getSession(conversationId);
		this._sessions.set(conversationId, { ...session, ...update });
	}

	clearSession(conversationId: string): void {
		this._sessions.delete(conversationId);
	}

	getActiveSessions(): Array<{ conversationId: string; session: AgenticSession }> {
		const active: Array<{ conversationId: string; session: AgenticSession }> = [];
		for (const [conversationId, session] of this._sessions.entries()) {
			if (session.isRunning) active.push({ conversationId, session });
		}
		return active;
	}

	isRunning(conversationId: string): boolean {
		return this.getSession(conversationId).isRunning;
	}

	currentTurn(conversationId: string): number {
		return this.getSession(conversationId).currentTurn;
	}

	totalToolCalls(conversationId: string): number {
		return this.getSession(conversationId).totalToolCalls;
	}

	lastError(conversationId: string): Error | null {
		return this.getSession(conversationId).lastError;
	}

	streamingToolCall(conversationId: string): { name: string; arguments: string } | null {
		return this.getSession(conversationId).streamingToolCall;
	}

	clearError(conversationId: string): void {
		this.updateSession(conversationId, { lastError: null });
	}

	getConfig(settings: SettingsConfigType, perChatOverrides?: McpServerOverride[]): AgenticConfig {
		const maxTurns = Number(settings.agenticMaxTurns) || DEFAULT_AGENTIC_CONFIG.maxTurns;
		const maxToolPreviewLines =
			Number(settings.agenticMaxToolPreviewLines) || DEFAULT_AGENTIC_CONFIG.maxToolPreviewLines;
		return {
			enabled: mcpStore.hasEnabledServers(perChatOverrides) && DEFAULT_AGENTIC_CONFIG.enabled,
			maxTurns,
			maxToolPreviewLines
		};
	}

	async runAgenticFlow(params: AgenticFlowParams): Promise<AgenticFlowResult> {
		const { conversationId, messages, options = {}, callbacks, signal, perChatOverrides } = params;
		const {
			onChunk,
			onReasoningChunk,
			onToolCallChunk,
			onAttachments,
			onModel,
			onComplete,
			onError,
			onTimings
		} = callbacks;

		const agenticConfig = this.getConfig(config(), perChatOverrides);
		if (!agenticConfig.enabled) return { handled: false };

		const initialized = await mcpStore.ensureInitialized(perChatOverrides);
		if (!initialized) {
			console.log('[AgenticStore] MCP not initialized, falling back to standard chat');
			return { handled: false };
		}

		const tools = mcpStore.getToolDefinitionsForLLM();
		if (tools.length === 0) {
			console.log('[AgenticStore] No tools available, falling back to standard chat');
			return { handled: false };
		}

		console.log(`[AgenticStore] Starting agentic flow with ${tools.length} tools`);

		const normalizedMessages: ApiChatMessageData[] = messages
			.map((msg) => {
				if ('id' in msg && 'convId' in msg && 'timestamp' in msg)
					return ChatService.convertDbMessageToApiChatMessageData(
						msg as DatabaseMessage & { extra?: DatabaseMessageExtra[] }
					);
				return msg as ApiChatMessageData;
			})
			.filter((msg) => {
				if (msg.role === MessageRole.SYSTEM) {
					const content = typeof msg.content === 'string' ? msg.content : '';
					return content.trim().length > 0;
				}
				return true;
			});

		this.updateSession(conversationId, {
			isRunning: true,
			currentTurn: 0,
			totalToolCalls: 0,
			lastError: null
		});
		mcpStore.acquireConnection();

		try {
			await this.executeAgenticLoop({
				conversationId,
				messages: normalizedMessages,
				options,
				tools,
				agenticConfig,
				callbacks: {
					onChunk,
					onReasoningChunk,
					onToolCallChunk,
					onAttachments,
					onModel,
					onComplete,
					onError,
					onTimings
				},
				signal
			});
			return { handled: true };
		} catch (error) {
			const normalizedError = error instanceof Error ? error : new Error(String(error));
			this.updateSession(conversationId, { lastError: normalizedError });
			onError?.(normalizedError);
			return { handled: true, error: normalizedError };
		} finally {
			this.updateSession(conversationId, { isRunning: false });
			await mcpStore
				.releaseConnection()
				.catch((err: unknown) =>
					console.warn('[AgenticStore] Failed to release MCP connection:', err)
				);
		}
	}

	private async executeAgenticLoop(params: {
		conversationId: string;
		messages: ApiChatMessageData[];
		options: AgenticFlowOptions;
		tools: ReturnType<typeof mcpStore.getToolDefinitionsForLLM>;
		agenticConfig: AgenticConfig;
		callbacks: AgenticFlowCallbacks;
		signal?: AbortSignal;
	}): Promise<void> {
		const { conversationId, messages, options, tools, agenticConfig, callbacks, signal } = params;
		const {
			onChunk,
			onReasoningChunk,
			onToolCallChunk,
			onAttachments,
			onModel,
			onComplete,
			onTimings
		} = callbacks;

		const sessionMessages: AgenticMessage[] = toAgenticMessages(messages);
		const allToolCalls: ApiChatCompletionToolCall[] = [];
		let capturedTimings: ChatMessageTimings | undefined;

		const agenticTimings: ChatMessageAgenticTimings = {
			turns: 0,
			toolCallsCount: 0,
			toolsMs: 0,
			toolCalls: [],
			perTurn: [],
			llm: { predicted_n: 0, predicted_ms: 0, prompt_n: 0, prompt_ms: 0 }
		};
		const maxTurns = agenticConfig.maxTurns;
		const maxToolPreviewLines = agenticConfig.maxToolPreviewLines;

		for (let turn = 0; turn < maxTurns; turn++) {
			this.updateSession(conversationId, { currentTurn: turn + 1 });
			agenticTimings.turns = turn + 1;

			if (signal?.aborted) {
				onComplete?.(
					'',
					undefined,
					this.buildFinalTimings(capturedTimings, agenticTimings),
					undefined
				);
				return;
			}

			let turnContent = '';
			let turnToolCalls: ApiChatCompletionToolCall[] = [];
			let lastStreamingToolCallName = '';
			let lastStreamingToolCallArgsLength = 0;
			const emittedToolCallStates = new SvelteMap<
				number,
				{ emittedOnce: boolean; lastArgs: string }
			>();
			let turnTimings: ChatMessageTimings | undefined;

			const turnStats: ChatMessageAgenticTurnStats = {
				turn: turn + 1,
				llm: { predicted_n: 0, predicted_ms: 0, prompt_n: 0, prompt_ms: 0 },
				toolCalls: [],
				toolsMs: 0
			};

			try {
				await ChatService.sendMessage(
					sessionMessages as ApiChatMessageData[],
					{
						...options,
						stream: true,
						tools: tools.length > 0 ? tools : undefined,
						onChunk: (chunk: string) => {
							turnContent += chunk;
							onChunk?.(chunk);
						},
						onReasoningChunk,
						onToolCallChunk: (serialized: string) => {
							try {
								turnToolCalls = JSON.parse(serialized) as ApiChatCompletionToolCall[];
								for (let i = 0; i < turnToolCalls.length; i++) {
									const toolCall = turnToolCalls[i];
									const toolName = toolCall.function?.name ?? '';
									const toolArgs = toolCall.function?.arguments ?? '';
									const state = emittedToolCallStates.get(i) || {
										emittedOnce: false,
										lastArgs: ''
									};
									if (!state.emittedOnce) {
										const output = `\n\n${AGENTIC_TAGS.TOOL_CALL_START}\n${AGENTIC_TAGS.TOOL_NAME_PREFIX}${toolName}${AGENTIC_TAGS.TAG_SUFFIX}\n${AGENTIC_TAGS.TOOL_ARGS_START}\n${toolArgs}`;
										onChunk?.(output);
										state.emittedOnce = true;
										state.lastArgs = toolArgs;
										emittedToolCallStates.set(i, state);
									} else if (toolArgs.length > state.lastArgs.length) {
										onChunk?.(toolArgs.slice(state.lastArgs.length));
										state.lastArgs = toolArgs;
										emittedToolCallStates.set(i, state);
									}
								}
								if (turnToolCalls.length > 0 && turnToolCalls[0]?.function) {
									const name = turnToolCalls[0].function.name || '';
									const args = turnToolCalls[0].function.arguments || '';
									const argsLengthBucket = Math.floor(args.length / 100);
									if (
										name !== lastStreamingToolCallName ||
										argsLengthBucket !== lastStreamingToolCallArgsLength
									) {
										lastStreamingToolCallName = name;
										lastStreamingToolCallArgsLength = argsLengthBucket;
										this.updateSession(conversationId, {
											streamingToolCall: { name, arguments: args }
										});
									}
								}
							} catch {
								/* Ignore parse errors during streaming */
							}
						},
						onModel,
						onTimings: (timings?: ChatMessageTimings, progress?: ChatMessagePromptProgress) => {
							onTimings?.(timings, progress);
							if (timings) {
								capturedTimings = timings;
								turnTimings = timings;
							}
						},
						onComplete: () => {
							/* Completion handled after sendMessage resolves */
						},
						onError: (error: Error) => {
							throw error;
						}
					},
					undefined,
					signal
				);

				this.updateSession(conversationId, { streamingToolCall: null });

				if (turnTimings) {
					agenticTimings.llm.predicted_n += turnTimings.predicted_n || 0;
					agenticTimings.llm.predicted_ms += turnTimings.predicted_ms || 0;
					agenticTimings.llm.prompt_n += turnTimings.prompt_n || 0;
					agenticTimings.llm.prompt_ms += turnTimings.prompt_ms || 0;
					turnStats.llm.predicted_n = turnTimings.predicted_n || 0;
					turnStats.llm.predicted_ms = turnTimings.predicted_ms || 0;
					turnStats.llm.prompt_n = turnTimings.prompt_n || 0;
					turnStats.llm.prompt_ms = turnTimings.prompt_ms || 0;
				}
			} catch (error) {
				if (signal?.aborted) {
					onComplete?.(
						'',
						undefined,
						this.buildFinalTimings(capturedTimings, agenticTimings),
						undefined
					);

					return;
				}
				const normalizedError = error instanceof Error ? error : new Error('LLM stream error');
				onChunk?.(`\n\n\`\`\`\nUpstream LLM error:\n${normalizedError.message}\n\`\`\`\n`);
				onComplete?.(
					'',
					undefined,
					this.buildFinalTimings(capturedTimings, agenticTimings),
					undefined
				);

				throw normalizedError;
			}

			if (turnToolCalls.length === 0) {
				agenticTimings.perTurn!.push(turnStats);

				onComplete?.(
					'',
					undefined,
					this.buildFinalTimings(capturedTimings, agenticTimings),
					undefined
				);

				return;
			}

			const normalizedCalls = this.normalizeToolCalls(turnToolCalls);
			if (normalizedCalls.length === 0) {
				onComplete?.(
					'',
					undefined,
					this.buildFinalTimings(capturedTimings, agenticTimings),
					undefined
				);
				return;
			}

			for (const call of normalizedCalls) {
				allToolCalls.push({
					id: call.id,
					type: call.type,
					function: call.function ? { ...call.function } : undefined
				});
			}

			this.updateSession(conversationId, { totalToolCalls: allToolCalls.length });
			onToolCallChunk?.(JSON.stringify(allToolCalls));

			sessionMessages.push({
				role: MessageRole.ASSISTANT,
				content: turnContent || undefined,
				tool_calls: normalizedCalls
			});

			for (const toolCall of normalizedCalls) {
				if (signal?.aborted) {
					onComplete?.(
						'',
						undefined,
						this.buildFinalTimings(capturedTimings, agenticTimings),
						undefined
					);

					return;
				}

				const toolStartTime = performance.now();
				const mcpCall: MCPToolCall = {
					id: toolCall.id,
					function: { name: toolCall.function.name, arguments: toolCall.function.arguments }
				};

				let result: string;
				let toolSuccess = true;

				try {
					const executionResult = await mcpStore.executeTool(mcpCall, signal);
					result = executionResult.content;
				} catch (error) {
					if (isAbortError(error)) {
						onComplete?.(
							'',
							undefined,
							this.buildFinalTimings(capturedTimings, agenticTimings),
							undefined
						);

						return;
					}
					result = `Error: ${error instanceof Error ? error.message : String(error)}`;
					toolSuccess = false;
				}

				const toolDurationMs = performance.now() - toolStartTime;
				const toolTiming: ChatMessageToolCallTiming = {
					name: toolCall.function.name,
					duration_ms: Math.round(toolDurationMs),
					success: toolSuccess
				};

				agenticTimings.toolCalls!.push(toolTiming);
				agenticTimings.toolCallsCount++;
				agenticTimings.toolsMs += Math.round(toolDurationMs);
				turnStats.toolCalls.push(toolTiming);
				turnStats.toolsMs += Math.round(toolDurationMs);

				if (signal?.aborted) {
					onComplete?.(
						'',
						undefined,
						this.buildFinalTimings(capturedTimings, agenticTimings),
						undefined
					);

					return;
				}

				const { cleanedResult, attachments } = this.extractBase64Attachments(result);
				if (attachments.length > 0) onAttachments?.(attachments);

				this.emitToolCallResult(cleanedResult, maxToolPreviewLines, onChunk);

				const contentParts: ApiChatMessageContentPart[] = [
					{ type: ContentPartType.TEXT, text: cleanedResult }
				];
				for (const attachment of attachments) {
					if (attachment.type === AttachmentType.IMAGE) {
						if (modelsStore.modelSupportsVision(options.model ?? '')) {
							contentParts.push({
								type: ContentPartType.IMAGE_URL,
								image_url: { url: (attachment as DatabaseMessageExtraImageFile).base64Url }
							});
						} else {
							console.info(
								`[AgenticStore] Skipping image attachment (model "${options.model}" does not support vision)`
							);
						}
					}
				}

				sessionMessages.push({
					role: MessageRole.TOOL,
					tool_call_id: toolCall.id,
					content: contentParts.length === 1 ? cleanedResult : contentParts
				});
			}

			if (turnStats.toolCalls.length > 0) agenticTimings.perTurn!.push(turnStats);
		}

		onChunk?.('\n\n```\nTurn limit reached\n```\n');
		onComplete?.('', undefined, this.buildFinalTimings(capturedTimings, agenticTimings), undefined);
	}

	private buildFinalTimings(
		capturedTimings: ChatMessageTimings | undefined,
		agenticTimings: ChatMessageAgenticTimings
	): ChatMessageTimings | undefined {
		if (agenticTimings.toolCallsCount === 0) return capturedTimings;
		return {
			predicted_n: capturedTimings?.predicted_n,
			predicted_ms: capturedTimings?.predicted_ms,
			prompt_n: capturedTimings?.prompt_n,
			prompt_ms: capturedTimings?.prompt_ms,
			cache_n: capturedTimings?.cache_n,
			agentic: agenticTimings
		};
	}

	private normalizeToolCalls(toolCalls: ApiChatCompletionToolCall[]): AgenticToolCallList {
		if (!toolCalls) return [];
		return toolCalls.map((call, index) => ({
			id: call?.id ?? `tool_${index}`,
			type: (call?.type as 'function') ?? 'function',
			function: { name: call?.function?.name ?? '', arguments: call?.function?.arguments ?? '' }
		}));
	}

	private emitToolCallResult(
		result: string,
		maxLines: number,
		emit?: (chunk: string) => void
	): void {
		if (!emit) {
			return;
		}

		let output = `\n${AGENTIC_TAGS.TOOL_ARGS_END}`;
		const lines = result.split('\n');
		const trimmedLines = lines.length > maxLines ? lines.slice(-maxLines) : lines;

		output += `\n${trimmedLines.join('\n')}\n${AGENTIC_TAGS.TOOL_CALL_END}\n`;
		emit(output);
	}

	private extractBase64Attachments(result: string): {
		cleanedResult: string;
		attachments: DatabaseMessageExtra[];
	} {
		if (!result.trim()) {
			return { cleanedResult: result, attachments: [] };
		}

		const lines = result.split('\n');
		const attachments: DatabaseMessageExtra[] = [];
		let attachmentIndex = 0;

		const cleanedLines = lines.map((line) => {
			const trimmedLine = line.trim();

			const match = trimmedLine.match(/^data:([^;]+);base64,([A-Za-z0-9+/]+=*)$/);
			if (!match) {
				return line;
			}

			const mimeType = match[1].toLowerCase();
			const base64Data = match[2];

			if (!base64Data) {
				return line;
			}

			attachmentIndex += 1;
			const name = this.buildAttachmentName(mimeType, attachmentIndex);

			if (mimeType.startsWith('image/')) {
				attachments.push({ type: AttachmentType.IMAGE, name, base64Url: trimmedLine });

				return `[Attachment saved: ${name}]`;
			}

			return line;
		});

		return { cleanedResult: cleanedLines.join('\n'), attachments };
	}

	private buildAttachmentName(mimeType: string, index: number): string {
		const extension = IMAGE_MIME_TO_EXTENSION[mimeType] ?? 'img';
		return `mcp-attachment-${Date.now()}-${index}.${extension}`;
	}
}

export const agenticStore = new AgenticStore();

export function agenticIsRunning(conversationId: string) {
	return agenticStore.isRunning(conversationId);
}

export function agenticCurrentTurn(conversationId: string) {
	return agenticStore.currentTurn(conversationId);
}

export function agenticTotalToolCalls(conversationId: string) {
	return agenticStore.totalToolCalls(conversationId);
}

export function agenticLastError(conversationId: string) {
	return agenticStore.lastError(conversationId);
}

export function agenticStreamingToolCall(conversationId: string) {
	return agenticStore.streamingToolCall(conversationId);
}

export function agenticIsAnyRunning() {
	return agenticStore.isAnyRunning;
}
