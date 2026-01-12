/**
 * AgenticClient - Business Logic Facade for Agentic Loop Orchestration
 *
 * Coordinates the multi-turn agentic loop with MCP tools:
 * - LLM streaming with tool call detection
 * - Tool execution via MCPClient
 * - Session state management
 * - Turn limit enforcement
 *
 * **Architecture & Relationships:**
 * - **AgenticClient** (this class): Orchestrates multi-turn tool loop
 *   - Uses MCPClient for tool execution
 *   - Uses ChatService for LLM streaming
 *   - Updates agenticStore with reactive state
 *
 * - **MCPClient**: Tool execution facade
 * - **agenticStore**: Reactive state only ($state)
 *
 * **Key Features:**
 * - Multi-turn tool call orchestration
 * - Automatic routing of tool calls to appropriate MCP servers
 * - Raw LLM output streaming (UI formatting is separate concern)
 * - Lazy disconnect after flow completes
 */

import { mcpClient } from '$lib/clients';
import { ChatService } from '$lib/services';
import { config } from '$lib/stores/settings.svelte';
import { getAgenticConfig } from '$lib/utils/agentic';
import { toAgenticMessages } from '$lib/utils';
import type { AgenticMessage, AgenticToolCallList } from '$lib/types/agentic';
import type { ApiChatCompletionToolCall, ApiChatMessageData } from '$lib/types/api';
import type {
	ChatMessagePromptProgress,
	ChatMessageTimings,
	ChatMessageAgenticTimings,
	ChatMessageToolCallTiming,
	ChatMessageAgenticTurnStats
} from '$lib/types/chat';
import type { MCPToolCall } from '$lib/types/mcp';
import type { DatabaseMessage, DatabaseMessageExtra, McpServerOverride } from '$lib/types/database';

export interface AgenticFlowCallbacks {
	onChunk?: (chunk: string) => void;
	onReasoningChunk?: (chunk: string) => void;
	onToolCallChunk?: (serializedToolCalls: string) => void;
	onModel?: (model: string) => void;
	onComplete?: (
		content: string,
		reasoningContent?: string,
		timings?: ChatMessageTimings,
		toolCalls?: string
	) => void;
	onError?: (error: Error) => void;
	onTimings?: (timings?: ChatMessageTimings, promptProgress?: ChatMessagePromptProgress) => void;
}

export interface AgenticFlowOptions {
	stream?: boolean;
	model?: string;
	temperature?: number;
	max_tokens?: number;
	[key: string]: unknown;
}

export interface AgenticFlowParams {
	messages: (ApiChatMessageData | (DatabaseMessage & { extra?: DatabaseMessageExtra[] }))[];
	options?: AgenticFlowOptions;
	callbacks: AgenticFlowCallbacks;
	signal?: AbortSignal;
	/** Per-chat MCP server overrides */
	perChatOverrides?: McpServerOverride[];
}

export interface AgenticFlowResult {
	handled: boolean;
	error?: Error;
}

interface AgenticStoreStateCallbacks {
	setRunning: (running: boolean) => void;
	setCurrentTurn: (turn: number) => void;
	setTotalToolCalls: (count: number) => void;
	setLastError: (error: Error | null) => void;
	setStreamingToolCall: (tc: { name: string; arguments: string } | null) => void;
	clearStreamingToolCall: () => void;
}

export class AgenticClient {
	private storeCallbacks: AgenticStoreStateCallbacks | null = null;

	/**
	 *
	 *
	 * Store Integration
	 *
	 *
	 */

	/**
	 * Sets callbacks for store state updates.
	 * Called by agenticStore during initialization.
	 */
	setStoreCallbacks(callbacks: AgenticStoreStateCallbacks): void {
		this.storeCallbacks = callbacks;
	}

	private get store(): AgenticStoreStateCallbacks {
		if (!this.storeCallbacks) {
			throw new Error('AgenticClient: Store callbacks not initialized');
		}
		return this.storeCallbacks;
	}

	/**
	 *
	 *
	 * Agentic Flow
	 *
	 *
	 */

	/**
	 * Runs the agentic orchestration loop with MCP tools.
	 * Main entry point called by ChatClient when agentic mode is enabled.
	 *
	 * Coordinates: initial LLM request → tool call detection → tool execution → loop until done.
	 *
	 * @param params - Flow parameters including messages, options, callbacks, and signal
	 * @returns Result indicating if the flow handled the request
	 */
	async runAgenticFlow(params: AgenticFlowParams): Promise<AgenticFlowResult> {
		const { messages, options = {}, callbacks, signal, perChatOverrides } = params;
		const { onChunk, onReasoningChunk, onToolCallChunk, onModel, onComplete, onError, onTimings } =
			callbacks;

		// Get agentic configuration (considering per-chat MCP overrides)
		const agenticConfig = getAgenticConfig(config(), perChatOverrides);
		if (!agenticConfig.enabled) {
			return { handled: false };
		}

		// Ensure MCP is initialized with per-chat overrides
		const initialized = await mcpClient.ensureInitialized(perChatOverrides);
		if (!initialized) {
			console.log('[AgenticClient] MCP not initialized, falling back to standard chat');
			return { handled: false };
		}

		const tools = mcpClient.getToolDefinitionsForLLM();
		if (tools.length === 0) {
			console.log('[AgenticClient] No tools available, falling back to standard chat');
			return { handled: false };
		}

		console.log(`[AgenticClient] Starting agentic flow with ${tools.length} tools`);

		const normalizedMessages: ApiChatMessageData[] = messages
			.map((msg) => {
				if ('id' in msg && 'convId' in msg && 'timestamp' in msg) {
					// DatabaseMessage - use ChatService to convert
					return ChatService.convertDbMessageToApiChatMessageData(
						msg as DatabaseMessage & { extra?: DatabaseMessageExtra[] }
					);
				}
				return msg as ApiChatMessageData;
			})
			.filter((msg) => {
				if (msg.role === 'system') {
					const content = typeof msg.content === 'string' ? msg.content : '';
					return content.trim().length > 0;
				}
				return true;
			});

		this.store.setRunning(true);
		this.store.setCurrentTurn(0);
		this.store.setTotalToolCalls(0);
		this.store.setLastError(null);

		try {
			await this.executeAgenticLoop({
				messages: normalizedMessages,
				options,
				tools,
				agenticConfig,
				callbacks: {
					onChunk,
					onReasoningChunk,
					onToolCallChunk,
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
			this.store.setLastError(normalizedError);
			onError?.(normalizedError);
			return { handled: true, error: normalizedError };
		} finally {
			this.store.setRunning(false);
			// Lazy Disconnect: Close MCP connections after agentic flow completes
			// This prevents continuous keepalive/heartbeat polling when tools are not in use
			await mcpClient.shutdown().catch((err) => {
				console.warn('[AgenticClient] Failed to shutdown MCP after flow:', err);
			});

			console.log('[AgenticClient] MCP connections closed (lazy disconnect)');
		}
	}

	private async executeAgenticLoop(params: {
		messages: ApiChatMessageData[];
		options: AgenticFlowOptions;
		tools: ReturnType<typeof mcpClient.getToolDefinitionsForLLM>;
		agenticConfig: ReturnType<typeof getAgenticConfig>;
		callbacks: AgenticFlowCallbacks;
		signal?: AbortSignal;
	}): Promise<void> {
		const { messages, options, tools, agenticConfig, callbacks, signal } = params;
		const { onChunk, onReasoningChunk, onToolCallChunk, onModel, onComplete, onTimings } =
			callbacks;

		const sessionMessages: AgenticMessage[] = toAgenticMessages(messages);
		const allToolCalls: ApiChatCompletionToolCall[] = [];
		let capturedTimings: ChatMessageTimings | undefined;

		const agenticTimings: ChatMessageAgenticTimings = {
			turns: 0,
			toolCallsCount: 0,
			toolsMs: 0,
			toolCalls: [],
			perTurn: [],
			llm: {
				predicted_n: 0,
				predicted_ms: 0,
				prompt_n: 0,
				prompt_ms: 0
			}
		};

		const maxTurns = agenticConfig.maxTurns;
		const maxToolPreviewLines = agenticConfig.maxToolPreviewLines;

		for (let turn = 0; turn < maxTurns; turn++) {
			this.store.setCurrentTurn(turn + 1);
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

			// Filter reasoning content after first turn if configured
			const shouldFilterReasoning = agenticConfig.filterReasoningAfterFirstTurn && turn > 0;

			let turnContent = '';
			let turnToolCalls: ApiChatCompletionToolCall[] = [];
			let lastStreamingToolCallName = '';
			let lastStreamingToolCallArgsLength = 0;

			let turnTimings: ChatMessageTimings | undefined;

			const turnStats: ChatMessageAgenticTurnStats = {
				turn: turn + 1,
				llm: {
					predicted_n: 0,
					predicted_ms: 0,
					prompt_n: 0,
					prompt_ms: 0
				},
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
						onReasoningChunk: shouldFilterReasoning ? undefined : onReasoningChunk,
						onToolCallChunk: (serialized: string) => {
							try {
								turnToolCalls = JSON.parse(serialized) as ApiChatCompletionToolCall[];
								// Update store with streaming tool call state for UI visualization
								// Only update when values actually change to avoid memory pressure
								if (turnToolCalls.length > 0 && turnToolCalls[0]?.function) {
									const name = turnToolCalls[0].function.name || '';
									const args = turnToolCalls[0].function.arguments || '';
									// Only update if name changed or args grew significantly (every 100 chars)
									const argsLengthBucket = Math.floor(args.length / 100);
									if (
										name !== lastStreamingToolCallName ||
										argsLengthBucket !== lastStreamingToolCallArgsLength
									) {
										lastStreamingToolCallName = name;
										lastStreamingToolCallArgsLength = argsLengthBucket;
										this.store.setStreamingToolCall({ name, arguments: args });
									}
								}
							} catch {
								// Ignore parse errors during streaming
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
							// Completion handled after sendMessage resolves
						},
						onError: (error: Error) => {
							throw error;
						}
					},
					undefined,
					signal
				);

				this.store.clearStreamingToolCall();

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
			this.store.setTotalToolCalls(allToolCalls.length);
			onToolCallChunk?.(JSON.stringify(allToolCalls));

			sessionMessages.push({
				role: 'assistant',
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

				// Start timing BEFORE emitToolCallStart to capture full perceived execution time
				const toolStartTime = performance.now();
				this.emitToolCallStart(toolCall, onChunk);

				const mcpCall: MCPToolCall = {
					id: toolCall.id,
					function: {
						name: toolCall.function.name,
						arguments: toolCall.function.arguments
					}
				};

				let result: string;
				let toolSuccess = true;

				try {
					const executionResult = await mcpClient.executeTool(mcpCall, signal);
					result = executionResult.content;
				} catch (error) {
					if (error instanceof Error && error.name === 'AbortError') {
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

				this.emitToolCallResult(result, maxToolPreviewLines, onChunk);

				// Add tool result to session (sanitize base64 images for context)
				const contextValue = this.isBase64Image(result) ? '[Image displayed to user]' : result;
				sessionMessages.push({
					role: 'tool',
					tool_call_id: toolCall.id,
					content: contextValue
				});
			}

			// Save per-turn stats (only if there were tool calls in this turn)
			if (turnStats.toolCalls.length > 0) {
				agenticTimings.perTurn!.push(turnStats);
			}
		}

		onChunk?.('\n\n```\nTurn limit reached\n```\n');
		onComplete?.('', undefined, this.buildFinalTimings(capturedTimings, agenticTimings), undefined);
	}

	/**
	 *
	 *
	 * Timing & Statistics
	 *
	 *
	 */

	/**
	 * Builds final timings object with agentic stats.
	 * Single-turn flows return original timings; multi-turn includes aggregated stats.
	 */
	private buildFinalTimings(
		capturedTimings: ChatMessageTimings | undefined,
		agenticTimings: ChatMessageAgenticTimings
	): ChatMessageTimings | undefined {
		// If no tool calls were made, this was effectively a single-turn flow
		// Return the original timings without agentic data
		if (agenticTimings.toolCallsCount === 0) {
			return capturedTimings;
		}

		const finalTimings: ChatMessageTimings = {
			// Use the last turn's values as the "current" values for backward compatibility
			predicted_n: capturedTimings?.predicted_n,
			predicted_ms: capturedTimings?.predicted_ms,
			prompt_n: capturedTimings?.prompt_n,
			prompt_ms: capturedTimings?.prompt_ms,
			cache_n: capturedTimings?.cache_n,
			agentic: agenticTimings
		};

		return finalTimings;
	}

	/**
	 *
	 *
	 * Tool Call Processing
	 *
	 *
	 */

	private normalizeToolCalls(toolCalls: ApiChatCompletionToolCall[]): AgenticToolCallList {
		if (!toolCalls) return [];
		return toolCalls.map((call, index) => ({
			id: call?.id ?? `tool_${index}`,
			type: (call?.type as 'function') ?? 'function',
			function: {
				name: call?.function?.name ?? '',
				arguments: call?.function?.arguments ?? ''
			}
		}));
	}

	/**
	 * Emit tool call start marker (shows "pending" state in UI).
	 */
	private emitToolCallStart(
		toolCall: AgenticToolCallList[number],
		emit?: (chunk: string) => void
	): void {
		if (!emit) return;

		const toolName = toolCall.function.name;
		const toolArgs = toolCall.function.arguments;
		// Base64 encode args to avoid conflicts with markdown/HTML parsing
		const toolArgsBase64 = btoa(unescape(encodeURIComponent(toolArgs)));

		let output = `\n\n<<<AGENTIC_TOOL_CALL_START>>>`;
		output += `\n<<<TOOL_NAME:${toolName}>>>`;
		output += `\n<<<TOOL_ARGS_BASE64:${toolArgsBase64}>>>`;
		emit(output);
	}

	/**
	 * Emit tool call result and end marker.
	 */
	private emitToolCallResult(
		result: string,
		maxLines: number,
		emit?: (chunk: string) => void
	): void {
		if (!emit) return;

		let output = '';
		if (this.isBase64Image(result)) {
			output += `\n![tool-result](${result.trim()})`;
		} else {
			// Don't wrap in code fences - result may already be markdown with its own code blocks
			const lines = result.split('\n');
			const trimmedLines = lines.length > maxLines ? lines.slice(-maxLines) : lines;
			output += `\n${trimmedLines.join('\n')}`;
		}

		output += `\n<<<AGENTIC_TOOL_CALL_END>>>\n`;
		emit(output);
	}

	/**
	 *
	 *
	 * Utilities
	 *
	 *
	 */

	private isBase64Image(content: string): boolean {
		const trimmed = content.trim();
		if (!trimmed.startsWith('data:image/')) return false;

		const match = trimmed.match(/^data:image\/(png|jpe?g|gif|webp);base64,([A-Za-z0-9+/]+=*)$/);
		if (!match) return false;

		const base64Payload = match[2];
		return base64Payload.length > 0 && base64Payload.length % 4 === 0;
	}

	clearError(): void {
		this.store.setLastError(null);
	}
}

export const agenticClient = new AgenticClient();
