/**
 * agenticStore - Orchestration of the agentic loop with MCP tools
 *
 * This store is responsible for:
 * - Managing the agentic loop lifecycle
 * - Coordinating between LLM and MCP tool execution
 * - Tracking session state (messages, turns, tool calls)
 * - Emitting streaming content and tool results
 *
 * **Architecture & Relationships:**
 * - **mcpStore**: Provides MCP host manager for tool execution
 * - **chatStore**: Triggers agentic flow and receives streaming updates
 * - **OpenAISseClient**: LLM communication for streaming responses
 * - **settingsStore**: Provides agentic configuration (maxTurns, etc.)
 *
 * **Key Features:**
 * - Stateful session management (unlike stateless ChatService)
 * - Multi-turn tool call orchestration
 * - Automatic routing of tool calls to appropriate MCP servers
 * - Raw LLM output streaming (UI formatting is separate concern)
 */

import { mcpStore } from '$lib/stores/mcp.svelte';
import { OpenAISseClient, type OpenAISseTurnResult } from '$lib/agentic/openai-sse-client';
import {
	toAgenticMessages,
	type AgenticMessage,
	type AgenticChatCompletionRequest,
	type AgenticToolCallList
} from '$lib/agentic/types';
import type { ApiChatCompletionToolCall, ApiChatMessageData } from '$lib/types/api';
import type { ChatMessagePromptProgress, ChatMessageTimings } from '$lib/types/chat';
import type { MCPToolCall } from '$lib/types/mcp';
import type { DatabaseMessage, DatabaseMessageExtra, McpServerOverride } from '$lib/types/database';
import { getAgenticConfig } from '$lib/config/agentic';
import { config } from '$lib/stores/settings.svelte';
import { getAuthHeaders } from '$lib/utils';
import { ChatService } from '$lib/services';

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// Agentic Store
// ─────────────────────────────────────────────────────────────────────────────

class AgenticStore {
	// ─────────────────────────────────────────────────────────────────────────────
	// State
	// ─────────────────────────────────────────────────────────────────────────────

	private _isRunning = $state(false);
	private _currentTurn = $state(0);
	private _totalToolCalls = $state(0);
	private _lastError = $state<Error | null>(null);

	// ─────────────────────────────────────────────────────────────────────────────
	// Getters
	// ─────────────────────────────────────────────────────────────────────────────

	get isRunning(): boolean {
		return this._isRunning;
	}

	get currentTurn(): number {
		return this._currentTurn;
	}

	get totalToolCalls(): number {
		return this._totalToolCalls;
	}

	get lastError(): Error | null {
		return this._lastError;
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Main Agentic Flow
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Run the agentic orchestration loop with MCP tools.
	 *
	 * This is the main entry point called by chatStore when agentic mode is enabled.
	 * It coordinates:
	 * 1. Initial LLM request with available tools
	 * 2. Tool call detection and execution via MCP
	 * 3. Multi-turn loop until completion or turn limit
	 *
	 * @returns AgenticFlowResult indicating if the flow handled the request
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
		const hostManager = await mcpStore.ensureInitialized(perChatOverrides);
		if (!hostManager) {
			console.log('[AgenticStore] MCP not initialized, falling back to standard chat');
			return { handled: false };
		}

		const tools = mcpStore.getToolDefinitions();
		if (tools.length === 0) {
			console.log('[AgenticStore] No tools available, falling back to standard chat');
			return { handled: false };
		}

		console.log(`[AgenticStore] Starting agentic flow with ${tools.length} tools`);

		// Normalize messages to API format
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
				// Filter out empty system messages
				if (msg.role === 'system') {
					const content = typeof msg.content === 'string' ? msg.content : '';
					return content.trim().length > 0;
				}
				return true;
			});

		// Reset state
		this._isRunning = true;
		this._currentTurn = 0;
		this._totalToolCalls = 0;
		this._lastError = null;

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
			this._lastError = normalizedError;
			onError?.(normalizedError);
			return { handled: true, error: normalizedError };
		} finally {
			this._isRunning = false;
			// Lazy Disconnect: Close MCP connections after agentic flow completes
			// This prevents continuous keepalive/heartbeat polling when tools are not in use
			await mcpStore.shutdown().catch((err) => {
				console.warn('[AgenticStore] Failed to shutdown MCP after flow:', err);
			});

			console.log('[AgenticStore] MCP connections closed (lazy disconnect)');
		}
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Private: Agentic Loop Implementation
	// ─────────────────────────────────────────────────────────────────────────────

	private async executeAgenticLoop(params: {
		messages: ApiChatMessageData[];
		options: AgenticFlowOptions;
		tools: ReturnType<typeof mcpStore.getToolDefinitions>;
		agenticConfig: ReturnType<typeof getAgenticConfig>;
		callbacks: AgenticFlowCallbacks;
		signal?: AbortSignal;
	}): Promise<void> {
		const { messages, options, tools, agenticConfig, callbacks, signal } = params;
		const { onChunk, onReasoningChunk, onToolCallChunk, onModel, onComplete, onTimings } =
			callbacks;

		// Set up LLM client
		const llmClient = new OpenAISseClient({
			url: './v1/chat/completions',
			buildHeaders: () => getAuthHeaders()
		});

		// Prepare session state
		const sessionMessages: AgenticMessage[] = toAgenticMessages(messages);
		const allToolCalls: ApiChatCompletionToolCall[] = [];
		let capturedTimings: ChatMessageTimings | undefined;

		// Build base request from options (messages change per turn)
		const requestBase: AgenticChatCompletionRequest = {
			...options,
			stream: true,
			messages: []
		};

		const maxTurns = agenticConfig.maxTurns;
		const maxToolPreviewLines = agenticConfig.maxToolPreviewLines;

		// Run agentic loop
		for (let turn = 0; turn < maxTurns; turn++) {
			this._currentTurn = turn + 1;

			if (signal?.aborted) {
				onComplete?.('', undefined, capturedTimings, undefined);
				return;
			}

			// Build LLM request for this turn
			const llmRequest: AgenticChatCompletionRequest = {
				...requestBase,
				messages: sessionMessages,
				tools: tools.length > 0 ? tools : undefined
			};

			// Filter reasoning content after first turn if configured
			const shouldFilterReasoning = agenticConfig.filterReasoningAfterFirstTurn && turn > 0;

			// Stream from LLM
			let turnResult: OpenAISseTurnResult;
			try {
				turnResult = await llmClient.stream(
					llmRequest,
					{
						onChunk,
						onReasoningChunk: shouldFilterReasoning ? undefined : onReasoningChunk,
						onModel,
						onFirstValidChunk: undefined,
						onProcessingUpdate: (timings, progress) => {
							onTimings?.(timings, progress);
							if (timings) capturedTimings = timings;
						}
					},
					signal
				);
			} catch (error) {
				if (signal?.aborted) {
					onComplete?.('', undefined, capturedTimings, undefined);
					return;
				}
				const normalizedError = error instanceof Error ? error : new Error('LLM stream error');
				onChunk?.(`\n\n\`\`\`\nUpstream LLM error:\n${normalizedError.message}\n\`\`\`\n`);
				onComplete?.('', undefined, capturedTimings, undefined);
				throw normalizedError;
			}

			// Check if we should stop (no tool calls or finish reason isn't tool_calls)
			if (
				turnResult.toolCalls.length === 0 ||
				(turnResult.finishReason && turnResult.finishReason !== 'tool_calls')
			) {
				onComplete?.('', undefined, capturedTimings, undefined);
				return;
			}

			// Normalize and validate tool calls
			const normalizedCalls = this.normalizeToolCalls(turnResult.toolCalls);
			if (normalizedCalls.length === 0) {
				onComplete?.('', undefined, capturedTimings, undefined);
				return;
			}

			// Accumulate tool calls
			for (const call of normalizedCalls) {
				allToolCalls.push({
					id: call.id,
					type: call.type,
					function: call.function ? { ...call.function } : undefined
				});
			}
			this._totalToolCalls = allToolCalls.length;
			onToolCallChunk?.(JSON.stringify(allToolCalls));

			// Add assistant message with tool calls to session
			sessionMessages.push({
				role: 'assistant',
				content: turnResult.content || undefined,
				tool_calls: normalizedCalls
			});

			// Execute each tool call via MCP
			for (const toolCall of normalizedCalls) {
				if (signal?.aborted) {
					onComplete?.('', undefined, capturedTimings, undefined);
					return;
				}

				// Emit tool call start (shows "pending" state in UI)
				this.emitToolCallStart(toolCall, onChunk);

				const mcpCall: MCPToolCall = {
					id: toolCall.id,
					function: {
						name: toolCall.function.name,
						arguments: toolCall.function.arguments
					}
				};

				let result: string;
				try {
					const executionResult = await mcpStore.executeTool(mcpCall, signal);
					result = executionResult.content;
				} catch (error) {
					if (error instanceof Error && error.name === 'AbortError') {
						onComplete?.('', undefined, capturedTimings, undefined);
						return;
					}
					result = `Error: ${error instanceof Error ? error.message : String(error)}`;
				}

				if (signal?.aborted) {
					onComplete?.('', undefined, capturedTimings, undefined);
					return;
				}

				// Emit tool result and end marker
				this.emitToolCallResult(result, maxToolPreviewLines, onChunk);

				// Add tool result to session (sanitize base64 images for context)
				const contextValue = this.isBase64Image(result) ? '[Image displayed to user]' : result;
				sessionMessages.push({
					role: 'tool',
					tool_call_id: toolCall.id,
					content: contextValue
				});
			}
		}

		// Turn limit reached
		onChunk?.('\n\n```\nTurn limit reached\n```\n');
		onComplete?.('', undefined, capturedTimings, undefined);
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Private: Helper Methods
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Normalize tool calls from LLM response
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
	 * Check if content is a base64 image
	 */
	private isBase64Image(content: string): boolean {
		const trimmed = content.trim();
		if (!trimmed.startsWith('data:image/')) return false;

		const match = trimmed.match(/^data:image\/(png|jpe?g|gif|webp);base64,([A-Za-z0-9+/]+=*)$/);
		if (!match) return false;

		const base64Payload = match[2];
		return base64Payload.length > 0 && base64Payload.length % 4 === 0;
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Utilities
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Clear error state
	 */
	clearError(): void {
		this._lastError = null;
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Singleton Instance & Exports
// ─────────────────────────────────────────────────────────────────────────────

export const agenticStore = new AgenticStore();

// Reactive exports for components
export function agenticIsRunning() {
	return agenticStore.isRunning;
}

export function agenticCurrentTurn() {
	return agenticStore.currentTurn;
}

export function agenticTotalToolCalls() {
	return agenticStore.totalToolCalls;
}

export function agenticLastError() {
	return agenticStore.lastError;
}
