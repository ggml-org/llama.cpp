/**
 * agenticStore - Reactive State Store for Agentic Loop
 *
 * This store contains ONLY reactive state ($state).
 * All business logic is delegated to AgenticClient.
 *
 * **Architecture & Relationships:**
 * - **AgenticClient**: Business logic facade (loop orchestration, tool execution)
 * - **MCPClient**: Tool execution via MCP servers
 * - **agenticStore** (this): Reactive state for UI components
 *
 * **Responsibilities:**
 * - Hold per-conversation reactive state for UI binding
 * - Provide getters for computed values (scoped by conversationId)
 * - Expose setters for AgenticClient to update state
 * - Forward method calls to AgenticClient
 * - Track sampling requests for debugging
 *
 * **Per-Conversation Architecture:**
 * - Each conversation has its own AgenticSession
 * - Parallel agentic flows in different chats don't interfere
 * - Sessions are created on-demand and cleaned up when done
 *
 * @see AgenticClient in clients/agentic/ for business logic
 * @see MCPClient in clients/mcp/ for tool execution
 */

import { browser } from '$app/environment';
import type { AgenticFlowParams, AgenticFlowResult } from '$lib/clients';
import type { AgenticSession } from '$lib/types/agentic';

export type {
	AgenticFlowCallbacks,
	AgenticFlowOptions,
	AgenticFlowParams,
	AgenticFlowResult
} from '$lib/clients';

/**
 * Creates a fresh agentic session with default values.
 */
function createDefaultSession(): AgenticSession {
	return {
		isRunning: false,
		currentTurn: 0,
		totalToolCalls: 0,
		lastError: null,
		streamingToolCall: null
	};
}

class AgenticStore {
	/**
	 * Per-conversation agentic sessions.
	 * Key is conversationId, value is the session state.
	 */
	private _sessions = $state<Map<string, AgenticSession>>(new Map());

	/** Reference to the client (lazy loaded to avoid circular dependency) */
	private _client: typeof import('$lib/clients/agentic.client').agenticClient | null = null;

	private get client() {
		return this._client;
	}

	/** Check if store is ready (client initialized) */
	get isReady(): boolean {
		return this._client !== null;
	}

	/**
	 * Initialize the store by wiring up to the client.
	 * Must be called once after app startup.
	 */
	async init(): Promise<void> {
		if (!browser) return;
		if (this._client) return; // Already initialized

		const { agenticClient } = await import('$lib/clients/agentic.client');
		this._client = agenticClient;

		agenticClient.setStoreCallbacks({
			setRunning: (convId, running) => this.updateSession(convId, { isRunning: running }),
			setCurrentTurn: (convId, turn) => this.updateSession(convId, { currentTurn: turn }),
			setTotalToolCalls: (convId, count) => this.updateSession(convId, { totalToolCalls: count }),
			setLastError: (convId, error) => this.updateSession(convId, { lastError: error }),
			setStreamingToolCall: (convId, tc) => this.updateSession(convId, { streamingToolCall: tc }),
			clearStreamingToolCall: (convId) => this.updateSession(convId, { streamingToolCall: null })
		});
	}

	/**
	 *
	 * Session Management
	 *
	 */

	/**
	 * Get session for a conversation, creating if needed.
	 */
	getSession(conversationId: string): AgenticSession {
		let session = this._sessions.get(conversationId);
		if (!session) {
			session = createDefaultSession();
			this._sessions.set(conversationId, session);
		}
		return session;
	}

	/**
	 * Update session state for a conversation.
	 */
	private updateSession(conversationId: string, update: Partial<AgenticSession>): void {
		const session = this.getSession(conversationId);
		const updated = { ...session, ...update };
		this._sessions.set(conversationId, updated);
	}

	/**
	 * Clear session for a conversation.
	 */
	clearSession(conversationId: string): void {
		this._sessions.delete(conversationId);
	}

	/**
	 * Get all active sessions (conversations with running agentic flows).
	 */
	getActiveSessions(): Array<{ conversationId: string; session: AgenticSession }> {
		const active: Array<{ conversationId: string; session: AgenticSession }> = [];
		for (const [conversationId, session] of this._sessions.entries()) {
			if (session.isRunning) {
				active.push({ conversationId, session });
			}
		}
		return active;
	}

	/**
	 *
	 * Convenience Getters (for current/active conversation)
	 *
	 */

	/**
	 * Check if any agentic flow is running (global).
	 */
	get isAnyRunning(): boolean {
		for (const session of this._sessions.values()) {
			if (session.isRunning) return true;
		}
		return false;
	}

	/**
	 * Get running state for a specific conversation.
	 */
	isRunning(conversationId: string): boolean {
		return this.getSession(conversationId).isRunning;
	}

	/**
	 * Get current turn for a specific conversation.
	 */
	currentTurn(conversationId: string): number {
		return this.getSession(conversationId).currentTurn;
	}

	/**
	 * Get total tool calls for a specific conversation.
	 */
	totalToolCalls(conversationId: string): number {
		return this.getSession(conversationId).totalToolCalls;
	}

	/**
	 * Get last error for a specific conversation.
	 */
	lastError(conversationId: string): Error | null {
		return this.getSession(conversationId).lastError;
	}

	/**
	 * Get streaming tool call for a specific conversation.
	 */
	streamingToolCall(conversationId: string): { name: string; arguments: string } | null {
		return this.getSession(conversationId).streamingToolCall;
	}

	/**
	 *
	 * Agentic Flow Execution
	 *
	 */

	/**
	 * Run the agentic orchestration loop with MCP tools.
	 * Delegates to AgenticClient.
	 */
	async runAgenticFlow(params: AgenticFlowParams): Promise<AgenticFlowResult> {
		if (!this.client) {
			throw new Error('AgenticStore not initialized. Call init() first.');
		}
		return this.client.runAgenticFlow(params);
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

		// Wrapper to emit agentic tags progressively during streaming
		const emittedToolCallStates = $state(
			new Map<number, { emittedOnce: boolean; lastArgs: string }>()
		);
		const wrappedOnToolCallChunk = (serializedToolCalls: string) => {
			const toolCalls: ApiChatCompletionToolCall[] = JSON.parse(serializedToolCalls);

			for (let i = 0; i < toolCalls.length; i++) {
				const toolCall = toolCalls[i];
				const toolName = toolCall.function?.name ?? '';
				const toolArgs = toolCall.function?.arguments ?? '';

				const state = emittedToolCallStates.get(i) || { emittedOnce: false, lastArgs: '' };

				if (!state.emittedOnce) {
					// First emission: send full header + args
					let output = `\n\n<<<AGENTIC_TOOL_CALL_START>>>`;
					output += `\n<<<TOOL_NAME:${toolName}>>>`;
					output += `\n<<<TOOL_ARGS_START>>>\n`;
					output += toolArgs;
					onChunk?.(output);
					state.emittedOnce = true;
					state.lastArgs = toolArgs;
				} else if (toolArgs !== state.lastArgs) {
					// Subsequent emissions: send only delta
					const delta = toolArgs.slice(state.lastArgs.length);
					onChunk?.(delta);
					state.lastArgs = toolArgs;
				}

				emittedToolCallStates.set(i, state);
			}

			onToolCallChunk?.(serializedToolCalls);
		};
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
			emittedToolCallStates.clear(); // Reset for new turn
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
						onToolCallChunk: wrappedOnToolCallChunk,
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
	 * Emit tool call result and end marker.
	 */
	private emitToolCallResult(
		result: string,
		maxLines: number,
		emit?: (chunk: string) => void
	): void {
		if (!emit) return;

		let output = '';
		output += `\n<<<TOOL_ARGS_END>>>`;
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
	 * Clear error state for a conversation.
	 */
	clearError(conversationId: string): void {
		this.updateSession(conversationId, { lastError: null });
	}
}

export const agenticStore = new AgenticStore();

// Auto-initialize in browser
if (browser) {
	agenticStore.init();
}

/**
 * Helper functions for reactive access in components.
 * These require conversationId parameter for per-conversation state.
 */
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
