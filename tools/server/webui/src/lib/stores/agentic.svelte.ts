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
 * - Hold reactive state for UI binding (isRunning, currentTurn, etc.)
 * - Provide getters for computed values
 * - Expose setters for AgenticClient to update state
 * - Forward method calls to AgenticClient
 *
 * @see AgenticClient in clients/agentic/ for business logic
 * @see MCPClient in clients/mcp/ for tool execution
 */

import { browser } from '$app/environment';
import type { AgenticFlowParams, AgenticFlowResult } from '$lib/clients';

export type {
	AgenticFlowCallbacks,
	AgenticFlowOptions,
	AgenticFlowParams,
	AgenticFlowResult
} from '$lib/clients';

class AgenticStore {
	private _isRunning = $state(false);
	private _currentTurn = $state(0);
	private _totalToolCalls = $state(0);
	private _lastError = $state<Error | null>(null);
	private _streamingToolCall = $state<{ name: string; arguments: string } | null>(null);

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
			setRunning: (running) => (this._isRunning = running),
			setCurrentTurn: (turn) => (this._currentTurn = turn),
			setTotalToolCalls: (count) => (this._totalToolCalls = count),
			setLastError: (error) => (this._lastError = error),
			setStreamingToolCall: (tc) => (this._streamingToolCall = tc),
			clearStreamingToolCall: () => (this._streamingToolCall = null)
		});
	}

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

	get streamingToolCall(): { name: string; arguments: string } | null {
		return this._streamingToolCall;
	}

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
	 * Clear error state
	 */
	clearError(): void {
		if (!this.client) return;
		this.client.clearError();
	}
}

export const agenticStore = new AgenticStore();

// Auto-initialize in browser
if (browser) {
	agenticStore.init();
}

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

export function agenticStreamingToolCall() {
	return agenticStore.streamingToolCall;
}
