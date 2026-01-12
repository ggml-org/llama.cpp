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
