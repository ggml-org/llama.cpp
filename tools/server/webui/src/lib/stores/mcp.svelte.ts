import { browser } from '$app/environment';
import { MCPClient, type IMCPClient } from '$lib/mcp';
import { buildMcpClientConfig } from '$lib/config/mcp';
import { config } from '$lib/stores/settings.svelte';

/**
 * mcpStore - Reactive store for MCP (Model Context Protocol) client management
 *
 * This store manages:
 * - MCP client lifecycle (initialization, shutdown)
 * - Connection state tracking
 * - Available tools from connected MCP servers
 * - Error handling for MCP operations
 *
 * **Architecture & Relationships:**
 * - **MCPClient**: SDK-based client wrapper for MCP server communication
 * - **mcpStore** (this class): Reactive store for MCP state
 * - **ChatService**: Uses mcpStore for agentic orchestration
 * - **settingsStore**: Provides MCP server configuration
 *
 * **Key Features:**
 * - Reactive state with Svelte 5 runes ($state, $derived)
 * - Automatic reinitialization on config changes
 * - Graceful error handling with fallback to standard chat
 */
class MCPStore {
	// ─────────────────────────────────────────────────────────────────────────────
	// State
	// ─────────────────────────────────────────────────────────────────────────────

	private _client = $state<MCPClient | null>(null);
	private _isInitializing = $state(false);
	private _error = $state<string | null>(null);
	private _configSignature = $state<string | null>(null);
	private _initPromise: Promise<MCPClient | undefined> | null = null;

	// ─────────────────────────────────────────────────────────────────────────────
	// Computed Getters
	// ─────────────────────────────────────────────────────────────────────────────

	get client(): IMCPClient | null {
		return this._client;
	}

	get isInitializing(): boolean {
		return this._isInitializing;
	}

	get isInitialized(): boolean {
		return this._client !== null;
	}

	get error(): string | null {
		return this._error;
	}

	/**
	 * Check if MCP is enabled (has configured servers)
	 */
	get isEnabled(): boolean {
		const mcpConfig = buildMcpClientConfig(config());
		return (
			mcpConfig !== null && mcpConfig !== undefined && Object.keys(mcpConfig.servers).length > 0
		);
	}

	/**
	 * Get list of available tool names
	 */
	get availableTools(): string[] {
		return this._client?.listTools() ?? [];
	}

	/**
	 * Get tool definitions for LLM
	 */
	async getToolDefinitions(): Promise<
		{
			type: 'function';
			function: { name: string; description?: string; parameters: Record<string, unknown> };
		}[]
	> {
		if (!this._client) return [];
		return this._client.getToolsDefinition();
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Lifecycle
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Ensure MCP client is initialized with current config.
	 * Returns the client if successful, undefined otherwise.
	 * Handles config changes by reinitializing as needed.
	 */
	async ensureClient(): Promise<IMCPClient | undefined> {
		if (!browser) return undefined;

		const mcpConfig = buildMcpClientConfig(config());
		const signature = mcpConfig ? JSON.stringify(mcpConfig) : null;

		// No config - shutdown if needed
		if (!signature) {
			await this.shutdown();
			return undefined;
		}

		// Already initialized with correct config
		if (this._client && this._configSignature === signature) {
			return this._client;
		}

		// Init in progress with correct config - wait for it
		if (this._initPromise && this._configSignature === signature) {
			return this._initPromise;
		}

		// Config changed or first init - shutdown old client first
		if (this._client || this._initPromise) {
			await this.shutdown();
		}

		// Initialize new client
		return this.initialize(signature, mcpConfig!);
	}

	/**
	 * Initialize MCP client with given config
	 */
	private async initialize(
		signature: string,
		mcpConfig: ReturnType<typeof buildMcpClientConfig>
	): Promise<MCPClient | undefined> {
		if (!mcpConfig) return undefined;

		this._isInitializing = true;
		this._error = null;
		this._configSignature = signature;

		const client = new MCPClient(mcpConfig);

		this._initPromise = client
			.initialize()
			.then(() => {
				// Check if config changed during initialization
				if (this._configSignature !== signature) {
					void client.shutdown().catch((err) => {
						console.error('[MCP Store] Failed to shutdown stale client:', err);
					});
					return undefined;
				}

				this._client = client;
				this._isInitializing = false;
				console.log(
					`[MCP Store] Initialized with ${client.listTools().length} tools:`,
					client.listTools()
				);
				return client;
			})
			.catch((error) => {
				console.error('[MCP Store] Initialization failed:', error);
				this._error = error instanceof Error ? error.message : String(error);
				this._isInitializing = false;

				void client.shutdown().catch((err) => {
					console.error('[MCP Store] Failed to shutdown after error:', err);
				});

				return undefined;
			})
			.finally(() => {
				if (this._configSignature === signature) {
					this._initPromise = null;
				}
			});

		return this._initPromise;
	}

	/**
	 * Shutdown MCP client and clear state
	 */
	async shutdown(): Promise<void> {
		// Wait for any pending initialization
		if (this._initPromise) {
			await this._initPromise.catch(() => {});
			this._initPromise = null;
		}

		if (this._client) {
			const clientToShutdown = this._client;
			this._client = null;
			this._configSignature = null;
			this._error = null;

			try {
				await clientToShutdown.shutdown();
				console.log('[MCP Store] Client shutdown complete');
			} catch (error) {
				console.error('[MCP Store] Shutdown error:', error);
			}
		}
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Tool Execution
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Execute a tool call via MCP client
	 */
	async execute(
		toolCall: { id: string; function: { name: string; arguments: string } },
		abortSignal?: AbortSignal
	): Promise<string> {
		if (!this._client) {
			throw new Error('MCP client not initialized');
		}
		return this._client.execute(toolCall, abortSignal);
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Utilities
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Clear error state
	 */
	clearError(): void {
		this._error = null;
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Singleton Instance & Exports
// ─────────────────────────────────────────────────────────────────────────────

export const mcpStore = new MCPStore();

// Reactive exports for components
export function mcpClient() {
	return mcpStore.client;
}

export function mcpIsInitializing() {
	return mcpStore.isInitializing;
}

export function mcpIsInitialized() {
	return mcpStore.isInitialized;
}

export function mcpError() {
	return mcpStore.error;
}

export function mcpIsEnabled() {
	return mcpStore.isEnabled;
}

export function mcpAvailableTools() {
	return mcpStore.availableTools;
}
