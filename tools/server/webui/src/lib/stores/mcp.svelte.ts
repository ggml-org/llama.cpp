import { browser } from '$app/environment';
import {
	MCPHostManager,
	type OpenAIToolDefinition,
	type ServerStatus
} from '$lib/mcp/host-manager';
import type { ToolExecutionResult } from '$lib/mcp/server-connection';
import { buildMcpClientConfig, incrementMcpServerUsage } from '$lib/config/mcp';
import { config, settingsStore } from '$lib/stores/settings.svelte';
import type { MCPToolCall } from '$lib/types/mcp';
import type { McpServerOverride } from '$lib/types/database';
import { DEFAULT_MCP_CONFIG } from '$lib/constants/mcp';
import { MCPClient } from '$lib/mcp';
import { detectMcpTransportFromUrl } from '$lib/utils/mcp';

// ─────────────────────────────────────────────────────────────────────────────
// Health Check Types
// ─────────────────────────────────────────────────────────────────────────────

export type HealthCheckState =
	| { status: 'idle' }
	| { status: 'loading' }
	| { status: 'error'; message: string }
	| { status: 'success'; tools: { name: string; description?: string }[] };

/**
 * mcpStore - Reactive store for MCP (Model Context Protocol) host management
 *
 * This store manages:
 * - MCPHostManager lifecycle (initialization, shutdown)
 * - Connection state tracking for multiple MCP servers
 * - Aggregated tools from all connected MCP servers
 * - Error handling for MCP operations
 *
 * **Architecture & Relationships:**
 * - **MCPHostManager**: Coordinates multiple MCPServerConnection instances
 * - **MCPServerConnection**: Single SDK Client wrapper per server
 * - **mcpStore** (this class): Reactive Svelte store for MCP state
 * - **agenticStore**: Uses mcpStore for tool execution in agentic loop
 * - **settingsStore**: Provides MCP server configuration
 *
 * **Key Features:**
 * - Reactive state with Svelte 5 runes ($state, $derived)
 * - Automatic reinitialization on config changes
 * - Aggregates tools from multiple servers
 * - Routes tool calls to appropriate server automatically
 * - Graceful error handling with fallback to standard chat
 */
class MCPStore {
	// ─────────────────────────────────────────────────────────────────────────────
	// State
	// ─────────────────────────────────────────────────────────────────────────────

	private _hostManager = $state<MCPHostManager | null>(null);
	private _isInitializing = $state(false);
	private _error = $state<string | null>(null);
	private _configSignature = $state<string | null>(null);
	private _initPromise: Promise<MCPHostManager | undefined> | null = null;

	// Health check state (in-memory only, not persisted)
	private _healthChecks = $state<Record<string, HealthCheckState>>({});

	// ─────────────────────────────────────────────────────────────────────────────
	// Computed Getters
	// ─────────────────────────────────────────────────────────────────────────────

	get hostManager(): MCPHostManager | null {
		return this._hostManager;
	}

	get isInitializing(): boolean {
		return this._isInitializing;
	}

	get isInitialized(): boolean {
		return this._hostManager?.isInitialized ?? false;
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
	 * Get list of available tool names (aggregated from all servers)
	 */
	get availableTools(): string[] {
		return this._hostManager?.getToolNames() ?? [];
	}

	/**
	 * Get number of connected servers
	 */
	get connectedServerCount(): number {
		return this._hostManager?.connectedServerCount ?? 0;
	}

	/**
	 * Get names of connected servers
	 */
	get connectedServerNames(): string[] {
		return this._hostManager?.connectedServerNames ?? [];
	}

	/**
	 * Get total tool count
	 */
	get toolCount(): number {
		return this._hostManager?.toolCount ?? 0;
	}

	/**
	 * Get tool definitions for LLM (OpenAI function calling format)
	 */
	getToolDefinitions(): OpenAIToolDefinition[] {
		return this._hostManager?.getToolDefinitionsForLLM() ?? [];
	}

	/**
	 * Get status of all servers
	 */
	getServersStatus(): ServerStatus[] {
		return this._hostManager?.getServersStatus() ?? [];
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Lifecycle
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Ensure MCP host manager is initialized with current config.
	 * Returns the host manager if successful, undefined otherwise.
	 * Handles config changes by reinitializing as needed.
	 * @param perChatOverrides - Optional per-chat MCP server overrides
	 */
	async ensureInitialized(
		perChatOverrides?: McpServerOverride[]
	): Promise<MCPHostManager | undefined> {
		if (!browser) return undefined;

		const mcpConfig = buildMcpClientConfig(config(), perChatOverrides);
		const signature = mcpConfig ? JSON.stringify(mcpConfig) : null;

		// No config - shutdown if needed
		if (!signature) {
			await this.shutdown();
			return undefined;
		}

		// Already initialized with correct config
		if (this._hostManager?.isInitialized && this._configSignature === signature) {
			return this._hostManager;
		}

		// Init in progress with correct config - wait for it
		if (this._initPromise && this._configSignature === signature) {
			return this._initPromise;
		}

		// Config changed or first init - shutdown old manager first
		if (this._hostManager || this._initPromise) {
			await this.shutdown();
		}

		// Initialize new host manager
		return this.initialize(signature, mcpConfig!);
	}

	/**
	 * Initialize MCP host manager with given config
	 */
	private async initialize(
		signature: string,
		mcpConfig: NonNullable<ReturnType<typeof buildMcpClientConfig>>
	): Promise<MCPHostManager | undefined> {
		this._isInitializing = true;
		this._error = null;
		this._configSignature = signature;

		const hostManager = new MCPHostManager();

		this._initPromise = hostManager
			.initialize({
				servers: mcpConfig.servers,
				clientInfo: mcpConfig.clientInfo ?? DEFAULT_MCP_CONFIG.clientInfo,
				capabilities: mcpConfig.capabilities ?? DEFAULT_MCP_CONFIG.capabilities
			})
			.then(() => {
				// Check if config changed during initialization
				if (this._configSignature !== signature) {
					void hostManager.shutdown().catch((err) => {
						console.error('[MCP Store] Failed to shutdown stale host manager:', err);
					});
					return undefined;
				}

				this._hostManager = hostManager;
				this._isInitializing = false;

				const toolNames = hostManager.getToolNames();
				const serverNames = hostManager.connectedServerNames;

				console.log(
					`[MCP Store] Initialized: ${serverNames.length} servers, ${toolNames.length} tools`
				);
				console.log(`[MCP Store] Servers: ${serverNames.join(', ')}`);
				console.log(`[MCP Store] Tools: ${toolNames.join(', ')}`);

				return hostManager;
			})
			.catch((error) => {
				console.error('[MCP Store] Initialization failed:', error);
				this._error = error instanceof Error ? error.message : String(error);
				this._isInitializing = false;

				void hostManager.shutdown().catch((err) => {
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
	 * Shutdown MCP host manager and clear state
	 */
	async shutdown(): Promise<void> {
		// Wait for any pending initialization
		if (this._initPromise) {
			await this._initPromise.catch(() => {});
			this._initPromise = null;
		}

		if (this._hostManager) {
			const managerToShutdown = this._hostManager;
			this._hostManager = null;
			this._configSignature = null;
			this._error = null;

			try {
				await managerToShutdown.shutdown();
				console.log('[MCP Store] Host manager shutdown complete');
			} catch (error) {
				console.error('[MCP Store] Shutdown error:', error);
			}
		}
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Tool Execution
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Execute a tool call via MCP host manager.
	 * Automatically routes to the appropriate server.
	 */
	async executeTool(toolCall: MCPToolCall, signal?: AbortSignal): Promise<ToolExecutionResult> {
		if (!this._hostManager) {
			throw new Error('MCP host manager not initialized');
		}
		return this._hostManager.executeTool(toolCall, signal);
	}

	/**
	 * Execute a tool by name with arguments.
	 * Simpler interface for direct tool calls.
	 */
	async executeToolByName(
		toolName: string,
		args: Record<string, unknown>,
		signal?: AbortSignal
	): Promise<ToolExecutionResult> {
		if (!this._hostManager) {
			throw new Error('MCP host manager not initialized');
		}
		return this._hostManager.executeToolByName(toolName, args, signal);
	}

	/**
	 * Check if a tool exists
	 */
	hasTool(toolName: string): boolean {
		return this._hostManager?.hasTool(toolName) ?? false;
	}

	/**
	 * Get which server provides a specific tool
	 */
	getToolServer(toolName: string): string | undefined {
		return this._hostManager?.getToolServer(toolName);
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

	// ─────────────────────────────────────────────────────────────────────────────
	// Health Check (Settings UI)
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Get health check state for a specific server
	 */
	getHealthCheckState(serverId: string): HealthCheckState {
		return this._healthChecks[serverId] ?? { status: 'idle' };
	}

	/**
	 * Set health check state for a specific server
	 */
	private setHealthCheckState(serverId: string, state: HealthCheckState): void {
		this._healthChecks = { ...this._healthChecks, [serverId]: state };
	}

	/**
	 * Check if health check has been performed for a server
	 */
	hasHealthCheck(serverId: string): boolean {
		return serverId in this._healthChecks && this._healthChecks[serverId].status !== 'idle';
	}

	/**
	 * Parse custom headers from JSON string
	 */
	private parseHeaders(headersJson?: string): Record<string, string> | undefined {
		if (!headersJson?.trim()) return undefined;
		try {
			const parsed = JSON.parse(headersJson);
			if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
				return parsed as Record<string, string>;
			}
		} catch {
			console.warn('[MCP Store] Failed to parse custom headers JSON:', headersJson);
		}
		return undefined;
	}

	/**
	 * Run health check for a specific server
	 */
	async runHealthCheck(server: {
		id: string;
		url: string;
		requestTimeoutSeconds: number;
		headers?: string;
	}): Promise<void> {
		const trimmedUrl = server.url.trim();

		if (!trimmedUrl) {
			this.setHealthCheckState(server.id, {
				status: 'error',
				message: 'Please enter a server URL first.'
			});
			return;
		}

		this.setHealthCheckState(server.id, { status: 'loading' });

		const timeoutMs = Math.round(server.requestTimeoutSeconds * 1000);
		const headers = this.parseHeaders(server.headers);

		const mcpClient = new MCPClient({
			protocolVersion: DEFAULT_MCP_CONFIG.protocolVersion,
			capabilities: DEFAULT_MCP_CONFIG.capabilities,
			clientInfo: DEFAULT_MCP_CONFIG.clientInfo,
			requestTimeoutMs: timeoutMs,
			servers: {
				[server.id]: {
					url: trimmedUrl,
					transport: detectMcpTransportFromUrl(trimmedUrl),
					handshakeTimeoutMs: DEFAULT_MCP_CONFIG.connectionTimeoutMs,
					requestTimeoutMs: timeoutMs,
					headers
				}
			}
		});

		try {
			await mcpClient.initialize();
			const tools = (await mcpClient.getToolsDefinition()).map((tool) => ({
				name: tool.function.name,
				description: tool.function.description
			}));

			this.setHealthCheckState(server.id, { status: 'success', tools });
		} catch (error) {
			const message = error instanceof Error ? error.message : 'Unknown error occurred';
			this.setHealthCheckState(server.id, { status: 'error', message });
		} finally {
			try {
				await mcpClient.shutdown();
			} catch (shutdownError) {
				console.warn('[MCP Store] Failed to cleanly shutdown health check client', shutdownError);
			}
		}
	}

	/**
	 * Clear health check state for a specific server
	 */
	clearHealthCheck(serverId: string): void {
		// eslint-disable-next-line @typescript-eslint/no-unused-vars
		const { [serverId]: _removed, ...rest } = this._healthChecks;
		this._healthChecks = rest;
	}

	/**
	 * Clear all health check states
	 */
	clearAllHealthChecks(): void {
		this._healthChecks = {};
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Singleton Instance & Exports
// ─────────────────────────────────────────────────────────────────────────────

export const mcpStore = new MCPStore();

// Reactive exports for components
export function mcpHostManager() {
	return mcpStore.hostManager;
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

export function mcpConnectedServerCount() {
	return mcpStore.connectedServerCount;
}

export function mcpConnectedServerNames() {
	return mcpStore.connectedServerNames;
}

export function mcpToolCount() {
	return mcpStore.toolCount;
}

// Health check exports
export function mcpGetHealthCheckState(serverId: string) {
	return mcpStore.getHealthCheckState(serverId);
}

export function mcpHasHealthCheck(serverId: string) {
	return mcpStore.hasHealthCheck(serverId);
}

export async function mcpRunHealthCheck(server: {
	id: string;
	url: string;
	requestTimeoutSeconds: number;
	headers?: string;
}) {
	return mcpStore.runHealthCheck(server);
}

export function mcpClearHealthCheck(serverId: string) {
	return mcpStore.clearHealthCheck(serverId);
}
