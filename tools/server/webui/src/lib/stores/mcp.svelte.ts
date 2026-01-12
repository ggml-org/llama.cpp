/**
 * mcpStore - Reactive State Store for MCP (Model Context Protocol)
 *
 * This store contains ONLY reactive state ($state, $derived).
 * All business logic is delegated to MCPClient.
 *
 * **Architecture & Relationships:**
 * - **MCPClient**: Business logic facade (lifecycle, tool execution, health checks)
 * - **MCPService**: Stateless protocol layer (transport, connect, callTool)
 * - **mcpStore** (this): Reactive state for UI components
 *
 * **Responsibilities:**
 * - Hold reactive state for UI binding
 * - Provide getters for computed values
 * - Expose setters for MCPClient to update state
 * - Forward method calls to MCPClient
 *
 * @see MCPClient in clients/mcp/ for business logic
 * @see MCPService in services/mcp.ts for protocol operations
 */

import { browser } from '$app/environment';
import { mcpClient, type HealthCheckState, type HealthCheckParams } from '$lib/clients';
import type {
	OpenAIToolDefinition,
	ServerStatus,
	ToolExecutionResult,
	MCPToolCall
} from '$lib/types/mcp';
import type { McpServerOverride } from '$lib/types/database';
import { buildMcpClientConfig } from '$lib/utils/mcp';
import { config } from '$lib/stores/settings.svelte';

export type { HealthCheckState };

class MCPStore {
	private _isInitializing = $state(false);
	private _error = $state<string | null>(null);
	private _toolCount = $state(0);
	private _connectedServers = $state<string[]>([]);
	private _healthChecks = $state<Record<string, HealthCheckState>>({});

	constructor() {
		if (browser) {
			mcpClient.setStateChangeCallback((state) => {
				if (state.isInitializing !== undefined) {
					this._isInitializing = state.isInitializing;
				}
				if (state.error !== undefined) {
					this._error = state.error;
				}
				if (state.toolCount !== undefined) {
					this._toolCount = state.toolCount;
				}
				if (state.connectedServers !== undefined) {
					this._connectedServers = state.connectedServers;
				}
			});

			mcpClient.setHealthCheckCallback((serverId, state) => {
				this._healthChecks = { ...this._healthChecks, [serverId]: state };
			});
		}
	}

	get isInitializing(): boolean {
		return this._isInitializing;
	}

	get isInitialized(): boolean {
		return mcpClient.isInitialized;
	}

	get error(): string | null {
		return this._error;
	}

	get toolCount(): number {
		return this._toolCount;
	}

	get connectedServerCount(): number {
		return this._connectedServers.length;
	}

	get connectedServerNames(): string[] {
		return this._connectedServers;
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
		return mcpClient.getToolNames();
	}

	/**
	 * Ensure MCP is initialized with current config.
	 * @param perChatOverrides - Optional per-chat MCP server overrides
	 */
	async ensureInitialized(perChatOverrides?: McpServerOverride[]): Promise<boolean> {
		return mcpClient.ensureInitialized(perChatOverrides);
	}

	/**
	 * Shutdown MCP connections and clear state
	 */
	async shutdown(): Promise<void> {
		return mcpClient.shutdown();
	}

	/**
	 * Get tool definitions for LLM (OpenAI function calling format)
	 */
	getToolDefinitions(): OpenAIToolDefinition[] {
		return mcpClient.getToolDefinitionsForLLM();
	}

	/**
	 * Get status of all servers
	 */
	getServersStatus(): ServerStatus[] {
		return mcpClient.getServersStatus();
	}

	/**
	 * Execute a tool call via MCP.
	 */
	async executeTool(toolCall: MCPToolCall, signal?: AbortSignal): Promise<ToolExecutionResult> {
		return mcpClient.executeTool(toolCall, signal);
	}

	/**
	 * Execute a tool by name with arguments.
	 */
	async executeToolByName(
		toolName: string,
		args: Record<string, unknown>,
		signal?: AbortSignal
	): Promise<ToolExecutionResult> {
		return mcpClient.executeToolByName(toolName, args, signal);
	}

	/**
	 * Check if a tool exists
	 */
	hasTool(toolName: string): boolean {
		return mcpClient.hasTool(toolName);
	}

	/**
	 * Get which server provides a specific tool
	 */
	getToolServer(toolName: string): string | undefined {
		return mcpClient.getToolServer(toolName);
	}

	/**
	 * Get health check state for a specific server
	 */
	getHealthCheckState(serverId: string): HealthCheckState {
		return this._healthChecks[serverId] ?? { status: 'idle' };
	}

	/**
	 * Check if health check has been performed for a server
	 */
	hasHealthCheck(serverId: string): boolean {
		return serverId in this._healthChecks && this._healthChecks[serverId].status !== 'idle';
	}

	/**
	 * Run health check for a specific server
	 */
	async runHealthCheck(server: HealthCheckParams): Promise<void> {
		return mcpClient.runHealthCheck(server);
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

	/**
	 * Clear error state
	 */
	clearError(): void {
		this._error = null;
	}
}

export const mcpStore = new MCPStore();

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

export function mcpGetHealthCheckState(serverId: string) {
	return mcpStore.getHealthCheckState(serverId);
}

export function mcpHasHealthCheck(serverId: string) {
	return mcpStore.hasHealthCheck(serverId);
}

export async function mcpRunHealthCheck(server: HealthCheckParams) {
	return mcpStore.runHealthCheck(server);
}

export function mcpClearHealthCheck(serverId: string) {
	return mcpStore.clearHealthCheck(serverId);
}
