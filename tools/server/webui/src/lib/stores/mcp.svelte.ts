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

import { mcpClient } from '$lib/clients/mcp.client';
import type {
	HealthCheckState,
	MCPServerSettingsEntry,
	McpServerUsageStats,
	MCPPromptInfo,
	GetPromptResult
} from '$lib/types';
import type { McpServerOverride } from '$lib/types/database';
import { buildMcpClientConfig, parseMcpServerSettings, getMcpServerLabel } from '$lib/utils/mcp';
import { HealthCheckStatus } from '$lib/enums';
import { config, settingsStore } from '$lib/stores/settings.svelte';
import { DEFAULT_MCP_CONFIG } from '$lib/constants/mcp';

/**
 * Parses MCP server usage stats from settings.
 * @param rawStats - The raw stats to parse
 * @returns MCP server usage stats or empty object if invalid
 */
function parseMcpServerUsageStats(rawStats: unknown): McpServerUsageStats {
	if (!rawStats) return {};

	if (typeof rawStats === 'string') {
		const trimmed = rawStats.trim();
		if (!trimmed) return {};

		try {
			const parsed = JSON.parse(trimmed);
			if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
				return parsed as McpServerUsageStats;
			}
		} catch {
			console.warn('[MCP] Failed to parse mcpServerUsageStats JSON, ignoring value');
		}
	}

	return {};
}

class MCPStore {
	private _isInitializing = $state(false);
	private _error = $state<string | null>(null);
	private _toolCount = $state(0);
	private _connectedServers = $state<string[]>([]);
	private _healthChecks = $state<Record<string, HealthCheckState>>({});

	/**
	 * Update state from MCPClient
	 */
	updateState(state: {
		isInitializing?: boolean;
		error?: string | null;
		toolCount?: number;
		connectedServers?: string[];
	}): void {
		if (state.isInitializing !== undefined) this._isInitializing = state.isInitializing;
		if (state.error !== undefined) this._error = state.error;
		if (state.toolCount !== undefined) this._toolCount = state.toolCount;
		if (state.connectedServers !== undefined) this._connectedServers = state.connectedServers;
	}

	/**
	 * Update health check state from MCPClient.
	 */
	updateHealthCheck(serverId: string, state: HealthCheckState): void {
		this._healthChecks = { ...this._healthChecks, [serverId]: state };
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
	 * Clear health check state for a specific server.
	 */
	clearHealthCheck(serverId: string): void {
		// eslint-disable-next-line @typescript-eslint/no-unused-vars
		const { [serverId]: _removed, ...rest } = this._healthChecks;
		this._healthChecks = rest;
	}

	/**
	 * Clear all health check states.
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

	/**
	 *
	 * Server Management (CRUD)
	 *
	 */

	/**
	 * Get all configured MCP servers from settings (unsorted).
	 */
	getServers(): MCPServerSettingsEntry[] {
		return parseMcpServerSettings(config().mcpServers);
	}

	/**
	 * Check if any server is still loading (idle or connecting).
	 */
	isAnyServerLoading(): boolean {
		const servers = this.getServers();
		return servers.some((s) => {
			const state = this.getHealthCheckState(s.id);
			return (
				state.status === HealthCheckStatus.Idle || state.status === HealthCheckStatus.Connecting
			);
		});
	}

	/**
	 * Get servers sorted alphabetically by display label.
	 * Returns unsorted list while health checks are in progress to prevent UI jumping.
	 */
	getServersSorted(): MCPServerSettingsEntry[] {
		const servers = this.getServers();

		// Don't sort while any server is still loading - prevents UI jumping
		if (this.isAnyServerLoading()) {
			return servers;
		}

		// Sort alphabetically by display label once all health checks are done
		return [...servers].sort((a, b) => {
			const labelA = getMcpServerLabel(a, this.getHealthCheckState(a.id));
			const labelB = getMcpServerLabel(b, this.getHealthCheckState(b.id));
			return labelA.localeCompare(labelB);
		});
	}

	/**
	 * Add a new MCP server
	 */
	addServer(
		serverData: Omit<MCPServerSettingsEntry, 'id' | 'requestTimeoutSeconds'> & { id?: string }
	): void {
		const servers = this.getServers();
		const newServer: MCPServerSettingsEntry = {
			id: serverData.id || (crypto.randomUUID ? crypto.randomUUID() : `server-${Date.now()}`),
			enabled: serverData.enabled,
			url: serverData.url.trim(),
			name: serverData.name,
			headers: serverData.headers?.trim() || undefined,
			requestTimeoutSeconds: DEFAULT_MCP_CONFIG.requestTimeoutSeconds
		};
		settingsStore.updateConfig('mcpServers', JSON.stringify([...servers, newServer]));
	}

	/**
	 * Update an existing MCP server
	 */
	updateServer(id: string, updates: Partial<MCPServerSettingsEntry>): void {
		const servers = this.getServers();
		const nextServers = servers.map((server) =>
			server.id === id ? { ...server, ...updates } : server
		);
		settingsStore.updateConfig('mcpServers', JSON.stringify(nextServers));
	}

	/**
	 * Remove an MCP server by ID
	 */
	removeServer(id: string): void {
		const servers = this.getServers();
		settingsStore.updateConfig('mcpServers', JSON.stringify(servers.filter((s) => s.id !== id)));
		this.clearHealthCheck(id);
	}

	/**
	 * Check if there are any available MCP servers (enabled in settings).
	 * Used to determine if McpSelector should be shown.
	 */
	hasAvailableServers(): boolean {
		const servers = parseMcpServerSettings(config().mcpServers);
		return servers.some((s) => s.enabled && s.url.trim());
	}

	/**
	 * Check if there are any MCP servers enabled for the current chat.
	 */
	hasEnabledServers(perChatOverrides?: McpServerOverride[]): boolean {
		return Boolean(buildMcpClientConfig(config(), perChatOverrides));
	}

	/**
	 *
	 * Server Usage Stats
	 *
	 */

	/**
	 * Get parsed usage stats for all servers
	 */
	getUsageStats(): McpServerUsageStats {
		return parseMcpServerUsageStats(config().mcpServerUsageStats);
	}

	/**
	 * Get usage count for a specific server
	 */
	getServerUsageCount(serverId: string): number {
		const stats = this.getUsageStats();
		return stats[serverId] || 0;
	}

	/**
	 * Increment usage count for a server
	 */
	incrementServerUsage(serverId: string): void {
		const stats = this.getUsageStats();
		stats[serverId] = (stats[serverId] || 0) + 1;
		settingsStore.updateConfig('mcpServerUsageStats', JSON.stringify(stats));
	}

	/**
	 *
	 * Prompts
	 *
	 */

	/**
	 * Check if any connected server supports prompts
	 */
	hasPromptsSupport(): boolean {
		return mcpClient.hasPromptsSupport();
	}

	/**
	 * Get all prompts from all connected servers
	 */
	async getAllPrompts(): Promise<MCPPromptInfo[]> {
		return mcpClient.getAllPrompts();
	}

	/**
	 * Get a specific prompt by name from a server.
	 * Throws an error if the server is not found or prompt execution fails.
	 */
	async getPrompt(
		serverName: string,
		promptName: string,
		args?: Record<string, string>
	): Promise<GetPromptResult> {
		return mcpClient.getPrompt(serverName, promptName, args);
	}
}

export const mcpStore = new MCPStore();

export const mcpIsInitializing = () => mcpStore.isInitializing;
export const mcpIsInitialized = () => mcpStore.isInitialized;
export const mcpError = () => mcpStore.error;
export const mcpIsEnabled = () => mcpStore.isEnabled;
export const mcpAvailableTools = () => mcpStore.availableTools;
export const mcpConnectedServerCount = () => mcpStore.connectedServerCount;
export const mcpConnectedServerNames = () => mcpStore.connectedServerNames;
export const mcpToolCount = () => mcpStore.toolCount;
