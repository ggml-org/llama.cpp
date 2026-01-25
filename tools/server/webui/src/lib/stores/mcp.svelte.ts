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

import { mcpClient, buildMcpClientConfig } from '$lib/clients/mcp.client';
import type {
	HealthCheckState,
	MCPServerSettingsEntry,
	MCPPromptInfo,
	GetPromptResult
} from '$lib/types';
import type { McpServerOverride } from '$lib/types/database';
import { parseMcpServerSettings } from '$lib/utils';
import { HealthCheckStatus } from '$lib/enums';
import { config, settingsStore } from '$lib/stores/settings.svelte';
import { DEFAULT_MCP_CONFIG } from '$lib/constants/mcp';

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
	 * Gets a display label for an MCP server.
	 * Automatically fetches health state from store.
	 */
	getServerLabel(server: MCPServerSettingsEntry): string {
		const healthState = this.getHealthCheckState(server.id);
		if (healthState?.status === HealthCheckStatus.SUCCESS) {
			return (
				healthState.serverInfo?.title || healthState.serverInfo?.name || server.name || server.url
			);
		}

		return server.url;
	}

	/**
	 * Check if any server is still loading (idle or connecting).
	 */
	isAnyServerLoading(): boolean {
		const servers = this.getServers();
		return servers.some((s) => {
			const state = this.getHealthCheckState(s.id);
			return (
				state.status === HealthCheckStatus.IDLE || state.status === HealthCheckStatus.CONNECTING
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
			const labelA = this.getServerLabel(a);
			const labelB = this.getServerLabel(b);
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
	 * Gets enabled MCP servers for a conversation based on per-chat overrides.
	 * Returns servers that are both globally enabled AND enabled for this chat.
	 */
	getEnabledServersForConversation(
		perChatOverrides?: McpServerOverride[]
	): MCPServerSettingsEntry[] {
		if (!perChatOverrides?.length) {
			return [];
		}

		const allServers = this.getServers();

		return allServers.filter((server) => {
			if (!server.enabled) return false;
			const override = perChatOverrides.find((o) => o.serverId === server.id);

			return override?.enabled ?? false;
		});
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
