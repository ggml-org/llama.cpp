/**
 * mcpStore - Reactive State Store for MCP Operations
 *
 * Implements the "Host" role in MCP architecture, coordinating multiple server
 * connections and providing a unified interface for tool operations.
 *
 * **Architecture & Relationships:**
 * - **MCPService**: Stateless protocol layer (transport, connect, callTool)
 * - **mcpStore** (this): Reactive state + business logic
 *
 * **Key Responsibilities:**
 * - Lifecycle management (initialize, shutdown)
 * - Multi-server coordination
 * - Tool name conflict detection and resolution
 * - OpenAI-compatible tool definition generation
 * - Automatic tool-to-server routing
 * - Health checks
 *
 * @see MCPService in services/mcp.service.ts for protocol operations
 */

import { browser } from '$app/environment';
import { MCPService } from '$lib/services/mcp.service';
import { config, settingsStore } from '$lib/stores/settings.svelte';
import { mcpResourceStore } from '$lib/stores/mcp-resources.svelte';
import { parseMcpServerSettings, detectMcpTransportFromUrl, getFaviconUrl } from '$lib/utils';
import { MCPConnectionPhase, MCPLogLevel, HealthCheckStatus, MCPRefType } from '$lib/enums';
import {
	DEFAULT_MCP_CONFIG,
	MCP_SERVER_ID_PREFIX,
	MCP_RECONNECT_INITIAL_DELAY,
	MCP_RECONNECT_BACKOFF_MULTIPLIER,
	MCP_RECONNECT_MAX_DELAY
} from '$lib/constants/mcp';
import type {
	MCPToolCall,
	OpenAIToolDefinition,
	ServerStatus,
	ToolExecutionResult,
	MCPClientConfig,
	MCPConnection,
	HealthCheckParams,
	ServerCapabilities,
	ClientCapabilities,
	MCPCapabilitiesInfo,
	MCPConnectionLog,
	MCPPromptInfo,
	GetPromptResult,
	Tool,
	HealthCheckState,
	MCPServerSettingsEntry,
	MCPServerConfig
} from '$lib/types';
import type { ListChangedHandlers } from '@modelcontextprotocol/sdk/types.js';
import type { McpServerOverride } from '$lib/types/database';
import type { SettingsConfigType } from '$lib/types/settings';

function generateMcpServerId(id: unknown, index: number): string {
	if (typeof id === 'string' && id.trim()) {
		return id.trim();
	}

	return `${MCP_SERVER_ID_PREFIX}${index + 1}`;
}

function parseServerSettings(rawServers: unknown): MCPServerSettingsEntry[] {
	if (!rawServers) {
		return [];
	}

	let parsed: unknown;
	if (typeof rawServers === 'string') {
		const trimmed = rawServers.trim();
		if (!trimmed) {
			return [];
		}

		try {
			parsed = JSON.parse(trimmed);
		} catch (error) {
			console.warn('[MCP] Failed to parse mcpServers JSON:', error);

			return [];
		}
	} else {
		parsed = rawServers;
	}
	if (!Array.isArray(parsed)) {
		return [];
	}

	return parsed.map((entry, index) => {
		const url = typeof entry?.url === 'string' ? entry.url.trim() : '';
		const headers = typeof entry?.headers === 'string' ? entry.headers.trim() : undefined;

		return {
			id: generateMcpServerId((entry as { id?: unknown })?.id, index),
			enabled: Boolean((entry as { enabled?: unknown })?.enabled),
			url,
			name: (entry as { name?: string })?.name,
			requestTimeoutSeconds: DEFAULT_MCP_CONFIG.requestTimeoutSeconds,
			headers: headers || undefined,
			useProxy: Boolean((entry as { useProxy?: unknown })?.useProxy)
		} satisfies MCPServerSettingsEntry;
	});
}

function buildServerConfig(
	entry: MCPServerSettingsEntry,
	connectionTimeoutMs = DEFAULT_MCP_CONFIG.connectionTimeoutMs
): MCPServerConfig | undefined {
	if (!entry?.url) {
		return undefined;
	}

	let headers: Record<string, string> | undefined;
	if (entry.headers) {
		try {
			const parsed = JSON.parse(entry.headers);
			if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed))
				headers = parsed as Record<string, string>;
		} catch {
			console.warn('[MCP] Failed to parse custom headers JSON:', entry.headers);
		}
	}

	return {
		url: entry.url,
		transport: detectMcpTransportFromUrl(entry.url),
		handshakeTimeoutMs: connectionTimeoutMs,
		requestTimeoutMs: Math.round(entry.requestTimeoutSeconds * 1000),
		headers,
		useProxy: entry.useProxy
	};
}

function checkServerEnabled(
	server: MCPServerSettingsEntry,
	perChatOverrides?: McpServerOverride[]
): boolean {
	if (!server.enabled) {
		return false;
	}

	if (perChatOverrides) {
		const override = perChatOverrides.find((o) => o.serverId === server.id);

		return override?.enabled ?? false;
	}

	return false;
}

function buildMcpClientConfigInternal(
	cfg: SettingsConfigType,
	perChatOverrides?: McpServerOverride[]
): MCPClientConfig | undefined {
	const rawServers = parseServerSettings(cfg.mcpServers);
	if (!rawServers.length) {
		return undefined;
	}

	const servers: Record<string, MCPServerConfig> = {};

	for (const [index, entry] of rawServers.entries()) {
		if (!checkServerEnabled(entry, perChatOverrides)) continue;
		const normalized = buildServerConfig(entry);
		if (normalized) servers[generateMcpServerId(entry.id, index)] = normalized;
	}

	if (Object.keys(servers).length === 0) {
		return undefined;
	}

	return {
		protocolVersion: DEFAULT_MCP_CONFIG.protocolVersion,
		capabilities: DEFAULT_MCP_CONFIG.capabilities,
		clientInfo: DEFAULT_MCP_CONFIG.clientInfo,
		requestTimeoutMs: Math.round(DEFAULT_MCP_CONFIG.requestTimeoutSeconds * 1000),
		servers
	};
}

function buildCapabilitiesInfo(
	serverCaps?: ServerCapabilities,
	clientCaps?: ClientCapabilities
): MCPCapabilitiesInfo {
	return {
		server: {
			tools: serverCaps?.tools ? { listChanged: serverCaps.tools.listChanged } : undefined,
			prompts: serverCaps?.prompts ? { listChanged: serverCaps.prompts.listChanged } : undefined,
			resources: serverCaps?.resources
				? {
						subscribe: serverCaps.resources.subscribe,
						listChanged: serverCaps.resources.listChanged
					}
				: undefined,
			logging: !!serverCaps?.logging,
			completions: !!serverCaps?.completions,
			tasks: !!serverCaps?.tasks
		},
		client: {
			roots: clientCaps?.roots ? { listChanged: clientCaps.roots.listChanged } : undefined,
			sampling: !!clientCaps?.sampling,
			elicitation: clientCaps?.elicitation
				? { form: !!clientCaps.elicitation.form, url: !!clientCaps.elicitation.url }
				: undefined,
			tasks: !!clientCaps?.tasks
		}
	};
}

export function buildMcpClientConfig(
	cfg: SettingsConfigType,
	perChatOverrides?: McpServerOverride[]
): MCPClientConfig | undefined {
	return buildMcpClientConfigInternal(cfg, perChatOverrides);
}

class MCPStore {
	private _isInitializing = $state(false);
	private _error = $state<string | null>(null);
	private _toolCount = $state(0);
	private _connectedServers = $state<string[]>([]);
	private _healthChecks = $state<Record<string, HealthCheckState>>({});

	private connections = new Map<string, MCPConnection>();
	private toolsIndex = new Map<string, string>();
	private serverConfigs = new Map<string, MCPServerConfig>(); // Store configs for reconnection
	private reconnectingServers = new Set<string>(); // Guard against concurrent reconnections
	private configSignature: string | null = null;
	private initPromise: Promise<boolean> | null = null;
	private activeFlowCount = 0;

	get isInitializing(): boolean {
		return this._isInitializing;
	}

	get isInitialized(): boolean {
		return this.connections.size > 0;
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

	get isEnabled(): boolean {
		const mcpConfig = buildMcpClientConfigInternal(config());
		return (
			mcpConfig !== null && mcpConfig !== undefined && Object.keys(mcpConfig.servers).length > 0
		);
	}

	get availableTools(): string[] {
		return Array.from(this.toolsIndex.keys());
	}

	private updateState(state: {
		isInitializing?: boolean;
		error?: string | null;
		toolCount?: number;
		connectedServers?: string[];
	}): void {
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
	}

	updateHealthCheck(serverId: string, state: HealthCheckState): void {
		this._healthChecks = { ...this._healthChecks, [serverId]: state };
	}

	getHealthCheckState(serverId: string): HealthCheckState {
		return this._healthChecks[serverId] ?? { status: 'idle' };
	}

	hasHealthCheck(serverId: string): boolean {
		return serverId in this._healthChecks && this._healthChecks[serverId].status !== 'idle';
	}

	clearHealthCheck(serverId: string): void {
		// eslint-disable-next-line @typescript-eslint/no-unused-vars
		const { [serverId]: _removed, ...rest } = this._healthChecks;
		this._healthChecks = rest;
	}

	clearAllHealthChecks(): void {
		this._healthChecks = {};
	}

	clearError(): void {
		this._error = null;
	}

	getServers(): MCPServerSettingsEntry[] {
		return parseMcpServerSettings(config().mcpServers);
	}

	/**
	 * Get all active MCP connections.
	 * @returns Map of server names to connections
	 */
	getConnections(): Map<string, MCPConnection> {
		return this.connections;
	}

	getServerLabel(server: MCPServerSettingsEntry): string {
		const healthState = this.getHealthCheckState(server.id);
		if (healthState?.status === HealthCheckStatus.SUCCESS)
			return (
				healthState.serverInfo?.title || healthState.serverInfo?.name || server.name || server.url
			);
		return server.url;
	}

	getServerById(serverId: string): MCPServerSettingsEntry | undefined {
		return this.getServers().find((s) => s.id === serverId);
	}

	/**
	 * Get display name for an MCP server by its ID.
	 * Falls back to the server ID if server is not found.
	 */
	getServerDisplayName(serverId: string): string {
		const server = this.getServerById(serverId);
		return server ? this.getServerLabel(server) : serverId;
	}

	/**
	 * Get favicon URL for an MCP server by its ID.
	 * Uses Google's favicon service for consistent display.
	 * Returns null if server is not found.
	 */
	getServerFavicon(serverId: string): string | null {
		const server = this.getServerById(serverId);
		if (!server) {
			return null;
		}

		return getFaviconUrl(server.url);
	}

	isAnyServerLoading(): boolean {
		return this.getServers().some((s) => {
			const state = this.getHealthCheckState(s.id);

			return (
				state.status === HealthCheckStatus.IDLE || state.status === HealthCheckStatus.CONNECTING
			);
		});
	}

	getServersSorted(): MCPServerSettingsEntry[] {
		const servers = this.getServers();
		if (this.isAnyServerLoading()) {
			return servers;
		}

		return [...servers].sort((a, b) =>
			this.getServerLabel(a).localeCompare(this.getServerLabel(b))
		);
	}

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
			requestTimeoutSeconds: DEFAULT_MCP_CONFIG.requestTimeoutSeconds,
			useProxy: serverData.useProxy
		};
		settingsStore.updateConfig('mcpServers', JSON.stringify([...servers, newServer]));
	}

	updateServer(id: string, updates: Partial<MCPServerSettingsEntry>): void {
		const servers = this.getServers();
		settingsStore.updateConfig(
			'mcpServers',
			JSON.stringify(
				servers.map((server) => (server.id === id ? { ...server, ...updates } : server))
			)
		);
	}

	removeServer(id: string): void {
		const servers = this.getServers();
		settingsStore.updateConfig('mcpServers', JSON.stringify(servers.filter((s) => s.id !== id)));
		this.clearHealthCheck(id);
	}

	hasAvailableServers(): boolean {
		return parseMcpServerSettings(config().mcpServers).some((s) => s.enabled && s.url.trim());
	}
	hasEnabledServers(perChatOverrides?: McpServerOverride[]): boolean {
		return Boolean(buildMcpClientConfigInternal(config(), perChatOverrides));
	}

	getEnabledServersForConversation(
		perChatOverrides?: McpServerOverride[]
	): MCPServerSettingsEntry[] {
		if (!perChatOverrides?.length) {
			return [];
		}

		return this.getServers().filter((server) => {
			if (!server.enabled) {
				return false;
			}

			const override = perChatOverrides.find((o) => o.serverId === server.id);

			return override?.enabled ?? false;
		});
	}

	async ensureInitialized(perChatOverrides?: McpServerOverride[]): Promise<boolean> {
		if (!browser) {
			return false;
		}

		const mcpConfig = buildMcpClientConfigInternal(config(), perChatOverrides);
		const signature = mcpConfig ? JSON.stringify(mcpConfig) : null;
		if (!signature) {
			await this.shutdown();

			return false;
		}
		if (this.isInitialized && this.configSignature === signature) {
			return true;
		}

		if (this.initPromise && this.configSignature === signature) {
			return this.initPromise;
		}

		if (this.connections.size > 0 || this.initPromise) await this.shutdown();
		return this.initialize(signature, mcpConfig!);
	}

	private async initialize(signature: string, mcpConfig: MCPClientConfig): Promise<boolean> {
		this.updateState({ isInitializing: true, error: null });
		this.configSignature = signature;

		const serverEntries = Object.entries(mcpConfig.servers);

		if (serverEntries.length === 0) {
			this.updateState({ isInitializing: false, toolCount: 0, connectedServers: [] });

			return false;
		}
		this.initPromise = this.doInitialize(signature, mcpConfig, serverEntries);

		return this.initPromise;
	}

	private async doInitialize(
		signature: string,
		mcpConfig: MCPClientConfig,
		serverEntries: [string, MCPClientConfig['servers'][string]][]
	): Promise<boolean> {
		const clientInfo = mcpConfig.clientInfo ?? DEFAULT_MCP_CONFIG.clientInfo;
		const capabilities = mcpConfig.capabilities ?? DEFAULT_MCP_CONFIG.capabilities;
		const results = await Promise.allSettled(
			serverEntries.map(async ([name, serverConfig]) => {
				// Store config for reconnection
				this.serverConfigs.set(name, serverConfig);

				const listChangedHandlers = this.createListChangedHandlers(name);
				const connection = await MCPService.connect(
					name,
					serverConfig,
					clientInfo,
					capabilities,
					(phase) => {
						// Handle WebSocket disconnection
						if (phase === MCPConnectionPhase.DISCONNECTED) {
							console.log(`[MCPStore][${name}] Connection lost, starting auto-reconnect`);
							this.autoReconnect(name);
						}
					},
					listChangedHandlers
				);

				return { name, connection };
			})
		);
		if (this.configSignature !== signature) {
			for (const result of results) {
				if (result.status === 'fulfilled')
					await MCPService.disconnect(result.value.connection).catch(console.warn);
			}

			return false;
		}
		for (const result of results) {
			if (result.status === 'fulfilled') {
				const { name, connection } = result.value;

				this.connections.set(name, connection);

				for (const tool of connection.tools) {
					if (this.toolsIndex.has(tool.name))
						console.warn(
							`[MCPStore] Tool name conflict: "${tool.name}" exists in "${this.toolsIndex.get(tool.name)}" and "${name}". Using tool from "${name}".`
						);
					this.toolsIndex.set(tool.name, name);
				}
			} else {
				console.error(`[MCPStore] Failed to connect:`, result.reason);
			}
		}

		const successCount = this.connections.size;
		if (successCount === 0 && serverEntries.length > 0) {
			this.updateState({
				isInitializing: false,
				error: 'All MCP server connections failed',
				toolCount: 0,
				connectedServers: []
			});
			this.initPromise = null;

			return false;
		}

		this.updateState({
			isInitializing: false,
			error: null,
			toolCount: this.toolsIndex.size,
			connectedServers: Array.from(this.connections.keys())
		});
		this.initPromise = null;

		return true;
	}

	private createListChangedHandlers(serverName: string): ListChangedHandlers {
		return {
			tools: {
				onChanged: (error: Error | null, tools: Tool[] | null) => {
					if (error) {
						console.warn(`[MCPStore][${serverName}] Tools list changed error:`, error);
						return;
					}
					this.handleToolsListChanged(serverName, tools ?? []);
				}
			},
			prompts: {
				onChanged: (error: Error | null) => {
					if (error) {
						console.warn(`[MCPStore][${serverName}] Prompts list changed error:`, error);
						return;
					}
				}
			}
		};
	}

	private handleToolsListChanged(serverName: string, tools: Tool[]): void {
		const connection = this.connections.get(serverName);
		if (!connection) {
			return;
		}

		for (const [toolName, ownerServer] of this.toolsIndex.entries()) {
			if (ownerServer === serverName) this.toolsIndex.delete(toolName);
		}

		connection.tools = tools;

		for (const tool of tools) {
			if (this.toolsIndex.has(tool.name))
				console.warn(
					`[MCPStore] Tool name conflict after list change: "${tool.name}" exists in "${this.toolsIndex.get(tool.name)}" and "${serverName}". Using tool from "${serverName}".`
				);
			this.toolsIndex.set(tool.name, serverName);
		}
		this.updateState({ toolCount: this.toolsIndex.size });
	}

	acquireConnection(): void {
		this.activeFlowCount++;
	}

	/**
	 * Release a connection reference.
	 * By default, keeps connections alive for reuse (shutdownIfUnused=false).
	 * MCP spec encourages long-lived sessions to avoid reconnection overhead.
	 */
	async releaseConnection(shutdownIfUnused = false): Promise<void> {
		this.activeFlowCount = Math.max(0, this.activeFlowCount - 1);
		if (shutdownIfUnused && this.activeFlowCount === 0) {
			await this.shutdown();
		}
	}

	getActiveFlowCount(): number {
		return this.activeFlowCount;
	}

	async shutdown(): Promise<void> {
		if (this.initPromise) {
			await this.initPromise.catch(() => {});
			this.initPromise = null;
		}

		if (this.connections.size === 0) {
			return;
		}

		await Promise.all(
			Array.from(this.connections.values()).map((conn) =>
				MCPService.disconnect(conn).catch((error) =>
					console.warn(`[MCPStore] Error disconnecting ${conn.serverName}:`, error)
				)
			)
		);

		this.connections.clear();
		this.toolsIndex.clear();
		this.serverConfigs.clear();
		this.configSignature = null;
		this.updateState({ isInitializing: false, error: null, toolCount: 0, connectedServers: [] });
	}

	/**
	 * Auto-reconnect to a server with exponential backoff.
	 * Continues indefinitely until successful.
	 */
	private async autoReconnect(serverName: string): Promise<void> {
		// Guard against concurrent reconnections
		if (this.reconnectingServers.has(serverName)) {
			console.log(`[MCPStore][${serverName}] Reconnection already in progress, skipping`);

			return;
		}

		const serverConfig = this.serverConfigs.get(serverName);
		if (!serverConfig) {
			console.error(`[MCPStore] No config found for ${serverName}, cannot reconnect`);

			return;
		}

		this.reconnectingServers.add(serverName);
		let backoff = MCP_RECONNECT_INITIAL_DELAY;

		try {
			while (true) {
				await new Promise((resolve) => setTimeout(resolve, backoff));

				console.log(`[MCPStore][${serverName}] Auto-reconnecting...`);

				try {
					const listChangedHandlers = this.createListChangedHandlers(serverName);
					const connection = await MCPService.connect(
						serverName,
						serverConfig,
						DEFAULT_MCP_CONFIG.clientInfo,
						DEFAULT_MCP_CONFIG.capabilities,
						(phase) => {
							if (phase === MCPConnectionPhase.DISCONNECTED) {
								console.log(`[MCPStore][${serverName}] Connection lost, restarting auto-reconnect`);
								this.autoReconnect(serverName);
							}
						},
						listChangedHandlers
					);

					// Replace old connection with new one
					this.connections.set(serverName, connection);

					// Rebuild tool index for this server
					for (const tool of connection.tools) {
						this.toolsIndex.set(tool.name, serverName);
					}

					console.log(`[MCPStore][${serverName}] Reconnected successfully`);
					break;
				} catch (error) {
					console.warn(`[MCPStore][${serverName}] Reconnection failed:`, error);
					backoff = Math.min(backoff * MCP_RECONNECT_BACKOFF_MULTIPLIER, MCP_RECONNECT_MAX_DELAY);
				}
			}
		} finally {
			this.reconnectingServers.delete(serverName);
		}
	}

	getToolDefinitionsForLLM(): OpenAIToolDefinition[] {
		const tools: OpenAIToolDefinition[] = [];

		for (const connection of this.connections.values()) {
			for (const tool of connection.tools) {
				const rawSchema = (tool.inputSchema as Record<string, unknown>) ?? {
					type: 'object',
					properties: {},
					required: []
				};

				tools.push({
					type: 'function' as const,
					function: {
						name: tool.name,
						description: tool.description,
						parameters: this.normalizeSchemaProperties(rawSchema)
					}
				});
			}
		}

		return tools;
	}

	private normalizeSchemaProperties(schema: Record<string, unknown>): Record<string, unknown> {
		if (!schema || typeof schema !== 'object') {
			return schema;
		}

		const normalized = { ...schema };
		if (normalized.properties && typeof normalized.properties === 'object') {
			const props = normalized.properties as Record<string, Record<string, unknown>>;
			const normalizedProps: Record<string, Record<string, unknown>> = {};
			for (const [key, prop] of Object.entries(props)) {
				if (!prop || typeof prop !== 'object') {
					normalizedProps[key] = prop;
					continue;
				}
				const normalizedProp = { ...prop };
				if (!normalizedProp.type && normalizedProp.default !== undefined) {
					const defaultVal = normalizedProp.default;
					if (typeof defaultVal === 'string') normalizedProp.type = 'string';
					else if (typeof defaultVal === 'number')
						normalizedProp.type = Number.isInteger(defaultVal) ? 'integer' : 'number';
					else if (typeof defaultVal === 'boolean') normalizedProp.type = 'boolean';
					else if (Array.isArray(defaultVal)) normalizedProp.type = 'array';
					else if (typeof defaultVal === 'object' && defaultVal !== null)
						normalizedProp.type = 'object';
				}
				if (normalizedProp.properties)
					Object.assign(
						normalizedProp,
						this.normalizeSchemaProperties(normalizedProp as Record<string, unknown>)
					);
				if (normalizedProp.items && typeof normalizedProp.items === 'object')
					normalizedProp.items = this.normalizeSchemaProperties(
						normalizedProp.items as Record<string, unknown>
					);
				normalizedProps[key] = normalizedProp;
			}
			normalized.properties = normalizedProps;
		}

		return normalized;
	}

	getToolNames(): string[] {
		return Array.from(this.toolsIndex.keys());
	}

	hasTool(toolName: string): boolean {
		return this.toolsIndex.has(toolName);
	}

	getToolServer(toolName: string): string | undefined {
		return this.toolsIndex.get(toolName);
	}

	hasPromptsSupport(): boolean {
		for (const connection of this.connections.values()) {
			if (connection.serverCapabilities?.prompts) {
				return true;
			}
		}

		return false;
	}

	/**
	 * Check if any enabled server with successful health check supports prompts.
	 * Uses health check state since servers may not have active connections until
	 * the user actually sends a message or uses prompts.
	 * @param perChatOverrides - Per-chat server overrides to filter by enabled servers.
	 *                          If provided (even empty array), only checks enabled servers.
	 *                          If undefined, checks all servers with successful health checks.
	 */
	hasPromptsCapability(perChatOverrides?: McpServerOverride[]): boolean {
		// If perChatOverrides is provided (even empty array), filter by enabled servers
		if (perChatOverrides !== undefined) {
			const enabledServerIds = new Set(
				perChatOverrides.filter((o) => o.enabled).map((o) => o.serverId)
			);

			// No enabled servers = no capability
			if (enabledServerIds.size === 0) {
				return false;
			}

			// Check health check states for enabled servers with prompts capability
			for (const [serverId, state] of Object.entries(this._healthChecks)) {
				if (!enabledServerIds.has(serverId)) continue;
				if (
					state.status === HealthCheckStatus.SUCCESS &&
					state.capabilities?.server?.prompts !== undefined
				) {
					return true;
				}
			}

			// Also check active connections as fallback
			for (const [serverName, connection] of this.connections) {
				if (!enabledServerIds.has(serverName)) continue;
				if (connection.serverCapabilities?.prompts) {
					return true;
				}
			}

			return false;
		}

		// No overrides provided - check all servers (global mode)
		for (const state of Object.values(this._healthChecks)) {
			if (
				state.status === HealthCheckStatus.SUCCESS &&
				state.capabilities?.server?.prompts !== undefined
			) {
				return true;
			}
		}

		for (const connection of this.connections.values()) {
			if (connection.serverCapabilities?.prompts) {
				return true;
			}
		}

		return false;
	}

	async getAllPrompts(): Promise<MCPPromptInfo[]> {
		const results: MCPPromptInfo[] = [];

		for (const [serverName, connection] of this.connections) {
			if (!connection.serverCapabilities?.prompts) continue;

			const prompts = await MCPService.listPrompts(connection);

			for (const prompt of prompts) {
				results.push({
					name: prompt.name,
					description: prompt.description,
					title: prompt.title,
					serverName,
					arguments: prompt.arguments?.map((arg) => ({
						name: arg.name,
						description: arg.description,
						required: arg.required
					}))
				});
			}
		}

		return results;
	}

	async getPrompt(
		serverName: string,
		promptName: string,
		args?: Record<string, string>
	): Promise<GetPromptResult> {
		const connection = this.connections.get(serverName);
		if (!connection) throw new Error(`Server "${serverName}" not found for prompt "${promptName}"`);

		return MCPService.getPrompt(connection, promptName, args);
	}

	async executeTool(toolCall: MCPToolCall, signal?: AbortSignal): Promise<ToolExecutionResult> {
		const toolName = toolCall.function.name;

		const serverName = this.toolsIndex.get(toolName);
		if (!serverName) throw new Error(`Unknown tool: ${toolName}`);

		const connection = this.connections.get(serverName);
		if (!connection) throw new Error(`Server "${serverName}" is not connected`);

		const args = this.parseToolArguments(toolCall.function.arguments);

		return MCPService.callTool(connection, { name: toolName, arguments: args }, signal);
	}

	async executeToolByName(
		toolName: string,
		args: Record<string, unknown>,
		signal?: AbortSignal
	): Promise<ToolExecutionResult> {
		const serverName = this.toolsIndex.get(toolName);
		if (!serverName) throw new Error(`Unknown tool: ${toolName}`);
		const connection = this.connections.get(serverName);
		if (!connection) throw new Error(`Server "${serverName}" is not connected`);

		return MCPService.callTool(connection, { name: toolName, arguments: args }, signal);
	}

	private parseToolArguments(args: string | Record<string, unknown>): Record<string, unknown> {
		if (typeof args === 'string') {
			const trimmed = args.trim();
			if (trimmed === '') {
				return {};
			}

			try {
				const parsed = JSON.parse(trimmed);
				if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed))
					throw new Error(
						`Tool arguments must be an object, got ${Array.isArray(parsed) ? 'array' : typeof parsed}`
					);

				return parsed as Record<string, unknown>;
			} catch (error) {
				throw new Error(`Failed to parse tool arguments as JSON: ${(error as Error).message}`);
			}
		}

		if (typeof args === 'object' && args !== null && !Array.isArray(args)) {
			return args;
		}

		throw new Error(`Invalid tool arguments type: ${typeof args}`);
	}

	async getPromptCompletions(
		serverName: string,
		promptName: string,
		argumentName: string,
		argumentValue: string
	): Promise<{ values: string[]; total?: number; hasMore?: boolean } | null> {
		const connection = this.connections.get(serverName);
		if (!connection) {
			console.warn(`[MCPStore] Server "${serverName}" is not connected`);
			return null;
		}
		if (!connection.serverCapabilities?.completions) {
			return null;
		}

		return MCPService.complete(
			connection,
			{ type: MCPRefType.PROMPT, name: promptName },
			{ name: argumentName, value: argumentValue }
		);
	}

	private parseHeaders(headersJson?: string): Record<string, string> | undefined {
		if (!headersJson?.trim()) {
			return undefined;
		}

		try {
			const parsed = JSON.parse(headersJson);
			if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed))
				return parsed as Record<string, string>;
		} catch {
			console.warn('[MCPStore] Failed to parse custom headers JSON:', headersJson);
		}

		return undefined;
	}

	async runHealthChecksForServers(
		servers: {
			id: string;
			enabled: boolean;
			url: string;
			requestTimeoutSeconds: number;
			headers?: string;
		}[],
		skipIfChecked = true,
		promoteToActive = false
	): Promise<void> {
		const serversToCheck = skipIfChecked
			? servers.filter((s) => !this.hasHealthCheck(s.id) && s.url.trim())
			: servers.filter((s) => s.url.trim());

		if (serversToCheck.length === 0) {
			return;
		}

		const BATCH_SIZE = 5;
		for (let i = 0; i < serversToCheck.length; i += BATCH_SIZE) {
			const batch = serversToCheck.slice(i, i + BATCH_SIZE);
			await Promise.allSettled(batch.map((server) => this.runHealthCheck(server, promoteToActive)));
		}
	}

	/**
	 * Check if a server already has an active connection that can be reused.
	 * Returns the existing connection if available.
	 */
	getExistingConnection(serverId: string): MCPConnection | undefined {
		return this.connections.get(serverId);
	}

	/**
	 * Run a health check for a server.
	 * If the server already has an active connection, reuses it instead of creating a new one.
	 * If promoteToActive is true and server is enabled, the connection will be kept
	 * and promoted to an active connection instead of being disconnected.
	 */
	async runHealthCheck(server: HealthCheckParams, promoteToActive = false): Promise<void> {
		// Check if we already have an active connection for this server
		const existingConnection = this.connections.get(server.id);
		if (existingConnection) {
			// Reuse existing connection - just refresh tools list
			try {
				const tools = await MCPService.listTools(existingConnection);
				const capabilities = buildCapabilitiesInfo(
					existingConnection.serverCapabilities,
					existingConnection.clientCapabilities
				);
				this.updateHealthCheck(server.id, {
					status: HealthCheckStatus.SUCCESS,
					tools: tools.map((tool) => ({
						name: tool.name,
						description: tool.description,
						title: tool.title
					})),
					serverInfo: existingConnection.serverInfo,
					capabilities,
					transportType: existingConnection.transportType,
					protocolVersion: existingConnection.protocolVersion,
					instructions: existingConnection.instructions,
					connectionTimeMs: existingConnection.connectionTimeMs,
					logs: []
				});
				return;
			} catch (error) {
				console.warn(
					`[MCPStore] Failed to reuse connection for ${server.id}, creating new one:`,
					error
				);
				// Connection may be stale, remove it and create new one
				this.connections.delete(server.id);
			}
		}

		const trimmedUrl = server.url.trim();
		const logs: MCPConnectionLog[] = [];
		let currentPhase: MCPConnectionPhase = MCPConnectionPhase.IDLE;

		if (!trimmedUrl) {
			this.updateHealthCheck(server.id, {
				status: HealthCheckStatus.ERROR,
				message: 'Please enter a server URL first.',
				logs: []
			});
			return;
		}

		this.updateHealthCheck(server.id, {
			status: HealthCheckStatus.CONNECTING,
			phase: MCPConnectionPhase.TRANSPORT_CREATING,
			logs: []
		});

		const timeoutMs = Math.round(server.requestTimeoutSeconds * 1000);
		const headers = this.parseHeaders(server.headers);

		try {
			const serverConfig: MCPServerConfig = {
				url: trimmedUrl,
				transport: detectMcpTransportFromUrl(trimmedUrl),
				handshakeTimeoutMs: DEFAULT_MCP_CONFIG.connectionTimeoutMs,
				requestTimeoutMs: timeoutMs,
				headers,
				useProxy: server.useProxy
			};

			// Store config for reconnection
			this.serverConfigs.set(server.id, serverConfig);

			const connection = await MCPService.connect(
				server.id,
				serverConfig,
				DEFAULT_MCP_CONFIG.clientInfo,
				DEFAULT_MCP_CONFIG.capabilities,
				(phase, log) => {
					currentPhase = phase;
					logs.push(log);
					this.updateHealthCheck(server.id, {
						status: HealthCheckStatus.CONNECTING,
						phase,
						logs: [...logs]
					});

					// Handle WebSocket disconnection
					if (phase === MCPConnectionPhase.DISCONNECTED && promoteToActive) {
						console.log(
							`[MCPStore][${server.id}] Connection lost during health check, starting auto-reconnect`
						);
						this.autoReconnect(server.id);
					}
				}
			);

			const tools = connection.tools.map((tool) => ({
				name: tool.name,
				description: tool.description,
				title: tool.title
			}));

			const capabilities = buildCapabilitiesInfo(
				connection.serverCapabilities,
				connection.clientCapabilities
			);

			this.updateHealthCheck(server.id, {
				status: HealthCheckStatus.SUCCESS,
				tools,
				serverInfo: connection.serverInfo,
				capabilities,
				transportType: connection.transportType,
				protocolVersion: connection.protocolVersion,
				instructions: connection.instructions,
				connectionTimeMs: connection.connectionTimeMs,
				logs
			});

			// Promote to active connection or disconnect
			if (promoteToActive && server.enabled) {
				this.promoteHealthCheckToConnection(server.id, connection);
			} else {
				await MCPService.disconnect(connection);
			}
		} catch (error) {
			const message = error instanceof Error ? error.message : 'Unknown error occurred';

			logs.push({
				timestamp: new Date(),
				phase: MCPConnectionPhase.ERROR,
				message: `Connection failed: ${message}`,
				level: MCPLogLevel.ERROR
			});

			this.updateHealthCheck(server.id, {
				status: HealthCheckStatus.ERROR,
				message,
				phase: currentPhase,
				logs
			});
		}
	}

	/**
	 * Promote a health check connection to an active connection.
	 * This avoids the need to reconnect when the server is needed for agentic flows.
	 */
	private promoteHealthCheckToConnection(serverId: string, connection: MCPConnection): void {
		// Register tools from the connection
		for (const tool of connection.tools) {
			if (this.toolsIndex.has(tool.name)) {
				console.warn(
					`[MCPStore] Tool name conflict during promotion: "${tool.name}" exists in "${this.toolsIndex.get(tool.name)}" and "${serverId}". Using tool from "${serverId}".`
				);
			}
			this.toolsIndex.set(tool.name, serverId);
		}

		// Add to active connections
		this.connections.set(serverId, connection);

		// Update state
		this.updateState({
			toolCount: this.toolsIndex.size,
			connectedServers: Array.from(this.connections.keys())
		});
	}

	getServersStatus(): ServerStatus[] {
		const statuses: ServerStatus[] = [];

		for (const [name, connection] of this.connections) {
			statuses.push({
				name,
				isConnected: true,
				toolCount: connection.tools.length,
				error: undefined
			});
		}

		return statuses;
	}

	/**
	 * Get aggregated server instructions from all connected servers.
	 * Returns an array of { serverName, serverTitle, instructions } objects.
	 */
	getServerInstructions(): Array<{
		serverName: string;
		serverTitle?: string;
		instructions: string;
	}> {
		const results: Array<{ serverName: string; serverTitle?: string; instructions: string }> = [];

		for (const [serverName, connection] of this.connections) {
			if (connection.instructions) {
				results.push({
					serverName,
					serverTitle: connection.serverInfo?.title || connection.serverInfo?.name,
					instructions: connection.instructions
				});
			}
		}

		return results;
	}

	/**
	 * Get server instructions from health check results (for display before active connection).
	 * Useful for showing instructions in settings UI.
	 */
	getHealthCheckInstructions(): Array<{
		serverId: string;
		serverTitle?: string;
		instructions: string;
	}> {
		const results: Array<{ serverId: string; serverTitle?: string; instructions: string }> = [];

		for (const [serverId, state] of Object.entries(this._healthChecks)) {
			if (state.status === HealthCheckStatus.SUCCESS && state.instructions) {
				results.push({
					serverId,
					serverTitle: state.serverInfo?.title || state.serverInfo?.name,
					instructions: state.instructions
				});
			}
		}

		return results;
	}

	/**
	 * Check if any connected server has instructions.
	 */
	hasServerInstructions(): boolean {
		for (const connection of this.connections.values()) {
			if (connection.instructions) {
				return true;
			}
		}

		return false;
	}

	/**
	 *
	 *
	 * Resources Operations
	 *
	 *
	 */

	/**
	 * Check if any enabled server with successful health check supports resources.
	 * Uses health check state since servers may not have active connections until
	 * the user actually sends a message or uses prompts.
	 * @param perChatOverrides - Per-chat server overrides to filter by enabled servers.
	 *                          If provided (even empty array), only checks enabled servers.
	 *                          If undefined, checks all servers with successful health checks.
	 */
	hasResourcesCapability(perChatOverrides?: McpServerOverride[]): boolean {
		// If perChatOverrides is provided (even empty array), filter by enabled servers
		if (perChatOverrides !== undefined) {
			const enabledServerIds = new Set(
				perChatOverrides.filter((o) => o.enabled).map((o) => o.serverId)
			);
			// No enabled servers = no capability
			if (enabledServerIds.size === 0) {
				return false;
			}

			// Check health check states for enabled servers with resources capability
			for (const [serverId, state] of Object.entries(this._healthChecks)) {
				if (!enabledServerIds.has(serverId)) continue;
				if (
					state.status === HealthCheckStatus.SUCCESS &&
					state.capabilities?.server?.resources !== undefined
				) {
					return true;
				}
			}

			// Also check active connections as fallback
			for (const [serverName, connection] of this.connections) {
				if (!enabledServerIds.has(serverName)) continue;
				if (MCPService.supportsResources(connection)) {
					return true;
				}
			}

			return false;
		}

		// No overrides provided - check all servers (global mode)
		for (const state of Object.values(this._healthChecks)) {
			if (
				state.status === HealthCheckStatus.SUCCESS &&
				state.capabilities?.server?.resources !== undefined
			) {
				return true;
			}
		}

		for (const connection of this.connections.values()) {
			if (MCPService.supportsResources(connection)) {
				return true;
			}
		}

		return false;
	}

	/**
	 * Get list of servers that support resources.
	 * Checks active connections first, then health check state as fallback.
	 */
	getServersWithResources(): string[] {
		const servers: string[] = [];

		// Check active connections
		for (const [name, connection] of this.connections) {
			if (MCPService.supportsResources(connection) && !servers.includes(name)) {
				servers.push(name);
			}
		}

		// Also check health check states for servers not yet connected
		for (const [serverId, state] of Object.entries(this._healthChecks)) {
			if (
				!servers.includes(serverId) &&
				state.status === HealthCheckStatus.SUCCESS &&
				state.capabilities?.server?.resources !== undefined
			) {
				servers.push(serverId);
			}
		}

		return servers;
	}

	/**
	 * Fetch resources from all connected servers that support them.
	 * Updates mcpResourceStore with the results.
	 * @param forceRefresh - If true, bypass cache and fetch fresh data
	 */
	async fetchAllResources(forceRefresh: boolean = false): Promise<void> {
		const serversWithResources = this.getServersWithResources();
		if (serversWithResources.length === 0) {
			return;
		}

		// Check if we have cached resources and they're recent (unless force refresh)
		if (!forceRefresh) {
			const allServersCached = serversWithResources.every((serverName) => {
				const serverRes = mcpResourceStore.getServerResources(serverName);
				if (!serverRes || !serverRes.lastFetched) {
					return false;
				}

				// Cache is valid for 5 minutes
				const age = Date.now() - serverRes.lastFetched.getTime();

				return age < 5 * 60 * 1000;
			});

			if (allServersCached) {
				console.log('[MCPStore] Using cached resources');

				return;
			}
		}

		mcpResourceStore.setLoading(true);

		try {
			await Promise.all(
				serversWithResources.map((serverName) => this.fetchServerResources(serverName))
			);
		} finally {
			mcpResourceStore.setLoading(false);
		}
	}

	/**
	 * Fetch resources from a specific server.
	 * Updates mcpResourceStore with the results.
	 */
	async fetchServerResources(serverName: string): Promise<void> {
		const connection = this.connections.get(serverName);
		if (!connection) {
			console.warn(`[MCPStore] No connection found for server: ${serverName}`);
			return;
		}

		if (!MCPService.supportsResources(connection)) {
			return;
		}

		mcpResourceStore.setServerLoading(serverName, true);

		try {
			const [resources, templates] = await Promise.all([
				MCPService.listAllResources(connection),
				MCPService.listAllResourceTemplates(connection)
			]);

			mcpResourceStore.setServerResources(serverName, resources, templates);
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			mcpResourceStore.setServerError(serverName, message);
			console.error(`[MCPStore][${serverName}] Failed to fetch resources:`, error);
		}
	}

	/**
	 * Read resource content from a server.
	 * Caches the result in mcpResourceStore.
	 */
	async readResource(uri: string): Promise<import('$lib/types').MCPResourceContent[] | null> {
		// Check cache first
		const cached = mcpResourceStore.getCachedContent(uri);
		if (cached) {
			return cached.content;
		}

		// Find which server has this resource
		const serverName = mcpResourceStore.findServerForUri(uri);
		if (!serverName) {
			console.error(`[MCPStore] No server found for resource URI: ${uri}`);

			return null;
		}

		const connection = this.connections.get(serverName);
		if (!connection) {
			console.error(`[MCPStore] No connection found for server: ${serverName}`);

			return null;
		}

		try {
			const result = await MCPService.readResource(connection, uri);
			const resourceInfo = mcpResourceStore.findResourceByUri(uri);

			if (resourceInfo) {
				mcpResourceStore.cacheResourceContent(resourceInfo, result.contents);
			}

			return result.contents;
		} catch (error) {
			console.error(`[MCPStore] Failed to read resource ${uri}:`, error);

			return null;
		}
	}

	/**
	 * Subscribe to resource updates.
	 */
	async subscribeToResource(uri: string): Promise<boolean> {
		const serverName = mcpResourceStore.findServerForUri(uri);
		if (!serverName) {
			console.error(`[MCPStore] No server found for resource URI: ${uri}`);

			return false;
		}

		const connection = this.connections.get(serverName);
		if (!connection) {
			console.error(`[MCPStore] No connection found for server: ${serverName}`);

			return false;
		}

		if (!MCPService.supportsResourceSubscriptions(connection)) {
			return false;
		}

		try {
			await MCPService.subscribeResource(connection, uri);
			mcpResourceStore.addSubscription(uri, serverName);

			return true;
		} catch (error) {
			console.error(`[MCPStore] Failed to subscribe to resource ${uri}:`, error);

			return false;
		}
	}

	/**
	 * Unsubscribe from resource updates.
	 */
	async unsubscribeFromResource(uri: string): Promise<boolean> {
		const serverName = mcpResourceStore.findServerForUri(uri);
		if (!serverName) {
			console.error(`[MCPStore] No server found for resource URI: ${uri}`);

			return false;
		}

		const connection = this.connections.get(serverName);
		if (!connection) {
			console.error(`[MCPStore] No connection found for server: ${serverName}`);

			return false;
		}

		try {
			await MCPService.unsubscribeResource(connection, uri);
			mcpResourceStore.removeSubscription(uri);

			return true;
		} catch (error) {
			console.error(`[MCPStore] Failed to unsubscribe from resource ${uri}:`, error);

			return false;
		}
	}

	/**
	 * Add a resource as attachment to chat context.
	 * Automatically fetches content if not cached.
	 */
	async attachResource(uri: string): Promise<import('$lib/types').MCPResourceAttachment | null> {
		const resourceInfo = mcpResourceStore.findResourceByUri(uri);
		if (!resourceInfo) {
			console.error(`[MCPStore] Resource not found: ${uri}`);

			return null;
		}

		// Check if already attached
		if (mcpResourceStore.isAttached(uri)) {
			return null;
		}

		// Add attachment (initially loading)
		const attachment = mcpResourceStore.addAttachment(resourceInfo);

		// Fetch content
		try {
			const content = await this.readResource(uri);

			if (content) {
				mcpResourceStore.updateAttachmentContent(attachment.id, content);
			} else {
				mcpResourceStore.updateAttachmentError(attachment.id, 'Failed to read resource');
			}
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			mcpResourceStore.updateAttachmentError(attachment.id, message);
		}

		return mcpResourceStore.getAttachment(attachment.id) ?? null;
	}

	/**
	 * Remove a resource attachment from chat context.
	 */
	removeResourceAttachment(attachmentId: string): void {
		mcpResourceStore.removeAttachment(attachmentId);
	}

	/**
	 * Clear all resource attachments.
	 */
	clearResourceAttachments(): void {
		mcpResourceStore.clearAttachments();
	}

	/**
	 * Get formatted resource context for chat.
	 */
	getResourceContextForChat(): string {
		return mcpResourceStore.formatAttachmentsForContext();
	}

	/**
	 * Convert current resource attachments to DatabaseMessageExtra[] and clear them.
	 * Called during message send to persist resources with the user message.
	 */
	consumeResourceAttachmentsAsExtras(): import('$lib/types').DatabaseMessageExtraMcpResource[] {
		const extras = mcpResourceStore.toMessageExtras();
		if (extras.length > 0) {
			mcpResourceStore.clearAttachments();
		}
		return extras;
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
export const mcpServerInstructions = () => mcpStore.getServerInstructions();
export const mcpHasServerInstructions = () => mcpStore.hasServerInstructions();

// Resources exports
export const mcpHasResourcesCapability = () => mcpStore.hasResourcesCapability();
export const mcpServersWithResources = () => mcpStore.getServersWithResources();
export const mcpResourceContext = () => mcpStore.getResourceContextForChat();
