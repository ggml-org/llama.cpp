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
import { parseMcpServerSettings, detectMcpTransportFromUrl } from '$lib/utils';
import { MCPConnectionPhase, MCPLogLevel, HealthCheckStatus } from '$lib/enums';
import { DEFAULT_MCP_CONFIG, MCP_SERVER_ID_PREFIX } from '$lib/constants/mcp';
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
	Prompt,
	HealthCheckState,
	MCPServerSettingsEntry,
	MCPServerConfig
} from '$lib/types';
import type { ListChangedHandlers } from '@modelcontextprotocol/sdk/types.js';
import type { McpServerOverride } from '$lib/types/database';
import type { SettingsConfigType } from '$lib/types/settings';

function generateMcpServerId(id: unknown, index: number): string {
	if (typeof id === 'string' && id.trim()) return id.trim();
	return `${MCP_SERVER_ID_PREFIX}${index + 1}`;
}

function parseServerSettings(rawServers: unknown): MCPServerSettingsEntry[] {
	if (!rawServers) return [];
	let parsed: unknown;
	if (typeof rawServers === 'string') {
		const trimmed = rawServers.trim();
		if (!trimmed) return [];
		try {
			parsed = JSON.parse(trimmed);
		} catch (error) {
			console.warn('[MCP] Failed to parse mcpServers JSON:', error);
			return [];
		}
	} else {
		parsed = rawServers;
	}
	if (!Array.isArray(parsed)) return [];
	return parsed.map((entry, index) => {
		const url = typeof entry?.url === 'string' ? entry.url.trim() : '';
		const headers = typeof entry?.headers === 'string' ? entry.headers.trim() : undefined;
		return {
			id: generateMcpServerId((entry as { id?: unknown })?.id, index),
			enabled: Boolean((entry as { enabled?: unknown })?.enabled),
			url,
			name: (entry as { name?: string })?.name,
			requestTimeoutSeconds: DEFAULT_MCP_CONFIG.requestTimeoutSeconds,
			headers: headers || undefined
		} satisfies MCPServerSettingsEntry;
	});
}

function buildServerConfig(
	entry: MCPServerSettingsEntry,
	connectionTimeoutMs = DEFAULT_MCP_CONFIG.connectionTimeoutMs
): MCPServerConfig | undefined {
	if (!entry?.url) return undefined;
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
		headers
	};
}

function checkServerEnabled(
	server: MCPServerSettingsEntry,
	perChatOverrides?: McpServerOverride[]
): boolean {
	if (!server.enabled) return false;
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
	if (!rawServers.length) return undefined;
	const servers: Record<string, MCPServerConfig> = {};
	for (const [index, entry] of rawServers.entries()) {
		if (!checkServerEnabled(entry, perChatOverrides)) continue;
		const normalized = buildServerConfig(entry);
		if (normalized) servers[generateMcpServerId(entry.id, index)] = normalized;
	}
	if (Object.keys(servers).length === 0) return undefined;
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
		if (state.isInitializing !== undefined) this._isInitializing = state.isInitializing;
		if (state.error !== undefined) this._error = state.error;
		if (state.toolCount !== undefined) this._toolCount = state.toolCount;
		if (state.connectedServers !== undefined) this._connectedServers = state.connectedServers;
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

	getServerLabel(server: MCPServerSettingsEntry): string {
		const healthState = this.getHealthCheckState(server.id);
		if (healthState?.status === HealthCheckStatus.SUCCESS)
			return (
				healthState.serverInfo?.title || healthState.serverInfo?.name || server.name || server.url
			);
		return server.url;
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
		if (this.isAnyServerLoading()) return servers;
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
			requestTimeoutSeconds: DEFAULT_MCP_CONFIG.requestTimeoutSeconds
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
		if (!perChatOverrides?.length) return [];
		return this.getServers().filter((server) => {
			if (!server.enabled) return false;
			const override = perChatOverrides.find((o) => o.serverId === server.id);
			return override?.enabled ?? false;
		});
	}

	async ensureInitialized(perChatOverrides?: McpServerOverride[]): Promise<boolean> {
		if (!browser) return false;
		const mcpConfig = buildMcpClientConfigInternal(config(), perChatOverrides);
		const signature = mcpConfig ? JSON.stringify(mcpConfig) : null;
		if (!signature) {
			await this.shutdown();
			return false;
		}
		if (this.isInitialized && this.configSignature === signature) return true;
		if (this.initPromise && this.configSignature === signature) return this.initPromise;
		if (this.connections.size > 0 || this.initPromise) await this.shutdown();
		return this.initialize(signature, mcpConfig!);
	}

	private async initialize(signature: string, mcpConfig: MCPClientConfig): Promise<boolean> {
		console.log('[MCPStore] Starting initialization...');
		this.updateState({ isInitializing: true, error: null });
		this.configSignature = signature;
		const serverEntries = Object.entries(mcpConfig.servers);
		if (serverEntries.length === 0) {
			console.log('[MCPStore] No servers configured');
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
				const listChangedHandlers = this.createListChangedHandlers(name);
				const connection = await MCPService.connect(
					name,
					serverConfig,
					clientInfo,
					capabilities,
					undefined,
					listChangedHandlers
				);
				return { name, connection };
			})
		);
		if (this.configSignature !== signature) {
			console.log('[MCPStore] Config changed during init, aborting');
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
		console.log(
			`[MCPStore] Initialization complete: ${successCount}/${serverEntries.length} servers, ${this.toolsIndex.size} tools`
		);
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
					console.log(`[MCPStore][${serverName}] Tools list changed, ${tools?.length ?? 0} tools`);
					this.handleToolsListChanged(serverName, tools ?? []);
				}
			},
			prompts: {
				onChanged: (error: Error | null, prompts: Prompt[] | null) => {
					if (error) {
						console.warn(`[MCPStore][${serverName}] Prompts list changed error:`, error);
						return;
					}
					console.log(
						`[MCPStore][${serverName}] Prompts list changed, ${prompts?.length ?? 0} prompts`
					);
				}
			}
		};
	}

	private handleToolsListChanged(serverName: string, tools: Tool[]): void {
		const connection = this.connections.get(serverName);
		if (!connection) return;
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
		console.log(`[MCPStore] Connection acquired (active flows: ${this.activeFlowCount})`);
	}

	async releaseConnection(shutdownIfUnused = true): Promise<void> {
		this.activeFlowCount = Math.max(0, this.activeFlowCount - 1);
		console.log(`[MCPStore] Connection released (active flows: ${this.activeFlowCount})`);
		if (shutdownIfUnused && this.activeFlowCount === 0) {
			console.log('[MCPStore] No active flows, initiating lazy disconnect...');
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
		if (this.connections.size === 0) return;
		console.log(`[MCPStore] Shutting down ${this.connections.size} connections...`);
		await Promise.all(
			Array.from(this.connections.values()).map((conn) =>
				MCPService.disconnect(conn).catch((error) =>
					console.warn(`[MCPStore] Error disconnecting ${conn.serverName}:`, error)
				)
			)
		);
		this.connections.clear();
		this.toolsIndex.clear();
		this.configSignature = null;
		this.updateState({ isInitializing: false, error: null, toolCount: 0, connectedServers: [] });
		console.log('[MCPStore] Shutdown complete');
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
		if (!schema || typeof schema !== 'object') return schema;
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
			if (connection.serverCapabilities?.prompts) return true;
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
			if (trimmed === '') return {};
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
		if (typeof args === 'object' && args !== null && !Array.isArray(args)) return args;
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
		if (!connection.serverCapabilities?.completions) return null;
		return MCPService.complete(
			connection,
			{ type: 'ref/prompt', name: promptName },
			{ name: argumentName, value: argumentValue }
		);
	}

	private parseHeaders(headersJson?: string): Record<string, string> | undefined {
		if (!headersJson?.trim()) return undefined;
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
		skipIfChecked = true
	): Promise<void> {
		const serversToCheck = skipIfChecked
			? servers.filter((s) => !this.hasHealthCheck(s.id) && s.url.trim())
			: servers.filter((s) => s.url.trim());
		if (serversToCheck.length === 0) return;
		const BATCH_SIZE = 5;
		for (let i = 0; i < serversToCheck.length; i += BATCH_SIZE) {
			const batch = serversToCheck.slice(i, i + BATCH_SIZE);
			await Promise.all(batch.map((server) => this.runHealthCheck(server)));
		}
	}

	async runHealthCheck(server: HealthCheckParams): Promise<void> {
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
			const connection = await MCPService.connect(
				server.id,
				{
					url: trimmedUrl,
					transport: detectMcpTransportFromUrl(trimmedUrl),
					handshakeTimeoutMs: DEFAULT_MCP_CONFIG.connectionTimeoutMs,
					requestTimeoutMs: timeoutMs,
					headers
				},
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
			await MCPService.disconnect(connection);
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
