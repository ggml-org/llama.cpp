/**
 * MCPClient - Business Logic Facade for MCP Operations
 *
 * Implements the "Host" role in MCP architecture, coordinating multiple server
 * connections and providing a unified interface for tool operations.
 *
 * **Architecture & Relationships:**
 * - **MCPClient** (this class): Business logic facade
 *   - Uses MCPService for low-level protocol operations
 *   - Updates mcpStore with reactive state
 *   - Coordinates multiple server connections
 *   - Aggregates tools from all connected servers
 *   - Routes tool calls to the appropriate server
 *
 * - **MCPService**: Stateless protocol layer (transport, connect, callTool)
 * - **mcpStore**: Reactive state only ($state, getters, setters)
 *
 * **Key Responsibilities:**
 * - Lifecycle management (initialize, shutdown)
 * - Multi-server coordination
 * - Tool name conflict detection and resolution
 * - OpenAI-compatible tool definition generation
 * - Automatic tool-to-server routing
 * - Health checks
 * - Usage statistics tracking
 */

import { mcpStore } from '$lib/stores/mcp.svelte';
import { browser } from '$app/environment';
import { MCPService } from '$lib/services/mcp.service';
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
	Prompt
} from '$lib/types';
import type { ListChangedHandlers } from '@modelcontextprotocol/sdk/types.js';
import { MCPConnectionPhase, MCPLogLevel, HealthCheckStatus } from '$lib/enums';
import type { McpServerOverride } from '$lib/types/database';
import { detectMcpTransportFromUrl } from '$lib/utils';
import { config } from '$lib/stores/settings.svelte';
import { DEFAULT_MCP_CONFIG, MCP_SERVER_ID_PREFIX } from '$lib/constants/mcp';
import type { MCPServerConfig, MCPServerSettingsEntry } from '$lib/types';
import type { SettingsConfigType } from '$lib/types/settings';

/**
 * Generates a valid MCP server ID from user input.
 * Returns the trimmed ID if valid, otherwise generates 'server-{index+1}'.
 */
function generateMcpServerId(id: unknown, index: number): string {
	if (typeof id === 'string' && id.trim()) {
		return id.trim();
	}

	return `${MCP_SERVER_ID_PREFIX}${index + 1}`;
}

/**
 * Parses MCP server settings from a JSON string or array.
 * requestTimeoutSeconds is not user-configurable in the UI, so we always use the default value.
 * @param rawServers - The raw servers to parse
 * @returns An empty array if the input is invalid.
 */
function parseMcpServerSettings(rawServers: unknown): MCPServerSettingsEntry[] {
	if (!rawServers) return [];

	let parsed: unknown;
	if (typeof rawServers === 'string') {
		const trimmed = rawServers.trim();
		if (!trimmed) return [];

		try {
			parsed = JSON.parse(trimmed);
		} catch (error) {
			console.warn('[MCP] Failed to parse mcpServers JSON, ignoring value:', error);
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
			requestTimeoutSeconds: DEFAULT_MCP_CONFIG.requestTimeoutSeconds,
			headers: headers || undefined
		} satisfies MCPServerSettingsEntry;
	});
}

/**
 * Builds an MCP server configuration from a server settings entry.
 * @param entry - The server settings entry to build the configuration from
 * @param connectionTimeoutMs - The connection timeout in milliseconds
 * @returns The built server configuration, or undefined if the entry is invalid
 */
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
			if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
				headers = parsed as Record<string, string>;
			}
		} catch {
			console.warn('[MCP] Failed to parse custom headers JSON, ignoring:', entry.headers);
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

/**
 * Checks if a server is enabled for the current chat.
 * Server must be available (server.enabled) AND have a per-chat override enabling it.
 * Pure helper function - no side effects.
 */
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

/**
 * Builds MCP client configuration from settings.
 * Returns undefined if no valid servers are configured.
 * @param config - Global settings configuration
 * @param perChatOverrides - Optional per-chat server overrides
 */
export function buildMcpClientConfig(
	config: SettingsConfigType,
	perChatOverrides?: McpServerOverride[]
): MCPClientConfig | undefined {
	const rawServers = parseMcpServerSettings(config.mcpServers);

	if (!rawServers.length) {
		return undefined;
	}

	const servers: Record<string, MCPServerConfig> = {};
	for (const [index, entry] of rawServers.entries()) {
		if (!checkServerEnabled(entry, perChatOverrides)) continue;

		const normalized = buildServerConfig(entry);
		if (normalized) {
			servers[generateMcpServerId(entry.id, index)] = normalized;
		}
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

/**
 * Build capabilities info from server and client capabilities
 */
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
				? {
						form: !!clientCaps.elicitation.form,
						url: !!clientCaps.elicitation.url
					}
				: undefined,
			tasks: !!clientCaps?.tasks
		}
	};
}

export class MCPClient {
	private connections = new Map<string, MCPConnection>();
	private toolsIndex = new Map<string, string>();
	private configSignature: string | null = null;
	private initPromise: Promise<boolean> | null = null;

	/**
	 * Reference counter for active agentic flows using MCP connections.
	 * Prevents shutdown while any conversation is still using connections.
	 */
	private activeFlowCount = 0;

	/**
	 * Ensures MCP is initialized with current config.
	 * Handles config changes by reinitializing as needed.
	 * @param perChatOverrides - Optional per-chat MCP server overrides
	 */
	async ensureInitialized(perChatOverrides?: McpServerOverride[]): Promise<boolean> {
		if (!browser) return false;

		const mcpConfig = buildMcpClientConfig(config(), perChatOverrides);
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

		if (this.connections.size > 0 || this.initPromise) {
			await this.shutdown();
		}

		return this.initialize(signature, mcpConfig!);
	}

	/**
	 * Initialize connections to all configured MCP servers.
	 */
	private async initialize(signature: string, mcpConfig: MCPClientConfig): Promise<boolean> {
		console.log('[MCPClient] Starting initialization...');

		mcpStore.updateState({ isInitializing: true, error: null });
		this.configSignature = signature;

		const serverEntries = Object.entries(mcpConfig.servers);
		if (serverEntries.length === 0) {
			console.log('[MCPClient] No servers configured');
			mcpStore.updateState({ isInitializing: false, toolCount: 0, connectedServers: [] });
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
			console.log('[MCPClient] Config changed during init, aborting');
			for (const result of results) {
				if (result.status === 'fulfilled') {
					await MCPService.disconnect(result.value.connection).catch(console.warn);
				}
			}

			return false;
		}

		for (const result of results) {
			if (result.status === 'fulfilled') {
				const { name, connection } = result.value;
				this.connections.set(name, connection);

				for (const tool of connection.tools) {
					if (this.toolsIndex.has(tool.name)) {
						console.warn(
							`[MCPClient] Tool name conflict: "${tool.name}" exists in ` +
								`"${this.toolsIndex.get(tool.name)}" and "${name}". ` +
								`Using tool from "${name}".`
						);
					}
					this.toolsIndex.set(tool.name, name);
				}
			} else {
				console.error(`[MCPClient] Failed to connect:`, result.reason);
			}
		}

		const successCount = this.connections.size;
		const totalCount = serverEntries.length;

		if (successCount === 0 && totalCount > 0) {
			const error = 'All MCP server connections failed';
			mcpStore.updateState({
				isInitializing: false,
				error,
				toolCount: 0,
				connectedServers: []
			});
			this.initPromise = null;
			return false;
		}

		mcpStore.updateState({
			isInitializing: false,
			error: null,
			toolCount: this.toolsIndex.size,
			connectedServers: Array.from(this.connections.keys())
		});

		console.log(
			`[MCPClient] Initialization complete: ${successCount}/${totalCount} servers connected, ` +
				`${this.toolsIndex.size} tools available`
		);

		this.initPromise = null;
		return true;
	}

	/**
	 * Create list changed handlers for a server connection.
	 * These handlers are called when the server notifies about changes to tools, prompts, or resources.
	 */
	private createListChangedHandlers(serverName: string): ListChangedHandlers {
		return {
			tools: {
				onChanged: (error: Error | null, tools: Tool[] | null) => {
					if (error) {
						console.warn(`[MCPClient][${serverName}] Tools list changed error:`, error);
						return;
					}
					console.log(`[MCPClient][${serverName}] Tools list changed, ${tools?.length ?? 0} tools`);
					this.handleToolsListChanged(serverName, tools ?? []);
				}
			},
			prompts: {
				onChanged: (error: Error | null, prompts: Prompt[] | null) => {
					if (error) {
						console.warn(`[MCPClient][${serverName}] Prompts list changed error:`, error);
						return;
					}
					console.log(
						`[MCPClient][${serverName}] Prompts list changed, ${prompts?.length ?? 0} prompts`
					);
					this.handlePromptsListChanged(serverName);
				}
			}
		};
	}

	/**
	 * Handle tools list changed notification from a server.
	 * Updates the tools index and store.
	 */
	private handleToolsListChanged(serverName: string, tools: Tool[]): void {
		const connection = this.connections.get(serverName);
		if (!connection) return;

		// Remove old tools from this server from the index
		for (const [toolName, ownerServer] of this.toolsIndex.entries()) {
			if (ownerServer === serverName) {
				this.toolsIndex.delete(toolName);
			}
		}

		// Update connection tools
		connection.tools = tools;

		// Add new tools to the index
		for (const tool of tools) {
			if (this.toolsIndex.has(tool.name)) {
				console.warn(
					`[MCPClient] Tool name conflict after list change: "${tool.name}" exists in ` +
						`"${this.toolsIndex.get(tool.name)}" and "${serverName}". ` +
						`Using tool from "${serverName}".`
				);
			}
			this.toolsIndex.set(tool.name, serverName);
		}

		// Update store
		mcpStore.updateState({
			toolCount: this.toolsIndex.size
		});
	}

	/**
	 * Handle prompts list changed notification from a server.
	 * Triggers a refresh of the prompts cache if needed.
	 */
	private handlePromptsListChanged(serverName: string): void {
		// Prompts are fetched on-demand, so we just log the change
		// The UI will get fresh prompts on next getAllPrompts() call
		console.log(
			`[MCPClient][${serverName}] Prompts list updated - will be refreshed on next fetch`
		);
	}

	/**
	 * Acquire a reference to MCP connections for an agentic flow.
	 * Call this when starting an agentic flow to prevent premature shutdown.
	 */
	acquireConnection(): void {
		this.activeFlowCount++;
		console.log(`[MCPClient] Connection acquired (active flows: ${this.activeFlowCount})`);
	}

	/**
	 * Release a reference to MCP connections.
	 * Call this when an agentic flow completes.
	 * @param shutdownIfUnused - If true, shutdown connections when no flows are active
	 */
	async releaseConnection(shutdownIfUnused = true): Promise<void> {
		this.activeFlowCount = Math.max(0, this.activeFlowCount - 1);
		console.log(`[MCPClient] Connection released (active flows: ${this.activeFlowCount})`);

		if (shutdownIfUnused && this.activeFlowCount === 0) {
			console.log('[MCPClient] No active flows, initiating lazy disconnect...');
			await this.shutdown();
		}
	}

	/**
	 * Get the number of active agentic flows using MCP connections.
	 */
	getActiveFlowCount(): number {
		return this.activeFlowCount;
	}

	/**
	 * Shutdown all MCP connections and clear state.
	 * Note: This will force shutdown regardless of active flow count.
	 */
	async shutdown(): Promise<void> {
		if (this.initPromise) {
			await this.initPromise.catch(() => {});
			this.initPromise = null;
		}

		if (this.connections.size === 0) {
			return;
		}

		console.log(`[MCPClient] Shutting down ${this.connections.size} connections...`);

		await Promise.all(
			Array.from(this.connections.values()).map((conn) =>
				MCPService.disconnect(conn).catch((error) => {
					console.warn(`[MCPClient] Error disconnecting ${conn.serverName}:`, error);
				})
			)
		);

		this.connections.clear();
		this.toolsIndex.clear();
		this.configSignature = null;

		mcpStore.updateState({
			isInitializing: false,
			error: null,
			toolCount: 0,
			connectedServers: []
		});

		console.log('[MCPClient] Shutdown complete');
	}

	/**
	 *
	 *
	 * Tool Definitions
	 *
	 *
	 */

	/**
	 * Returns tools in OpenAI function calling format.
	 * Ready to be sent to /v1/chat/completions API.
	 */
	getToolDefinitionsForLLM(): OpenAIToolDefinition[] {
		const tools: OpenAIToolDefinition[] = [];

		for (const connection of this.connections.values()) {
			for (const tool of connection.tools) {
				const rawSchema = (tool.inputSchema as Record<string, unknown>) ?? {
					type: 'object',
					properties: {},
					required: []
				};

				const normalizedSchema = this.normalizeSchemaProperties(rawSchema);

				tools.push({
					type: 'function' as const,
					function: {
						name: tool.name,
						description: tool.description,
						parameters: normalizedSchema
					}
				});
			}
		}

		return tools;
	}

	/**
	 * Normalize JSON Schema properties to ensure all have explicit types.
	 * Infers type from default value if missing - fixes compatibility with
	 * llama.cpp which requires explicit types in tool schemas.
	 */
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

				// Infer type from default if missing
				if (!normalizedProp.type && normalizedProp.default !== undefined) {
					const defaultVal = normalizedProp.default;
					if (typeof defaultVal === 'string') {
						normalizedProp.type = 'string';
					} else if (typeof defaultVal === 'number') {
						normalizedProp.type = Number.isInteger(defaultVal) ? 'integer' : 'number';
					} else if (typeof defaultVal === 'boolean') {
						normalizedProp.type = 'boolean';
					} else if (Array.isArray(defaultVal)) {
						normalizedProp.type = 'array';
					} else if (typeof defaultVal === 'object' && defaultVal !== null) {
						normalizedProp.type = 'object';
					}
				}

				if (normalizedProp.properties) {
					Object.assign(
						normalizedProp,
						this.normalizeSchemaProperties(normalizedProp as Record<string, unknown>)
					);
				}

				if (normalizedProp.items && typeof normalizedProp.items === 'object') {
					normalizedProp.items = this.normalizeSchemaProperties(
						normalizedProp.items as Record<string, unknown>
					);
				}

				normalizedProps[key] = normalizedProp;
			}

			normalized.properties = normalizedProps;
		}

		return normalized;
	}

	/**
	 *
	 *
	 * Tool Queries
	 *
	 *
	 */

	/**
	 * Returns names of all available tools.
	 */
	getToolNames(): string[] {
		return Array.from(this.toolsIndex.keys());
	}

	/**
	 * Check if a tool exists.
	 */
	hasTool(toolName: string): boolean {
		return this.toolsIndex.has(toolName);
	}

	/**
	 * Get which server provides a specific tool.
	 */
	getToolServer(toolName: string): string | undefined {
		return this.toolsIndex.get(toolName);
	}

	/**
	 *
	 *
	 * Prompts
	 *
	 *
	 */

	/**
	 * Get all prompts from all connected servers that support prompts.
	 */
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

	/**
	 * Get a prompt by name from a specific server.
	 * Returns the prompt messages ready to be used in chat.
	 * Throws an error if the server is not found or prompt execution fails.
	 */
	async getPrompt(
		serverName: string,
		promptName: string,
		args?: Record<string, string>
	): Promise<GetPromptResult> {
		const connection = this.connections.get(serverName);

		if (!connection) {
			const errorMsg = `Server "${serverName}" not found for prompt "${promptName}"`;
			console.error(`[MCPClient] ${errorMsg}`);

			throw new Error(errorMsg);
		}

		return MCPService.getPrompt(connection, promptName, args);
	}

	/**
	 * Check if any connected server supports prompts.
	 */
	hasPromptsSupport(): boolean {
		for (const connection of this.connections.values()) {
			if (connection.serverCapabilities?.prompts) {
				return true;
			}
		}

		return false;
	}

	/**
	 *
	 *
	 * Tool Execution
	 *
	 *
	 */

	/**
	 * Executes a tool call, automatically routing to the appropriate server.
	 * Accepts the OpenAI-style tool call format.
	 * @param toolCall - Tool call with function name and arguments
	 * @param signal - Optional abort signal
	 * @returns Tool execution result
	 */
	async executeTool(toolCall: MCPToolCall, signal?: AbortSignal): Promise<ToolExecutionResult> {
		const toolName = toolCall.function.name;

		const serverName = this.toolsIndex.get(toolName);
		if (!serverName) {
			throw new Error(`Unknown tool: ${toolName}`);
		}

		const connection = this.connections.get(serverName);
		if (!connection) {
			throw new Error(`Server "${serverName}" is not connected`);
		}

		const args = this.parseToolArguments(toolCall.function.arguments);

		return MCPService.callTool(connection, { name: toolName, arguments: args }, signal);
	}

	/**
	 * Executes a tool by name with arguments object.
	 * Simpler interface for direct tool calls.
	 * @param toolName - Name of the tool to execute
	 * @param args - Tool arguments as key-value pairs
	 * @param signal - Optional abort signal
	 */
	async executeToolByName(
		toolName: string,
		args: Record<string, unknown>,
		signal?: AbortSignal
	): Promise<ToolExecutionResult> {
		const serverName = this.toolsIndex.get(toolName);
		if (!serverName) {
			throw new Error(`Unknown tool: ${toolName}`);
		}

		const connection = this.connections.get(serverName);
		if (!connection) {
			throw new Error(`Server "${serverName}" is not connected`);
		}

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
				if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
					throw new Error(
						`Tool arguments must be an object, got ${Array.isArray(parsed) ? 'array' : typeof parsed}`
					);
				}
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

	/**
	 *
	 *
	 * Completions
	 *
	 *
	 */

	/**
	 * Get completion suggestions for a prompt argument.
	 * Used for autocompleting prompt argument values.
	 *
	 * @param serverName - Name of the server hosting the prompt
	 * @param promptName - Name of the prompt
	 * @param argumentName - Name of the argument being completed
	 * @param argumentValue - Current partial value of the argument
	 * @returns Completion suggestions or null if not supported/error
	 */
	async getPromptCompletions(
		serverName: string,
		promptName: string,
		argumentName: string,
		argumentValue: string
	): Promise<{ values: string[]; total?: number; hasMore?: boolean } | null> {
		const connection = this.connections.get(serverName);
		if (!connection) {
			console.warn(`[MCPClient] Server "${serverName}" is not connected`);
			return null;
		}

		if (!connection.serverCapabilities?.completions) {
			return null;
		}

		return MCPService.complete(
			connection,
			{ type: 'ref/prompt', name: promptName },
			{ name: argumentName, value: argumentValue }
		);
	}

	/**
	 *
	 *
	 * Health Checks
	 *
	 *
	 */

	private parseHeaders(headersJson?: string): Record<string, string> | undefined {
		if (!headersJson?.trim()) return undefined;
		try {
			const parsed = JSON.parse(headersJson);
			if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
				return parsed as Record<string, string>;
			}
		} catch {
			console.warn('[MCPClient] Failed to parse custom headers JSON:', headersJson);
		}

		return undefined;
	}

	/**
	 * Run health checks for multiple servers that don't have a recent check.
	 * Useful for lazy-loading health checks when UI is opened.
	 * @param servers - Array of servers to check
	 * @param skipIfChecked - If true, skip servers that already have a health check result
	 */
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
			? servers.filter((s) => !mcpStore.hasHealthCheck(s.id) && s.url.trim())
			: servers.filter((s) => s.url.trim());

		if (serversToCheck.length === 0) return;

		const BATCH_SIZE = 5;

		for (let i = 0; i < serversToCheck.length; i += BATCH_SIZE) {
			const batch = serversToCheck.slice(i, i + BATCH_SIZE);
			await Promise.all(batch.map((server) => this.runHealthCheck(server)));
		}
	}

	/**
	 * Run health check for a specific server configuration.
	 * Creates a temporary connection to test connectivity and list tools.
	 * Tracks connection phases and collects detailed connection info.
	 */
	async runHealthCheck(server: HealthCheckParams): Promise<void> {
		const trimmedUrl = server.url.trim();
		const logs: MCPConnectionLog[] = [];
		let currentPhase: MCPConnectionPhase = MCPConnectionPhase.IDLE;

		if (!trimmedUrl) {
			mcpStore.updateHealthCheck(server.id, {
				status: HealthCheckStatus.ERROR,
				message: 'Please enter a server URL first.',
				logs: []
			});
			return;
		}

		// Initial connecting state
		mcpStore.updateHealthCheck(server.id, {
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
				// Phase callback for tracking progress
				(phase, log) => {
					currentPhase = phase;
					logs.push(log);
					mcpStore.updateHealthCheck(server.id, {
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

			mcpStore.updateHealthCheck(server.id, {
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
			mcpStore.updateHealthCheck(server.id, {
				status: HealthCheckStatus.ERROR,
				message,
				phase: currentPhase,
				logs
			});
		}
	}

	/**
	 *
	 *
	 * Status Getters
	 *
	 *
	 */

	get isInitialized(): boolean {
		return this.connections.size > 0;
	}

	get connectedServerCount(): number {
		return this.connections.size;
	}

	get connectedServerNames(): string[] {
		return Array.from(this.connections.keys());
	}

	get toolCount(): number {
		return this.toolsIndex.size;
	}

	/**
	 * Get status of all connected servers.
	 */
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

export const mcpClient = new MCPClient();
