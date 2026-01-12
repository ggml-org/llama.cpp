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

import { browser } from '$app/environment';
import { MCPService, type MCPConnection } from '$lib/services/mcp.service';
import type {
	MCPToolCall,
	OpenAIToolDefinition,
	ServerStatus,
	ToolExecutionResult,
	MCPClientConfig
} from '$lib/types/mcp';
import type { McpServerOverride } from '$lib/types/database';
import { MCPError } from '$lib/errors';
import { buildMcpClientConfig, detectMcpTransportFromUrl } from '$lib/utils/mcp';
import { config } from '$lib/stores/settings.svelte';
import { DEFAULT_MCP_CONFIG } from '$lib/constants/mcp';

export type HealthCheckState =
	| { status: 'idle' }
	| { status: 'loading' }
	| { status: 'error'; message: string }
	| { status: 'success'; tools: { name: string; description?: string }[] };

export interface HealthCheckParams {
	id: string;
	url: string;
	requestTimeoutSeconds: number;
	headers?: string;
}

export class MCPClient {
	private connections = new Map<string, MCPConnection>();
	private toolsIndex = new Map<string, string>();
	private configSignature: string | null = null;
	private initPromise: Promise<boolean> | null = null;

	private onStateChange?: (state: {
		isInitializing?: boolean;
		error?: string | null;
		toolCount?: number;
		connectedServers?: string[];
	}) => void;

	private onHealthCheckChange?: (serverId: string, state: HealthCheckState) => void;
	private onServerUsage?: (serverId: string) => void;

	/**
	 *
	 *
	 * Store Integration
	 *
	 *
	 */

	/**
	 * Sets callback for state changes.
	 * Called by mcpStore to sync reactive state.
	 */
	setStateChangeCallback(
		callback: (state: {
			isInitializing?: boolean;
			error?: string | null;
			toolCount?: number;
			connectedServers?: string[];
		}) => void
	): void {
		this.onStateChange = callback;
	}

	/**
	 * Set callback for health check state changes
	 */
	setHealthCheckCallback(callback: (serverId: string, state: HealthCheckState) => void): void {
		this.onHealthCheckChange = callback;
	}

	/**
	 * Set callback for server usage tracking
	 */
	setServerUsageCallback(callback: (serverId: string) => void): void {
		this.onServerUsage = callback;
	}

	private notifyStateChange(state: Parameters<NonNullable<typeof this.onStateChange>>[0]): void {
		this.onStateChange?.(state);
	}

	private notifyHealthCheck(serverId: string, state: HealthCheckState): void {
		this.onHealthCheckChange?.(serverId, state);
	}

	/**
	 *
	 *
	 * Lifecycle
	 *
	 *
	 */

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

		this.notifyStateChange({ isInitializing: true, error: null });
		this.configSignature = signature;

		const serverEntries = Object.entries(mcpConfig.servers);
		if (serverEntries.length === 0) {
			console.log('[MCPClient] No servers configured');
			this.notifyStateChange({ isInitializing: false, toolCount: 0, connectedServers: [] });
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
				const connection = await MCPService.connect(name, serverConfig, clientInfo, capabilities);
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
			this.notifyStateChange({
				isInitializing: false,
				error,
				toolCount: 0,
				connectedServers: []
			});
			this.initPromise = null;
			return false;
		}

		this.notifyStateChange({
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
	 * Shutdown all MCP connections and clear state.
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

		this.notifyStateChange({
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
			throw new MCPError(`Unknown tool: ${toolName}`, -32601);
		}

		const connection = this.connections.get(serverName);
		if (!connection) {
			throw new MCPError(`Server "${serverName}" is not connected`, -32000);
		}

		this.onServerUsage?.(serverName);

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
			throw new MCPError(`Unknown tool: ${toolName}`, -32601);
		}

		const connection = this.connections.get(serverName);
		if (!connection) {
			throw new MCPError(`Server "${serverName}" is not connected`, -32000);
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
					throw new MCPError(
						`Tool arguments must be an object, got ${Array.isArray(parsed) ? 'array' : typeof parsed}`,
						-32602
					);
				}
				return parsed as Record<string, unknown>;
			} catch (error) {
				if (error instanceof MCPError) {
					throw error;
				}
				throw new MCPError(
					`Failed to parse tool arguments as JSON: ${(error as Error).message}`,
					-32700
				);
			}
		}

		if (typeof args === 'object' && args !== null && !Array.isArray(args)) {
			return args;
		}

		throw new MCPError(`Invalid tool arguments type: ${typeof args}`, -32602);
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
	 * Run health check for a specific server configuration.
	 * Creates a temporary connection to test connectivity and list tools.
	 */
	async runHealthCheck(server: HealthCheckParams): Promise<void> {
		const trimmedUrl = server.url.trim();

		if (!trimmedUrl) {
			this.notifyHealthCheck(server.id, {
				status: 'error',
				message: 'Please enter a server URL first.'
			});
			return;
		}

		this.notifyHealthCheck(server.id, { status: 'loading' });

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
				DEFAULT_MCP_CONFIG.capabilities
			);

			const tools = connection.tools.map((tool) => ({
				name: tool.name,
				description: tool.description
			}));

			this.notifyHealthCheck(server.id, { status: 'success', tools });
			await MCPService.disconnect(connection);
		} catch (error) {
			const message = error instanceof Error ? error.message : 'Unknown error occurred';
			this.notifyHealthCheck(server.id, { status: 'error', message });
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
