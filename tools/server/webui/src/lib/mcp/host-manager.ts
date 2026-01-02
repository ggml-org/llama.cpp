/**
 * MCPHostManager - Agregator wielu połączeń MCP.
 *
 * Zgodnie z architekturą MCP, Host:
 * - Koordynuje wiele instancji Client (MCPServerConnection)
 * - Agreguje tools/resources/prompts ze wszystkich serwerów
 * - Routuje tool calls do odpowiedniego serwera
 * - Zarządza lifecycle wszystkich połączeń
 */

import { MCPServerConnection, type ToolExecutionResult } from './server-connection';
import type {
	MCPClientConfig,
	MCPToolCall,
	ClientCapabilities,
	Implementation
} from '$lib/types/mcp';
import { MCPError } from '$lib/types/mcp';
import type { Tool } from '@modelcontextprotocol/sdk/types.js';

export interface MCPHostManagerConfig {
	/** Server configurations keyed by server name */
	servers: MCPClientConfig['servers'];
	/** Client info to advertise to all servers */
	clientInfo?: Implementation;
	/** Default capabilities to advertise */
	capabilities?: ClientCapabilities;
}

export interface OpenAIToolDefinition {
	type: 'function';
	function: {
		name: string;
		description?: string;
		parameters: Record<string, unknown>;
	};
}

export interface ServerStatus {
	name: string;
	isConnected: boolean;
	toolCount: number;
	error?: string;
}

/**
 * MCPHostManager manages multiple MCP server connections.
 *
 * This corresponds to the "Host" role in MCP architecture:
 * - Coordinates multiple Client instances (MCPServerConnection)
 * - Aggregates tools from all connected servers
 * - Routes tool calls to the appropriate server
 */
export class MCPHostManager {
	private connections = new Map<string, MCPServerConnection>();
	private toolsIndex = new Map<string, string>(); // toolName → serverName
	private _isInitialized = false;
	private _initializationError: Error | null = null;

	// ─────────────────────────────────────────────────────────────────────────
	// Lifecycle
	// ─────────────────────────────────────────────────────────────────────────

	async initialize(config: MCPHostManagerConfig): Promise<void> {
		console.log('[MCPHost] Starting initialization...');

		// Clean up previous connections
		await this.shutdown();

		const serverEntries = Object.entries(config.servers);
		if (serverEntries.length === 0) {
			console.log('[MCPHost] No servers configured');
			this._isInitialized = true;
			return;
		}

		// Connect to each server in parallel
		const connectionPromises = serverEntries.map(async ([name, serverConfig]) => {
			try {
				const connection = new MCPServerConnection({
					name,
					server: serverConfig,
					clientInfo: config.clientInfo,
					capabilities: config.capabilities
				});

				await connection.connect();
				return { name, connection, success: true, error: null };
			} catch (error) {
				const errorMessage = error instanceof Error ? error.message : String(error);
				console.error(`[MCPHost] Failed to connect to ${name}:`, errorMessage);
				return { name, connection: null, success: false, error: errorMessage };
			}
		});

		const results = await Promise.all(connectionPromises);

		// Store successful connections
		for (const result of results) {
			if (result.success && result.connection) {
				this.connections.set(result.name, result.connection);
			}
		}

		// Build tools index
		this.rebuildToolsIndex();

		const successCount = this.connections.size;
		const totalCount = serverEntries.length;

		if (successCount === 0 && totalCount > 0) {
			this._initializationError = new Error('All MCP server connections failed');
			throw this._initializationError;
		}

		this._isInitialized = true;
		this._initializationError = null;

		console.log(
			`[MCPHost] Initialization complete: ${successCount}/${totalCount} servers connected, ` +
				`${this.toolsIndex.size} tools available`
		);
	}

	async shutdown(): Promise<void> {
		if (this.connections.size === 0) {
			return;
		}

		console.log(`[MCPHost] Shutting down ${this.connections.size} connections...`);

		const shutdownPromises = Array.from(this.connections.values()).map((conn) =>
			conn.disconnect().catch((error) => {
				console.warn(`[MCPHost] Error disconnecting ${conn.serverName}:`, error);
			})
		);

		await Promise.all(shutdownPromises);

		this.connections.clear();
		this.toolsIndex.clear();
		this._isInitialized = false;

		console.log('[MCPHost] Shutdown complete');
	}

	private rebuildToolsIndex(): void {
		this.toolsIndex.clear();

		for (const [serverName, connection] of this.connections) {
			for (const tool of connection.tools) {
				// Check for name conflicts
				if (this.toolsIndex.has(tool.name)) {
					console.warn(
						`[MCPHost] Tool name conflict: "${tool.name}" exists in ` +
							`"${this.toolsIndex.get(tool.name)}" and "${serverName}". ` +
							`Using tool from "${serverName}".`
					);
				}
				this.toolsIndex.set(tool.name, serverName);
			}
		}
	}

	// ─────────────────────────────────────────────────────────────────────────
	// Tool Aggregation
	// ─────────────────────────────────────────────────────────────────────────

	/**
	 * Returns ALL tools from ALL connected servers.
	 * This is what we send to LLM as available tools.
	 */
	getAllTools(): Tool[] {
		const allTools: Tool[] = [];
		for (const connection of this.connections.values()) {
			allTools.push(...connection.tools);
		}
		return allTools;
	}

	/**
	 * Normalize JSON Schema properties to ensure all have explicit types.
	 * Infers type from default value if missing - fixes compatibility with
	 * llama.cpp which requires explicit types in tool schemas.
	 */
	private normalizeSchemaProperties(schema: Record<string, unknown>): Record<string, unknown> {
		if (!schema || typeof schema !== 'object') return schema;

		const normalized = { ...schema };

		// Process properties object
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

				// Recursively normalize nested schemas
				if (normalizedProp.properties) {
					Object.assign(
						normalizedProp,
						this.normalizeSchemaProperties(normalizedProp as Record<string, unknown>)
					);
				}

				// Normalize items in array schemas
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
	 * Returns tools in OpenAI function calling format.
	 * Ready to be sent to /v1/chat/completions API.
	 */
	getToolDefinitionsForLLM(): OpenAIToolDefinition[] {
		return this.getAllTools().map((tool) => {
			const rawSchema = (tool.inputSchema as Record<string, unknown>) ?? {
				type: 'object',
				properties: {},
				required: []
			};

			// Normalize schema to fix missing types
			const normalizedSchema = this.normalizeSchemaProperties(rawSchema);

			return {
				type: 'function' as const,
				function: {
					name: tool.name,
					description: tool.description,
					parameters: normalizedSchema
				}
			};
		});
	}

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

	// ─────────────────────────────────────────────────────────────────────────
	// Tool Execution
	// ─────────────────────────────────────────────────────────────────────────

	/**
	 * Execute a tool call, automatically routing to the appropriate server.
	 * Accepts the OpenAI-style tool call format.
	 */
	async executeTool(toolCall: MCPToolCall, signal?: AbortSignal): Promise<ToolExecutionResult> {
		const toolName = toolCall.function.name;

		// Find which server handles this tool
		const serverName = this.toolsIndex.get(toolName);
		if (!serverName) {
			throw new MCPError(`Unknown tool: ${toolName}`, -32601);
		}

		const connection = this.connections.get(serverName);
		if (!connection) {
			throw new MCPError(`Server "${serverName}" is not connected`, -32000);
		}

		// Parse arguments
		const args = this.parseToolArguments(toolCall.function.arguments);

		// Delegate to the appropriate server
		return connection.callTool({ name: toolName, arguments: args }, signal);
	}

	/**
	 * Execute a tool by name with arguments object.
	 * Simpler interface for direct tool calls.
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

		return connection.callTool({ name: toolName, arguments: args }, signal);
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

	// ─────────────────────────────────────────────────────────────────────────
	// State
	// ─────────────────────────────────────────────────────────────────────────

	get isInitialized(): boolean {
		return this._isInitialized;
	}

	get initializationError(): Error | null {
		return this._initializationError;
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
	 * Get status of all configured servers.
	 */
	getServersStatus(): ServerStatus[] {
		const statuses: ServerStatus[] = [];

		for (const [name, connection] of this.connections) {
			statuses.push({
				name,
				isConnected: connection.isConnected,
				toolCount: connection.tools.length,
				error: connection.lastError?.message
			});
		}

		return statuses;
	}

	/**
	 * Get a specific server connection (for advanced use cases).
	 */
	getServerConnection(name: string): MCPServerConnection | undefined {
		return this.connections.get(name);
	}
}
