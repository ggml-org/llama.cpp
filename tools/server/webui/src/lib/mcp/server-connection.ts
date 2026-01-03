/**
 * MCPServerConnection - Wrapper na SDK Client dla pojedynczego serwera MCP.
 *
 * Zgodnie z architekturą MCP:
 * - Jeden MCPServerConnection = jedno połączenie = jeden SDK Client
 * - Izolacja między serwerami - każdy ma własny transport i capabilities
 * - Własny lifecycle (connect, disconnect)
 */

import { Client } from '@modelcontextprotocol/sdk/client';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { SSEClientTransport } from '@modelcontextprotocol/sdk/client/sse.js';
import { WebSocketClientTransport } from '@modelcontextprotocol/sdk/client/websocket.js';
import type { Tool } from '@modelcontextprotocol/sdk/types.js';
import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js';
import type { MCPServerConfig, ClientCapabilities, Implementation } from '$lib/types/mcp';
import { MCPError } from '$lib/types/mcp';
import { DEFAULT_MCP_CONFIG } from '$lib/constants/mcp';

// Type for tool call result content item
interface ToolResultContentItem {
	type: string;
	text?: string;
	data?: string;
	mimeType?: string;
	resource?: { text?: string; blob?: string; uri?: string };
}

// Type for tool call result
interface ToolCallResult {
	content?: ToolResultContentItem[];
	isError?: boolean;
	_meta?: Record<string, unknown>;
}

export interface MCPServerConnectionConfig {
	/** Unique server name/identifier */
	name: string;
	/** Server configuration */
	server: MCPServerConfig;
	/** Client info to advertise */
	clientInfo?: Implementation;
	/** Capabilities to advertise */
	capabilities?: ClientCapabilities;
}

export interface ToolCallParams {
	name: string;
	arguments: Record<string, unknown>;
}

export interface ToolExecutionResult {
	content: string;
	isError: boolean;
}

/**
 * Represents a single connection to an MCP server.
 * Wraps the SDK Client and provides a clean interface for tool operations.
 */
export class MCPServerConnection {
	private client: Client;
	private transport: Transport | null = null;
	private _tools: Tool[] = [];
	private _isConnected = false;
	private _lastError: Error | null = null;

	readonly serverName: string;
	readonly config: MCPServerConnectionConfig;

	constructor(config: MCPServerConnectionConfig) {
		this.serverName = config.name;
		this.config = config;

		const clientInfo = config.clientInfo ?? DEFAULT_MCP_CONFIG.clientInfo;
		const capabilities = config.capabilities ?? DEFAULT_MCP_CONFIG.capabilities;

		// Create SDK Client with our host info
		this.client = new Client(
			{
				name: clientInfo.name,
				version: clientInfo.version ?? '1.0.0'
			},
			{ capabilities }
		);
	}

	// ─────────────────────────────────────────────────────────────────────────
	// Lifecycle
	// ─────────────────────────────────────────────────────────────────────────

	async connect(): Promise<void> {
		if (this._isConnected) {
			console.log(`[MCP][${this.serverName}] Already connected`);
			return;
		}

		try {
			console.log(`[MCP][${this.serverName}] Creating transport...`);
			this.transport = await this.createTransport();

			console.log(`[MCP][${this.serverName}] Connecting to server...`);
			// SDK Client.connect() performs:
			// 1. initialize request → server
			// 2. Receives server capabilities
			// 3. Sends initialized notification
			await this.client.connect(this.transport);

			console.log(`[MCP][${this.serverName}] Connected, listing tools...`);
			await this.refreshTools();

			this._isConnected = true;
			this._lastError = null;
			console.log(
				`[MCP][${this.serverName}] Initialization complete with ${this._tools.length} tools`
			);
		} catch (error) {
			this._lastError = error instanceof Error ? error : new Error(String(error));
			console.error(`[MCP][${this.serverName}] Connection failed:`, error);
			throw error;
		}
	}

	async disconnect(): Promise<void> {
		if (!this._isConnected) {
			return;
		}

		console.log(`[MCP][${this.serverName}] Disconnecting...`);
		try {
			await this.client.close();
		} catch (error) {
			console.warn(`[MCP][${this.serverName}] Error during disconnect:`, error);
		}

		this._isConnected = false;
		this._tools = [];
		this.transport = null;
	}

	private async createTransport(): Promise<Transport> {
		const serverConfig = this.config.server;

		if (!serverConfig.url) {
			throw new Error('MCP server configuration is missing url');
		}

		const url = new URL(serverConfig.url);
		const requestInit: RequestInit = {};

		if (serverConfig.headers) {
			requestInit.headers = serverConfig.headers;
		}
		if (serverConfig.credentials) {
			requestInit.credentials = serverConfig.credentials;
		}

		if (serverConfig.transport === 'websocket') {
			console.log(`[MCP][${this.serverName}] Using WebSocket transport...`);
			return new WebSocketClientTransport(url);
		}

		// Try StreamableHTTP first (modern), fall back to SSE (legacy)
		try {
			console.log(`[MCP][${this.serverName}] Trying StreamableHTTP transport...`);
			const transport = new StreamableHTTPClientTransport(url, {
				requestInit,
				sessionId: serverConfig.sessionId
			});
			return transport;
		} catch (httpError) {
			console.warn(
				`[MCP][${this.serverName}] StreamableHTTP failed, trying SSE transport...`,
				httpError
			);

			try {
				const transport = new SSEClientTransport(url, {
					requestInit
				});
				return transport;
			} catch (sseError) {
				const httpMsg = httpError instanceof Error ? httpError.message : String(httpError);
				const sseMsg = sseError instanceof Error ? sseError.message : String(sseError);
				throw new Error(`Failed to create transport. StreamableHTTP: ${httpMsg}; SSE: ${sseMsg}`);
			}
		}
	}

	// ─────────────────────────────────────────────────────────────────────────
	// Tool Discovery
	// ─────────────────────────────────────────────────────────────────────────

	private async refreshTools(): Promise<void> {
		try {
			const toolsResult = await this.client.listTools();
			this._tools = toolsResult.tools ?? [];
		} catch (error) {
			console.warn(`[MCP][${this.serverName}] Failed to list tools:`, error);
			this._tools = [];
		}
	}

	get tools(): Tool[] {
		return this._tools;
	}

	get toolNames(): string[] {
		return this._tools.map((t) => t.name);
	}

	// ─────────────────────────────────────────────────────────────────────────
	// Tool Execution
	// ─────────────────────────────────────────────────────────────────────────

	async callTool(params: ToolCallParams, signal?: AbortSignal): Promise<ToolExecutionResult> {
		if (!this._isConnected) {
			throw new MCPError(`Server ${this.serverName} is not connected`, -32000);
		}

		if (signal?.aborted) {
			throw new DOMException('Aborted', 'AbortError');
		}

		try {
			const result = await this.client.callTool(
				{ name: params.name, arguments: params.arguments },
				undefined,
				{ signal }
			);

			return {
				content: this.formatToolResult(result as ToolCallResult),
				isError: (result as ToolCallResult).isError ?? false
			};
		} catch (error) {
			if (error instanceof DOMException && error.name === 'AbortError') {
				throw error;
			}
			const message = error instanceof Error ? error.message : String(error);
			throw new MCPError(`Tool execution failed: ${message}`, -32603);
		}
	}

	// ─────────────────────────────────────────────────────────────────────────
	// State
	// ─────────────────────────────────────────────────────────────────────────

	get isConnected(): boolean {
		return this._isConnected;
	}

	get lastError(): Error | null {
		return this._lastError;
	}

	// ─────────────────────────────────────────────────────────────────────────
	// Formatting
	// ─────────────────────────────────────────────────────────────────────────

	private formatToolResult(result: ToolCallResult): string {
		const content = result.content;
		if (Array.isArray(content)) {
			return content
				.map((item) => this.formatSingleContent(item))
				.filter(Boolean)
				.join('\n');
		}
		return '';
	}

	private formatSingleContent(content: ToolResultContentItem): string {
		if (content.type === 'text' && content.text) {
			return content.text;
		}
		if (content.type === 'image' && content.data) {
			return `data:${content.mimeType ?? 'image/png'};base64,${content.data}`;
		}
		if (content.type === 'resource' && content.resource) {
			const resource = content.resource;
			if (resource.text) {
				return resource.text;
			}
			if (resource.blob) {
				return resource.blob;
			}
			return JSON.stringify(resource);
		}
		// audio type
		if (content.data && content.mimeType) {
			return `data:${content.mimeType};base64,${content.data}`;
		}
		return JSON.stringify(content);
	}
}
