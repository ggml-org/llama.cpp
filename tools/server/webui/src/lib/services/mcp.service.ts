/**
 * MCPService - Stateless MCP Protocol Communication Layer
 *
 * Low-level MCP operations:
 * - Transport creation (WebSocket, StreamableHTTP, SSE)
 * - Server connection/disconnection
 * - Tool discovery (listTools)
 * - Tool execution (callTool)
 *
 * NO business logic, NO state management, NO orchestration.
 * This is the protocol layer - pure MCP SDK operations.
 *
 * @see MCPClient in clients/mcp/ for business logic facade
 */

import { Client } from '@modelcontextprotocol/sdk/client';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { SSEClientTransport } from '@modelcontextprotocol/sdk/client/sse.js';
import { WebSocketClientTransport } from '@modelcontextprotocol/sdk/client/websocket.js';
import type { Tool } from '@modelcontextprotocol/sdk/types.js';
import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js';
import type {
	MCPServerConfig,
	ToolCallParams,
	ToolExecutionResult,
	Implementation,
	ClientCapabilities
} from '$lib/types/mcp';
import { MCPError } from '$lib/errors';
import { DEFAULT_MCP_CONFIG } from '$lib/constants/mcp';

/**
 * Represents an active MCP server connection.
 * Returned by MCPService.connect() and used for subsequent operations.
 */
export interface MCPConnection {
	/** MCP SDK Client instance */
	client: Client;
	/** Active transport */
	transport: Transport;
	/** Discovered tools from this server */
	tools: Tool[];
	/** Server identifier */
	serverName: string;
}

interface ToolResultContentItem {
	type: string;
	text?: string;
	data?: string;
	mimeType?: string;
	resource?: { text?: string; blob?: string; uri?: string };
}

interface ToolCallResult {
	content?: ToolResultContentItem[];
	isError?: boolean;
	_meta?: Record<string, unknown>;
}

export class MCPService {
	/**
	 * Create transport based on server configuration.
	 * Supports WebSocket, StreamableHTTP (modern), and SSE (legacy) transports.
	 */
	static createTransport(config: MCPServerConfig): Transport {
		if (!config.url) {
			throw new Error('MCP server configuration is missing url');
		}

		const url = new URL(config.url);
		const requestInit: RequestInit = {};

		if (config.headers) {
			requestInit.headers = config.headers;
		}
		if (config.credentials) {
			requestInit.credentials = config.credentials;
		}

		if (config.transport === 'websocket') {
			console.log(`[MCPService] Creating WebSocket transport for ${url.href}`);
			return new WebSocketClientTransport(url);
		}

		try {
			console.log(`[MCPService] Creating StreamableHTTP transport for ${url.href}`);
			return new StreamableHTTPClientTransport(url, {
				requestInit,
				sessionId: config.sessionId
			});
		} catch (httpError) {
			console.warn(`[MCPService] StreamableHTTP failed, trying SSE transport...`, httpError);

			try {
				return new SSEClientTransport(url, { requestInit });
			} catch (sseError) {
				const httpMsg = httpError instanceof Error ? httpError.message : String(httpError);
				const sseMsg = sseError instanceof Error ? sseError.message : String(sseError);
				throw new Error(`Failed to create transport. StreamableHTTP: ${httpMsg}; SSE: ${sseMsg}`);
			}
		}
	}

	/**
	 * Connect to a single MCP server.
	 * Returns connection object with client, transport, and discovered tools.
	 */
	static async connect(
		serverName: string,
		serverConfig: MCPServerConfig,
		clientInfo?: Implementation,
		capabilities?: ClientCapabilities
	): Promise<MCPConnection> {
		const effectiveClientInfo = clientInfo ?? DEFAULT_MCP_CONFIG.clientInfo;
		const effectiveCapabilities = capabilities ?? DEFAULT_MCP_CONFIG.capabilities;

		console.log(`[MCPService][${serverName}] Creating transport...`);
		const transport = this.createTransport(serverConfig);

		const client = new Client(
			{
				name: effectiveClientInfo.name,
				version: effectiveClientInfo.version ?? '1.0.0'
			},
			{ capabilities: effectiveCapabilities }
		);

		console.log(`[MCPService][${serverName}] Connecting to server...`);
		await client.connect(transport);

		console.log(`[MCPService][${serverName}] Connected, listing tools...`);
		const tools = await this.listTools({ client, transport, tools: [], serverName });

		console.log(`[MCPService][${serverName}] Initialization complete with ${tools.length} tools`);

		return {
			client,
			transport,
			tools,
			serverName
		};
	}

	/**
	 * Disconnect from a server.
	 */
	static async disconnect(connection: MCPConnection): Promise<void> {
		console.log(`[MCPService][${connection.serverName}] Disconnecting...`);
		try {
			await connection.client.close();
		} catch (error) {
			console.warn(`[MCPService][${connection.serverName}] Error during disconnect:`, error);
		}
	}

	/**
	 * List tools from a connection.
	 */
	static async listTools(connection: MCPConnection): Promise<Tool[]> {
		try {
			const result = await connection.client.listTools();
			return result.tools ?? [];
		} catch (error) {
			console.warn(`[MCPService][${connection.serverName}] Failed to list tools:`, error);
			return [];
		}
	}

	/**
	 * Execute a tool call on a connection.
	 */
	static async callTool(
		connection: MCPConnection,
		params: ToolCallParams,
		signal?: AbortSignal
	): Promise<ToolExecutionResult> {
		if (signal?.aborted) {
			throw new DOMException('Aborted', 'AbortError');
		}

		try {
			const result = await connection.client.callTool(
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

	/**
	 * Format tool result content to string.
	 */
	private static formatToolResult(result: ToolCallResult): string {
		const content = result.content;
		if (!Array.isArray(content)) return '';

		return content
			.map((item) => this.formatSingleContent(item))
			.filter(Boolean)
			.join('\n');
	}

	private static formatSingleContent(content: ToolResultContentItem): string {
		if (content.type === 'text' && content.text) {
			return content.text;
		}
		if (content.type === 'image' && content.data) {
			return `data:${content.mimeType ?? 'image/png'};base64,${content.data}`;
		}
		if (content.type === 'resource' && content.resource) {
			const resource = content.resource;
			if (resource.text) return resource.text;
			if (resource.blob) return resource.blob;
			return JSON.stringify(resource);
		}
		if (content.data && content.mimeType) {
			return `data:${content.mimeType};base64,${content.data}`;
		}
		return JSON.stringify(content);
	}
}
