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
	ClientCapabilities,
	MCPConnection,
	MCPPhaseCallback,
	MCPConnectionLog,
	MCPServerInfo
} from '$lib/types';
import { MCPConnectionPhase, MCPLogLevel, MCPTransportType } from '$lib/enums';
import { MCPError } from '$lib/errors';
import { DEFAULT_MCP_CONFIG } from '$lib/constants/mcp';

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
	 * Create a connection log entry
	 */
	private static createLog(
		phase: MCPConnectionPhase,
		message: string,
		level: MCPLogLevel = MCPLogLevel.Info,
		details?: unknown
	): MCPConnectionLog {
		return {
			timestamp: new Date(),
			phase,
			message,
			level,
			details
		};
	}

	/**
	 * Create transport based on server configuration.
	 * Supports WebSocket, StreamableHTTP (modern), and SSE (legacy) transports.
	 * Returns both transport and the type used.
	 */
	static createTransport(config: MCPServerConfig): {
		transport: Transport;
		type: MCPTransportType;
	} {
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

			return {
				transport: new WebSocketClientTransport(url),
				type: MCPTransportType.Websocket
			};
		}

		try {
			console.log(`[MCPService] Creating StreamableHTTP transport for ${url.href}`);

			return {
				transport: new StreamableHTTPClientTransport(url, {
					requestInit,
					sessionId: config.sessionId
				}),
				type: MCPTransportType.StreamableHttp
			};
		} catch (httpError) {
			console.warn(`[MCPService] StreamableHTTP failed, trying SSE transport...`, httpError);

			try {
				return {
					transport: new SSEClientTransport(url, { requestInit }),
					type: MCPTransportType.SSE
				};
			} catch (sseError) {
				const httpMsg = httpError instanceof Error ? httpError.message : String(httpError);
				const sseMsg = sseError instanceof Error ? sseError.message : String(sseError);
				throw new Error(`Failed to create transport. StreamableHTTP: ${httpMsg}; SSE: ${sseMsg}`);
			}
		}
	}

	/**
	 * Extract server info from SDK Implementation type
	 */
	private static extractServerInfo(impl: Implementation | undefined): MCPServerInfo | undefined {
		if (!impl) return undefined;
		return {
			name: impl.name,
			version: impl.version,
			title: impl.title,
			description: impl.description,
			websiteUrl: impl.websiteUrl,
			icons: impl.icons?.map((icon) => ({
				src: icon.src,
				mimeType: icon.mimeType,
				sizes: icon.sizes
			}))
		};
	}

	/**
	 * Connect to a single MCP server with detailed phase tracking.
	 * Returns connection object with client, transport, discovered tools, and connection details.
	 * @param onPhase - Optional callback for connection phase changes
	 */
	static async connect(
		serverName: string,
		serverConfig: MCPServerConfig,
		clientInfo?: Implementation,
		capabilities?: ClientCapabilities,
		onPhase?: MCPPhaseCallback
	): Promise<MCPConnection> {
		const startTime = performance.now();
		const effectiveClientInfo = clientInfo ?? DEFAULT_MCP_CONFIG.clientInfo;
		const effectiveCapabilities = capabilities ?? DEFAULT_MCP_CONFIG.capabilities;

		// Phase: Creating transport
		onPhase?.(
			MCPConnectionPhase.TransportCreating,
			this.createLog(
				MCPConnectionPhase.TransportCreating,
				`Creating transport for ${serverConfig.url}`
			)
		);

		console.log(`[MCPService][${serverName}] Creating transport...`);
		const { transport, type: transportType } = this.createTransport(serverConfig);

		// Phase: Transport ready
		onPhase?.(
			MCPConnectionPhase.TransportReady,
			this.createLog(MCPConnectionPhase.TransportReady, `Transport ready (${transportType})`),
			{ transportType }
		);

		const client = new Client(
			{
				name: effectiveClientInfo.name,
				version: effectiveClientInfo.version ?? '1.0.0'
			},
			{ capabilities: effectiveCapabilities }
		);

		// Phase: Initializing
		onPhase?.(
			MCPConnectionPhase.Initializing,
			this.createLog(MCPConnectionPhase.Initializing, 'Sending initialize request...')
		);

		console.log(`[MCPService][${serverName}] Connecting to server...`);
		await client.connect(transport);

		const serverVersion = client.getServerVersion();
		const serverCapabilities = client.getServerCapabilities();
		const instructions = client.getInstructions();
		const serverInfo = this.extractServerInfo(serverVersion);

		// Phase: Capabilities exchanged
		onPhase?.(
			MCPConnectionPhase.CapabilitiesExchanged,
			this.createLog(
				MCPConnectionPhase.CapabilitiesExchanged,
				'Capabilities exchanged successfully',
				MCPLogLevel.Info,
				{
					serverCapabilities,
					serverInfo
				}
			),
			{
				serverInfo,
				serverCapabilities,
				clientCapabilities: effectiveCapabilities,
				instructions
			}
		);

		// Phase: Listing tools
		onPhase?.(
			MCPConnectionPhase.ListingTools,
			this.createLog(MCPConnectionPhase.ListingTools, 'Listing available tools...')
		);

		console.log(`[MCPService][${serverName}] Connected, listing tools...`);
		const tools = await this.listTools({
			client,
			transport,
			tools: [],
			serverName,
			transportType,
			connectionTimeMs: 0
		});

		const connectionTimeMs = Math.round(performance.now() - startTime);

		// Phase: Connected
		onPhase?.(
			MCPConnectionPhase.Connected,
			this.createLog(
				MCPConnectionPhase.Connected,
				`Connection established with ${tools.length} tools (${connectionTimeMs}ms)`
			)
		);

		console.log(
			`[MCPService][${serverName}] Initialization complete with ${tools.length} tools in ${connectionTimeMs}ms`
		);

		return {
			client,
			transport,
			tools,
			serverName,
			transportType,
			serverInfo,
			serverCapabilities,
			clientCapabilities: effectiveCapabilities,
			protocolVersion: DEFAULT_MCP_CONFIG.protocolVersion,
			instructions,
			connectionTimeMs
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
