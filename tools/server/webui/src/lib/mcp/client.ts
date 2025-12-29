/**
 * MCP Client implementation using the official @modelcontextprotocol/sdk
 *
 * This module provides a wrapper around the SDK's Client class that maintains
 * backward compatibility with our existing MCPClient API.
 */

import { Client } from '@modelcontextprotocol/sdk/client';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { SSEClientTransport } from '@modelcontextprotocol/sdk/client/sse.js';
import type { Tool } from '@modelcontextprotocol/sdk/types.js';
import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js';
import type { MCPClientConfig, MCPServerConfig, MCPToolCall } from '$lib/types/mcp';
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

// Type for tool call result (SDK uses complex union type)
interface ToolCallResult {
	content?: ToolResultContentItem[];
	isError?: boolean;
	_meta?: Record<string, unknown>;
}

interface ServerConnection {
	client: Client;
	transport: Transport;
	tools: Tool[];
}

/**
 * MCP Client using the official @modelcontextprotocol/sdk.
 */
export class MCPClient {
	private readonly servers: Map<string, ServerConnection> = new Map();
	private readonly toolsToServer: Map<string, string> = new Map();
	private readonly config: MCPClientConfig;

	constructor(config: MCPClientConfig) {
		if (!config?.servers || Object.keys(config.servers).length === 0) {
			throw new Error('MCPClient requires at least one server configuration');
		}
		this.config = config;
	}

	async initialize(): Promise<void> {
		const entries = Object.entries(this.config.servers);
		const results = await Promise.allSettled(
			entries.map(([name, serverConfig]) => this.initializeServer(name, serverConfig))
		);

		// Log any failures but don't throw if at least one server connected
		const failures = results.filter((r) => r.status === 'rejected');
		if (failures.length > 0) {
			for (const failure of failures) {
				console.error(
					'[MCP] Server initialization failed:',
					(failure as PromiseRejectedResult).reason
				);
			}
		}

		const successes = results.filter((r) => r.status === 'fulfilled');
		if (successes.length === 0) {
			throw new Error('All MCP server connections failed');
		}
	}

	listTools(): string[] {
		return Array.from(this.toolsToServer.keys());
	}

	async getToolsDefinition(): Promise<
		{
			type: 'function';
			function: { name: string; description?: string; parameters: Record<string, unknown> };
		}[]
	> {
		const tools: {
			type: 'function';
			function: { name: string; description?: string; parameters: Record<string, unknown> };
		}[] = [];

		for (const [, server] of this.servers) {
			for (const tool of server.tools) {
				tools.push({
					type: 'function',
					function: {
						name: tool.name,
						description: tool.description,
						parameters: (tool.inputSchema as Record<string, unknown>) ?? {
							type: 'object',
							properties: {},
							required: []
						}
					}
				});
			}
		}

		return tools;
	}

	async execute(toolCall: MCPToolCall, abortSignal?: AbortSignal): Promise<string> {
		const toolName = toolCall.function.name;
		const serverName = this.toolsToServer.get(toolName);
		if (!serverName) {
			throw new MCPError(`Unknown tool: ${toolName}`, -32601);
		}

		const connection = this.servers.get(serverName);
		if (!connection) {
			throw new MCPError(`Server ${serverName} is not connected`, -32000);
		}

		if (abortSignal?.aborted) {
			throw new DOMException('Aborted', 'AbortError');
		}

		// Parse arguments
		let args: Record<string, unknown>;
		const originalArgs = toolCall.function.arguments;
		if (typeof originalArgs === 'string') {
			const trimmed = originalArgs.trim();
			if (trimmed === '') {
				args = {};
			} else {
				try {
					const parsed = JSON.parse(trimmed);
					if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
						throw new MCPError(
							`Tool arguments must be an object, got ${Array.isArray(parsed) ? 'array' : typeof parsed}`,
							-32602
						);
					}
					args = parsed as Record<string, unknown>;
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
		} else if (
			typeof originalArgs === 'object' &&
			originalArgs !== null &&
			!Array.isArray(originalArgs)
		) {
			args = originalArgs as Record<string, unknown>;
		} else {
			throw new MCPError(`Invalid tool arguments type: ${typeof originalArgs}`, -32602);
		}

		try {
			const result = await connection.client.callTool(
				{ name: toolName, arguments: args },
				undefined,
				{ signal: abortSignal }
			);

			return MCPClient.formatToolResult(result as ToolCallResult);
		} catch (error) {
			if (error instanceof DOMException && error.name === 'AbortError') {
				throw error;
			}
			const message = error instanceof Error ? error.message : String(error);
			throw new MCPError(`Tool execution failed: ${message}`, -32603);
		}
	}

	async shutdown(): Promise<void> {
		const closePromises: Promise<void>[] = [];

		for (const [name, connection] of this.servers) {
			console.log(`[MCP][${name}] Closing connection...`);
			closePromises.push(
				connection.client.close().catch((error) => {
					console.warn(`[MCP][${name}] Error closing client:`, error);
				})
			);
		}

		await Promise.allSettled(closePromises);
		this.servers.clear();
		this.toolsToServer.clear();
	}

	private async initializeServer(name: string, config: MCPServerConfig): Promise<void> {
		console.log(`[MCP][${name}] Starting server initialization...`);

		const clientInfo = this.config.clientInfo ?? DEFAULT_MCP_CONFIG.clientInfo;
		const capabilities =
			config.capabilities ?? this.config.capabilities ?? DEFAULT_MCP_CONFIG.capabilities;

		// Create SDK client
		const client = new Client(
			{ name: clientInfo.name, version: clientInfo.version ?? '1.0.0' },
			{ capabilities }
		);

		// Create transport with fallback
		const transport = await this.createTransportWithFallback(name, config);

		console.log(`[MCP][${name}] Connecting to server...`);
		await client.connect(transport);
		console.log(`[MCP][${name}] Connected, listing tools...`);

		// List available tools
		const toolsResult = await client.listTools();
		const tools = toolsResult.tools ?? [];
		console.log(`[MCP][${name}] Found ${tools.length} tools`);

		// Store connection
		const connection: ServerConnection = {
			client,
			transport,
			tools
		};
		this.servers.set(name, connection);

		// Map tools to server
		for (const tool of tools) {
			this.toolsToServer.set(tool.name, name);
		}

		// Note: Tool list changes will be handled by re-calling listTools when needed
		// The SDK's listChanged handler requires server capability support

		console.log(`[MCP][${name}] Server initialization complete`);
	}

	private async createTransportWithFallback(
		name: string,
		config: MCPServerConfig
	): Promise<Transport> {
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

		// Try StreamableHTTP first (modern), fall back to SSE (legacy)
		try {
			console.log(`[MCP][${name}] Trying StreamableHTTP transport...`);
			const transport = new StreamableHTTPClientTransport(url, {
				requestInit,
				sessionId: config.sessionId
			});
			return transport;
		} catch (httpError) {
			console.warn(`[MCP][${name}] StreamableHTTP failed, trying SSE transport...`, httpError);

			try {
				const transport = new SSEClientTransport(url, {
					requestInit
				});
				return transport;
			} catch (sseError) {
				// Both failed, throw combined error
				const httpMsg = httpError instanceof Error ? httpError.message : String(httpError);
				const sseMsg = sseError instanceof Error ? sseError.message : String(sseError);
				throw new Error(`Failed to create transport. StreamableHTTP: ${httpMsg}; SSE: ${sseMsg}`);
			}
		}
	}

	private static formatToolResult(result: ToolCallResult): string {
		const content = result.content;
		if (Array.isArray(content)) {
			return content
				.map((item) => MCPClient.formatSingleContent(item))
				.filter(Boolean)
				.join('\n');
		}
		return '';
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
