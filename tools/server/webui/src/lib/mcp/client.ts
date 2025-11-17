import { getDefaultMcpConfig } from '$lib/config/mcp';
import { JsonRpcProtocol } from './protocol';
import type {
	JsonRpcMessage,
	MCPClientConfig,
	MCPServerCapabilities,
	MCPServerConfig,
	MCPToolCall,
	MCPToolDefinition,
	MCPToolsCallResult
} from './types';
import { MCPError } from './types';
import type { MCPTransport } from './transports/types';
import { WebSocketTransport } from './transports/websocket';
import { StreamableHttpTransport } from './transports/streamable-http';

const MCP_DEFAULTS = getDefaultMcpConfig();

interface PendingRequest {
	resolve: (value: Record<string, unknown>) => void;
	reject: (reason?: unknown) => void;
	timeout: ReturnType<typeof setTimeout>;
}

interface ServerState {
	transport: MCPTransport;
	pending: Map<number, PendingRequest>;
	requestId: number;
	tools: MCPToolDefinition[];
	requestTimeoutMs?: number;
	capabilities?: MCPServerCapabilities;
	protocolVersion?: string;
}

export class MCPClient {
	private readonly servers: Map<string, ServerState> = new Map();
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
		await Promise.all(
			entries.map(([name, serverConfig]) => this.initializeServer(name, serverConfig))
		);
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
						parameters: tool.inputSchema ?? {
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

		if (abortSignal?.aborted) {
			throw new DOMException('Aborted', 'AbortError');
		}

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

		const response = await this.call(
			serverName,
			'tools/call',
			{
				name: toolName,
				arguments: args
			},
			abortSignal
		);

		return MCPClient.formatToolResult(response as MCPToolsCallResult);
	}

	async shutdown(): Promise<void> {
		for (const [, state] of this.servers) {
			await state.transport.stop();
		}
		this.servers.clear();
		this.toolsToServer.clear();
	}

	private async initializeServer(name: string, config: MCPServerConfig): Promise<void> {
		const protocolVersion = this.config.protocolVersion ?? MCP_DEFAULTS.protocolVersion;
		const transport = this.createTransport(config, protocolVersion);
		await transport.start();

		const state: ServerState = {
			transport,
			pending: new Map(),
			requestId: 0,
			tools: [],
			requestTimeoutMs: config.requestTimeoutMs
		};

		transport.onMessage((message) => this.handleMessage(name, message));
		this.servers.set(name, state);

		const clientInfo = this.config.clientInfo ?? MCP_DEFAULTS.clientInfo;
		const capabilities =
			config.capabilities ?? this.config.capabilities ?? MCP_DEFAULTS.capabilities;

		const initResult = await this.call(name, 'initialize', {
			protocolVersion,
			capabilities,
			clientInfo
		});

		const negotiatedVersion = (initResult?.protocolVersion as string) ?? protocolVersion;

		state.capabilities = (initResult?.capabilities as MCPServerCapabilities) ?? {};
		state.protocolVersion = negotiatedVersion;

		const notification = JsonRpcProtocol.createNotification('notifications/initialized');
		await state.transport.send(notification as JsonRpcMessage);

		await this.refreshTools(name);
	}

	private createTransport(config: MCPServerConfig, protocolVersion: string): MCPTransport {
		if (!config.url) {
			throw new Error('MCP server configuration is missing url');
		}

		const transportType = config.transport ?? 'websocket';

		if (transportType === 'streamable_http') {
			return new StreamableHttpTransport({
				url: config.url,
				headers: config.headers,
				credentials: config.credentials,
				protocolVersion,
				sessionId: config.sessionId
			});
		}

		if (transportType !== 'websocket') {
			throw new Error(`Unsupported transport "${transportType}" in webui environment`);
		}

		return new WebSocketTransport({
			url: config.url,
			protocols: config.protocols,
			handshakeTimeoutMs: config.handshakeTimeoutMs
		});
	}

	private async refreshTools(serverName: string): Promise<void> {
		const state = this.servers.get(serverName);
		if (!state) return;

		const response = await this.call(serverName, 'tools/list');
		const tools = (response.tools as MCPToolDefinition[]) ?? [];
		state.tools = tools;

		for (const [tool, owner] of Array.from(this.toolsToServer.entries())) {
			if (owner === serverName && !tools.find((t) => t.name === tool)) {
				this.toolsToServer.delete(tool);
			}
		}

		for (const tool of tools) {
			this.toolsToServer.set(tool.name, serverName);
		}
	}

	private call(
		serverName: string,
		method: string,
		params?: Record<string, unknown>,
		abortSignal?: AbortSignal
	): Promise<Record<string, unknown>> {
		const state = this.servers.get(serverName);
		if (!state) {
			return Promise.reject(new MCPError(`Server ${serverName} is not connected`, -32000));
		}

		const id = ++state.requestId;
		const message = JsonRpcProtocol.createRequest(id, method, params);

		const timeoutDuration =
			state.requestTimeoutMs ??
			this.config.requestTimeoutMs ??
			MCP_DEFAULTS.requestTimeoutSeconds * 1000;

		if (abortSignal?.aborted) {
			return Promise.reject(new DOMException('Aborted', 'AbortError'));
		}

		return new Promise((resolve, reject) => {
			const cleanupTasks: Array<() => void> = [];
			const cleanup = () => {
				for (const task of cleanupTasks.splice(0)) {
					task();
				}
			};

			const timeout = setTimeout(() => {
				cleanup();
				reject(new Error(`Timeout while waiting for ${method} response from ${serverName}`));
			}, timeoutDuration);
			cleanupTasks.push(() => clearTimeout(timeout));
			cleanupTasks.push(() => state.pending.delete(id));

			if (abortSignal) {
				const abortHandler = () => {
					cleanup();
					reject(new DOMException('Aborted', 'AbortError'));
				};
				abortSignal.addEventListener('abort', abortHandler, { once: true });
				cleanupTasks.push(() => abortSignal.removeEventListener('abort', abortHandler));
			}

			state.pending.set(id, {
				resolve: (value) => {
					cleanup();
					resolve(value);
				},
				reject: (reason) => {
					cleanup();
					reject(reason);
				},
				timeout
			});

			const handleSendError = (error: unknown) => {
				cleanup();
				reject(error);
			};

			try {
				void state.transport
					.send(message as JsonRpcMessage)
					.catch((error) => handleSendError(error));
			} catch (error) {
				handleSendError(error);
			}
		});
	}

	private handleMessage(serverName: string, message: JsonRpcMessage): void {
		const state = this.servers.get(serverName);
		if (!state) {
			return;
		}

		if ('method' in message && !('id' in message)) {
			this.handleNotification(serverName, message.method, message.params);
			return;
		}

		const response = JsonRpcProtocol.parseResponse(message);
		if (!response) {
			return;
		}

		const pending = state.pending.get(response.id as number);
		if (!pending) {
			return;
		}

		state.pending.delete(response.id as number);
		clearTimeout(pending.timeout);

		if (response.error) {
			pending.reject(
				new MCPError(response.error.message, response.error.code, response.error.data)
			);
			return;
		}

		pending.resolve(response.result ?? {});
	}

	private handleNotification(
		serverName: string,
		method: string,
		params?: Record<string, unknown>
	): void {
		if (method === 'notifications/tools/list_changed') {
			void this.refreshTools(serverName).catch((error) => {
				console.error(`[MCP] Failed to refresh tools for ${serverName}:`, error);
			});
		} else if (method === 'notifications/logging/message' && params) {
			console.debug(`[MCP][${serverName}]`, params);
		}
	}

	private static formatToolResult(result: MCPToolsCallResult): string {
		const content = result.content;
		if (Array.isArray(content)) {
			return content
				.map((item) => MCPClient.formatSingleContent(item))
				.filter(Boolean)
				.join('\n');
		}
		if (content) {
			return MCPClient.formatSingleContent(content);
		}
		if (result.result !== undefined) {
			return typeof result.result === 'string' ? result.result : JSON.stringify(result.result);
		}
		return '';
	}

	private static formatSingleContent(content: unknown): string {
		if (content === null || content === undefined) {
			return '';
		}

		if (typeof content === 'string') {
			return content;
		}

		if (typeof content === 'object') {
			const typed = content as {
				type?: string;
				text?: string;
				data?: string;
				mimeType?: string;
				resource?: unknown;
			};
			if (typed.type === 'text' && typeof typed.text === 'string') {
				return typed.text;
			}
			if (typed.type === 'image' && typeof typed.data === 'string' && typed.mimeType) {
				return `data:${typed.mimeType};base64,${typed.data}`;
			}
			if (typed.type === 'resource' && typed.resource) {
				return JSON.stringify(typed.resource);
			}
			if (typeof typed.text === 'string') {
				return typed.text;
			}
		}

		return JSON.stringify(content);
	}
}
