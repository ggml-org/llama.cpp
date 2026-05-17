import { Client } from '@modelcontextprotocol/sdk/client';
import {
	StreamableHTTPClientTransport,
	StreamableHTTPError
} from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { SSEClientTransport } from '@modelcontextprotocol/sdk/client/sse.js';
import { WebSocketClientTransport } from '@modelcontextprotocol/sdk/client/websocket.js';
import type {
	Tool,
	Prompt,
	GetPromptResult,
	ListChangedHandlers
} from '@modelcontextprotocol/sdk/types.js';
import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js';
import {
	DEFAULT_MCP_CONFIG,
	DEFAULT_CLIENT_VERSION,
	DEFAULT_IMAGE_MIME_TYPE,
	MCP_PARTIAL_REDACT_HEADERS
} from '$lib/constants';
import {
	MCPConnectionPhase,
	MCPLogLevel,
	MCPTransportType,
	MCPContentType,
	MCPRefType
} from '$lib/enums';
import type {
	MCPServerConfig,
	MCPResourceIcon,
	ToolCallParams,
	ToolExecutionResult,
	Implementation,
	ClientCapabilities,
	MCPConnection,
	MCPPhaseCallback,
	MCPConnectionLog,
	MCPServerInfo,
	MCPResource,
	MCPResourceTemplate,
	MCPResourceContent,
	MCPReadResourceResult
} from '$lib/types';
import {
	buildProxiedUrl,
	buildProxiedHeaders,
	getAuthHeaders,
	sanitizeHeaders,
	throwIfAborted,
	isAbortError,
	createBase64DataUrl,
	getRequestUrl,
	getRequestMethod,
	getRequestBody,
	summarizeRequestBody,
	formatDiagnosticErrorMessage,
	extractJsonRpcMethods,
	type RequestBodySummary
} from '$lib/utils';

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

interface DiagnosticRequestDetails {
	url: string;
	method: string;
	credentials?: RequestCredentials;
	mode?: RequestMode;
	headers: Record<string, string>;
	body: RequestBodySummary;
	jsonRpcMethods?: string[];
}

export class MCPService {
	/**
	 * Create a connection log entry for phase tracking.
	 *
	 * @param phase - The connection phase this log belongs to
	 * @param message - Human-readable log message
	 * @param level - Log severity level (default: INFO)
	 * @param details - Optional structured details for debugging
	 * @returns Formatted connection log entry
	 */
	private static createLog(
		phase: MCPConnectionPhase,
		message: string,
		level: MCPLogLevel = MCPLogLevel.INFO,
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

	private static createDiagnosticRequestDetails(
		input: RequestInfo | URL,
		init: RequestInit | undefined,
		baseInit: RequestInit,
		requestHeaders: Headers,
		extraRedactedHeaders?: Iterable<string>
	): DiagnosticRequestDetails {
		const body = getRequestBody(input, init);
		const details: DiagnosticRequestDetails = {
			url: getRequestUrl(input),
			method: getRequestMethod(input, init, baseInit).toUpperCase(),
			credentials: init?.credentials ?? baseInit.credentials,
			mode: init?.mode ?? baseInit.mode,
			headers: sanitizeHeaders(requestHeaders, extraRedactedHeaders, MCP_PARTIAL_REDACT_HEADERS),
			body: summarizeRequestBody(body)
		};
		const jsonRpcMethods = extractJsonRpcMethods(body);

		if (jsonRpcMethods) {
			details.jsonRpcMethods = jsonRpcMethods;
		}

		return details;
	}

	private static summarizeError(error: unknown): Record<string, unknown> {
		if (error instanceof Error) {
			return {
				name: error.name,
				message: error.message,
				cause:
					error.cause instanceof Error
						? { name: error.cause.name, message: error.cause.message }
						: error.cause,
				stack: error.stack?.split('\n').slice(0, 6).join('\n')
			};
		}

		return { value: String(error) };
	}

	private static getBrowserContext(
		targetUrl: URL,
		useProxy: boolean
	): Record<string, unknown> | undefined {
		if (typeof window === 'undefined') {
			return undefined;
		}

		return {
			location: window.location.href,
			origin: window.location.origin,
			protocol: window.location.protocol,
			isSecureContext: window.isSecureContext,
			targetOrigin: targetUrl.origin,
			targetProtocol: targetUrl.protocol,
			sameOrigin: window.location.origin === targetUrl.origin,
			useProxy
		};
	}

	private static getConnectionHints(
		targetUrl: URL,
		config: MCPServerConfig,
		error: unknown
	): string[] {
		const hints: string[] = [];
		const message = error instanceof Error ? error.message : String(error);
		const headerNames = Object.keys(config.headers ?? {});

		if (typeof window !== 'undefined') {
			if (
				window.location.protocol === 'https:' &&
				targetUrl.protocol === 'http:' &&
				!config.useProxy
			) {
				hints.push(
					'The page is running over HTTPS but the MCP server is HTTP. Browsers often block this as mixed content; enable the proxy or use HTTPS/WSS for the MCP server.'
				);
			}

			if (window.location.origin !== targetUrl.origin && !config.useProxy) {
				hints.push(
					'This is a cross-origin browser request. If the server is reachable from curl or Node but not from the browser, missing CORS headers are the most likely cause.'
				);
			}
		}

		if (headerNames.length > 0) {
			hints.push(
				`Custom request headers are configured (${headerNames.join(', ')}). That triggers a CORS preflight, so the server must allow OPTIONS and include the matching Access-Control-Allow-Headers response.`
			);
		}

		if (config.credentials && config.credentials !== 'omit') {
			hints.push(
				'Credentials are enabled for this connection. Cross-origin credentialed requests need Access-Control-Allow-Credentials: true and cannot use a wildcard Access-Control-Allow-Origin.'
			);
		}

		if (message.includes('Failed to fetch')) {
			hints.push(
				'"Failed to fetch" is a browser-level network failure. Common causes are CORS rejection, mixed-content blocking, certificate/TLS errors, DNS failures, or nothing listening on the target port.'
			);
		}

		return hints;
	}

	private static createDiagnosticFetch(
		serverName: string,
		config: MCPServerConfig,
		baseInit: RequestInit,
		targetUrl: URL,
		useProxy: boolean,
		onLog?: (log: MCPConnectionLog) => void
	): {
		fetch: typeof fetch;
		disable: () => void;
	} {
		let enabled = true;
		const logIfEnabled = (log: MCPConnectionLog) => {
			if (enabled) {
				onLog?.(log);
			}
		};

		return {
			fetch: async (input, init) => {
				const startedAt = performance.now();
				const requestHeaders = new Headers(baseInit.headers);

				if (typeof Request !== 'undefined' && input instanceof Request) {
					for (const [key, value] of input.headers.entries()) {
						requestHeaders.set(key, value);
					}
				}

				if (init?.headers) {
					for (const [key, value] of new Headers(init.headers).entries()) {
						requestHeaders.set(key, value);
					}
				}

				const request = this.createDiagnosticRequestDetails(
					input,
					init,
					baseInit,
					requestHeaders,
					Object.keys(config.headers ?? {})
				);
				const { method, url } = request;

				logIfEnabled(
					this.createLog(
						MCPConnectionPhase.INITIALIZING,
						`HTTP ${method} ${url}`,
						MCPLogLevel.INFO,
						{
							serverName,
							request
						}
					)
				);

				try {
					const response = await fetch(input, {
						...baseInit,
						...init,
						headers: requestHeaders
					});
					const durationMs = Math.round(performance.now() - startedAt);

					logIfEnabled(
						this.createLog(
							MCPConnectionPhase.INITIALIZING,
							`HTTP ${response.status} ${method} ${url} (${durationMs}ms)`,
							response.ok ? MCPLogLevel.INFO : MCPLogLevel.WARN,
							{
								response: {
									url,
									status: response.status,
									statusText: response.statusText,
									headers: sanitizeHeaders(response.headers, undefined, MCP_PARTIAL_REDACT_HEADERS),
									durationMs
								}
							}
						)
					);

					return response;
				} catch (error) {
					const durationMs = Math.round(performance.now() - startedAt);

					logIfEnabled(
						this.createLog(
							MCPConnectionPhase.ERROR,
							`HTTP ${method} ${url} failed: ${formatDiagnosticErrorMessage(error)}`,
							MCPLogLevel.ERROR,
							{
								serverName,
								request,
								error: this.summarizeError(error),
								browser: this.getBrowserContext(targetUrl, useProxy),
								hints: this.getConnectionHints(targetUrl, config, error),
								durationMs
							}
						)
					);

					throw error;
				}
			},
			disable: () => {
				enabled = false;
			}
		};
	}

	/**
	 * Detect if an error indicates an expired/invalidated MCP session.
	 * Per MCP spec 2025-11-25: HTTP 404 means session invalidated, client MUST
	 * discard its session ID and start a new session with a fresh initialize request.
	 *
	 * @param error - The caught error to inspect
	 * @returns true if the error is a StreamableHTTP 404 (session not found)
	 */
	static isSessionExpiredError(error: unknown): boolean {
		return error instanceof StreamableHTTPError && error.code === 404;
	}

	/**
	 * Create transport based on server configuration.
	 * Supports WebSocket, StreamableHTTP (modern), and SSE (legacy) transports.
	 * When `useProxy` is enabled, routes HTTP requests through llama-server's CORS proxy.
	 *
	 * **Fallback Order:**
	 * 1. WebSocket — if explicitly configured (no CORS proxy support)
	 * 2. StreamableHTTP — default for HTTP connections
	 * 3. SSE — automatic fallback if StreamableHTTP fails
	 *
	 * @param config - Server configuration with url, transport type, proxy, and auth settings
	 * @returns Object containing the created transport and the transport type used
	 * @throws {Error} If url is missing, WebSocket + proxy combination, or all transports fail
	 */
	static createTransport(
		serverName: string,
		config: MCPServerConfig,
		onLog?: (log: MCPConnectionLog) => void
	): {
		transport: Transport;
		type: MCPTransportType;
		stopPhaseLogging: () => void;
	} {
		if (!config.url) {
			throw new Error('MCP server configuration is missing url');
		}

		const useProxy = config.useProxy ?? false;
		const requestInit: RequestInit = {};

		if (config.headers) {
			requestInit.headers = config.useProxy ? buildProxiedHeaders(config.headers) : config.headers;
		}

		if (useProxy) {
			requestInit.headers = {
				...getAuthHeaders(),
				...(requestInit.headers as Record<string, string>)
			};
		}

		if (config.credentials) {
			requestInit.credentials = config.credentials;
		}

		if (config.transport === MCPTransportType.WEBSOCKET) {
			if (useProxy) {
				throw new Error(
					'WebSocket transport is not supported when using CORS proxy. Use HTTP transport instead.'
				);
			}

			const url = new URL(config.url);

			if (import.meta.env.DEV) {
				console.log(`[MCPService] Creating WebSocket transport for ${url.href}`);
			}

			return {
				transport: new WebSocketClientTransport(url),
				type: MCPTransportType.WEBSOCKET,
				stopPhaseLogging: () => {}
			};
		}

		const url = useProxy ? buildProxiedUrl(config.url) : new URL(config.url);
		const { fetch: diagnosticFetch, disable: stopPhaseLogging } = this.createDiagnosticFetch(
			serverName,
			config,
			requestInit,
			url,
			useProxy,
			onLog
		);

		if (useProxy && import.meta.env.DEV) {
			console.log(`[MCPService] Using CORS proxy for ${config.url} -> ${url.href}`);
		}

		try {
			if (import.meta.env.DEV) {
				console.log(`[MCPService] Creating StreamableHTTP transport for ${url.href}`);
			}

			return {
				transport: new StreamableHTTPClientTransport(url, {
					requestInit,
					fetch: diagnosticFetch
				}),
				type: MCPTransportType.STREAMABLE_HTTP,
				stopPhaseLogging
			};
		} catch (httpError) {
			console.warn(`[MCPService] StreamableHTTP failed, trying SSE transport...`, httpError);

			try {
				return {
					transport: new SSEClientTransport(url, {
						requestInit,
						fetch: diagnosticFetch,
						eventSourceInit: { fetch: diagnosticFetch }
					}),
					type: MCPTransportType.SSE,
					stopPhaseLogging
				};
			} catch (sseError) {
				const httpMsg = httpError instanceof Error ? httpError.message : String(httpError);
				const sseMsg = sseError instanceof Error ? sseError.message : String(sseError);

				throw new Error(`Failed to create transport. StreamableHTTP: ${httpMsg}; SSE: ${sseMsg}`);
			}
		}
	}

	/**
	 * Extract server info from SDK Implementation type.
	 * Normalizes the SDK's server version response into our MCPServerInfo type.
	 *
	 * @param impl - Raw Implementation object from MCP SDK
	 * @returns Normalized server info or undefined if input is empty
	 */
	private static extractServerInfo(impl: Implementation | undefined): MCPServerInfo | undefined {
		if (!impl) {
			return undefined;
		}

		return {
			name: impl.name,
			version: impl.version,
			title: impl.title,
			description: impl.description,
			websiteUrl: impl.websiteUrl,
			icons: impl.icons?.map((icon: MCPResourceIcon) => ({
				src: icon.src,
				mimeType: icon.mimeType,
				sizes: icon.sizes,
				theme: icon.theme
			}))
		};
	}

	/**
	 * Connect to a single MCP server with detailed phase tracking.
	 *
	 * Performs the full MCP connection lifecycle:
	 * 1. Transport creation (with automatic fallback)
	 * 2. Client initialization and capability exchange
	 * 3. Tool discovery via `listTools`
	 *
	 * Reports progress via `onPhase` callback at each step, enabling
	 * UI progress indicators during connection.
	 *
	 * @param serverName - Display name for the server (used in logging)
	 * @param serverConfig - Server URL, transport type, proxy, and auth configuration
	 * @param clientInfo - Optional client identification (defaults to app info)
	 * @param capabilities - Optional client capability declaration
	 * @param onPhase - Optional callback for connection phase progress updates
	 * @param listChangedHandlers - Optional handlers for server-initiated list change notifications
	 * @returns Full connection object with client, transport, tools, server info, and timing
	 * @throws {Error} If transport creation or connection fails
	 */
	static async connect(
		serverName: string,
		serverConfig: MCPServerConfig,
		clientInfo?: Implementation,
		capabilities?: ClientCapabilities,
		onPhase?: MCPPhaseCallback,
		listChangedHandlers?: ListChangedHandlers
	): Promise<MCPConnection> {
		const startTime = performance.now();
		const effectiveClientInfo = clientInfo ?? DEFAULT_MCP_CONFIG.clientInfo;
		const effectiveCapabilities = capabilities ?? DEFAULT_MCP_CONFIG.capabilities;

		// Phase: Creating transport
		onPhase?.(
			MCPConnectionPhase.TRANSPORT_CREATING,
			this.createLog(
				MCPConnectionPhase.TRANSPORT_CREATING,
				`Creating transport for ${serverConfig.url}`
			)
		);

		if (import.meta.env.DEV) {
			console.log(`[MCPService][${serverName}] Creating transport...`);
		}

		const {
			transport,
			type: transportType,
			stopPhaseLogging
		} = this.createTransport(serverName, serverConfig, (log) => onPhase?.(log.phase, log));

		// Setup WebSocket reconnection handler
		if (transportType === MCPTransportType.WEBSOCKET) {
			transport.onclose = () => {
				console.log(`[MCPService][${serverName}] WebSocket closed, notifying for reconnection`);
				onPhase?.(
					MCPConnectionPhase.DISCONNECTED,
					this.createLog(MCPConnectionPhase.DISCONNECTED, 'WebSocket connection closed')
				);
			};
		}

		// Phase: Transport ready
		onPhase?.(
			MCPConnectionPhase.TRANSPORT_READY,
			this.createLog(MCPConnectionPhase.TRANSPORT_READY, `Transport ready (${transportType})`),
			{ transportType }
		);

		const client = new Client(
			{
				name: effectiveClientInfo.name,
				version: effectiveClientInfo.version ?? DEFAULT_CLIENT_VERSION
			},
			{
				capabilities: effectiveCapabilities,
				listChanged: listChangedHandlers
			}
		);

		const runtimeErrorHandler = (error: Error) => {
			console.error(`[MCPService][${serverName}] Protocol error after initialize:`, error);
		};

		client.onerror = (error) => {
			onPhase?.(
				MCPConnectionPhase.ERROR,
				this.createLog(
					MCPConnectionPhase.ERROR,
					`Protocol error: ${error.message}`,
					MCPLogLevel.ERROR,
					{
						error: this.summarizeError(error)
					}
				)
			);
		};

		// Phase: Initializing
		onPhase?.(
			MCPConnectionPhase.INITIALIZING,
			this.createLog(MCPConnectionPhase.INITIALIZING, 'Sending initialize request...')
		);

		try {
			await client.connect(transport);
			// Transport diagnostics are only for the initial handshake, not long-lived traffic.
			stopPhaseLogging();
			client.onerror = runtimeErrorHandler;
		} catch (error) {
			client.onerror = runtimeErrorHandler;
			const url =
				(serverConfig.useProxy ?? false)
					? buildProxiedUrl(serverConfig.url)
					: new URL(serverConfig.url);

			onPhase?.(
				MCPConnectionPhase.ERROR,
				this.createLog(
					MCPConnectionPhase.ERROR,
					`Connection failed during initialize: ${
						error instanceof Error ? error.message : String(error)
					}`,
					MCPLogLevel.ERROR,
					{
						error: this.summarizeError(error),
						config: {
							serverName,
							configuredUrl: serverConfig.url,
							effectiveUrl: url.href,
							transportType,
							useProxy: serverConfig.useProxy ?? false,
							headers: sanitizeHeaders(
								serverConfig.headers,
								Object.keys(serverConfig.headers ?? {}),
								MCP_PARTIAL_REDACT_HEADERS
							),
							credentials: serverConfig.credentials
						},
						browser: this.getBrowserContext(url, serverConfig.useProxy ?? false),
						hints: this.getConnectionHints(url, serverConfig, error)
					}
				)
			);

			throw error;
		}

		const serverVersion = client.getServerVersion();
		const serverCapabilities = client.getServerCapabilities();
		const instructions = client.getInstructions();
		const serverInfo = this.extractServerInfo(serverVersion);

		// Phase: Capabilities exchanged
		onPhase?.(
			MCPConnectionPhase.CAPABILITIES_EXCHANGED,
			this.createLog(
				MCPConnectionPhase.CAPABILITIES_EXCHANGED,
				'Capabilities exchanged successfully',
				MCPLogLevel.INFO,
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
			MCPConnectionPhase.LISTING_TOOLS,
			this.createLog(MCPConnectionPhase.LISTING_TOOLS, 'Listing available tools...')
		);

		console.log(`[MCPService][${serverName}] Connected, listing tools...`);
		const tools = await this.listTools({
			client,
			transport,
			tools: [],
			serverName,
			transportType,
			connectionTimeMs: 0,
			requestTimeoutMs:
				serverConfig.requestTimeoutMs ?? DEFAULT_MCP_CONFIG.requestTimeoutSeconds * 1000
		});

		const connectionTimeMs = Math.round(performance.now() - startTime);

		// Phase: Connected
		onPhase?.(
			MCPConnectionPhase.CONNECTED,
			this.createLog(
				MCPConnectionPhase.CONNECTED,
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
			connectionTimeMs,
			requestTimeoutMs:
				serverConfig.requestTimeoutMs ?? DEFAULT_MCP_CONFIG.requestTimeoutSeconds * 1000
		};
	}

	/**
	 * Disconnect from a server.
	 * Clears the `onclose` handler to prevent reconnection attempts on voluntary disconnect.
	 *
	 * @param connection - The active MCP connection to close
	 */
	static async disconnect(connection: MCPConnection): Promise<void> {
		console.log(`[MCPService][${connection.serverName}] Disconnecting...`);
		try {
			// Prevent reconnection on voluntary disconnect
			if (connection.transport.onclose) {
				connection.transport.onclose = undefined;
			}

			await connection.client.close();
		} catch (error) {
			console.warn(`[MCPService][${connection.serverName}] Error during disconnect:`, error);
		}
	}

	/**
	 * List tools from a connection.
	 * Silently returns empty array on failure (logged as warning).
	 *
	 * @param connection - The MCP connection to query
	 * @returns Array of available tools, or empty array on error
	 */
	static async listTools(connection: MCPConnection): Promise<Tool[]> {
		try {
			const result = await connection.client.listTools();

			return result.tools ?? [];
		} catch (error) {
			// Let session-expired errors propagate for reconnection handling
			if (this.isSessionExpiredError(error)) {
				throw error;
			}

			console.warn(`[MCPService][${connection.serverName}] Failed to list tools:`, error);

			return [];
		}
	}

	/**
	 * List prompts from a connection.
	 * Silently returns empty array on failure (logged as warning).
	 *
	 * @param connection - The MCP connection to query
	 * @returns Array of available prompts, or empty array on error
	 */
	static async listPrompts(connection: MCPConnection): Promise<Prompt[]> {
		try {
			const result = await connection.client.listPrompts();

			return result.prompts ?? [];
		} catch (error) {
			// Let session-expired errors propagate for reconnection handling
			if (this.isSessionExpiredError(error)) {
				throw error;
			}

			console.warn(`[MCPService][${connection.serverName}] Failed to list prompts:`, error);

			return [];
		}
	}

	/**
	 * Get a specific prompt with arguments.
	 * Unlike list operations, this throws on failure since the caller explicitly
	 * requested a specific prompt and needs to handle the error.
	 *
	 * @param connection - The MCP connection to use
	 * @param name - The prompt name to retrieve
	 * @param args - Optional key-value arguments to pass to the prompt
	 * @returns The prompt result with messages and metadata
	 * @throws {Error} If the prompt retrieval fails
	 */
	static async getPrompt(
		connection: MCPConnection,
		name: string,
		args?: Record<string, string>
	): Promise<GetPromptResult> {
		try {
			return await connection.client.getPrompt({ name, arguments: args });
		} catch (error) {
			console.error(`[MCPService][${connection.serverName}] Failed to get prompt:`, error);

			throw error;
		}
	}

	/**
	 * Execute a tool call on a connection.
	 * Supports abort signal for cancellable operations (e.g., when user stops generation).
	 * Formats the raw tool result into a string representation.
	 *
	 * @param connection - The MCP connection to execute against
	 * @param params - Tool name and arguments to execute
	 * @param signal - Optional AbortSignal for cancellation support
	 * @returns Formatted tool execution result with content string and error flag
	 * @throws {Error} If tool execution fails or is aborted
	 */
	static async callTool(
		connection: MCPConnection,
		params: ToolCallParams,
		signal?: AbortSignal
	): Promise<ToolExecutionResult> {
		throwIfAborted(signal);

		try {
			const result = await connection.client.callTool(
				{ name: params.name, arguments: params.arguments },
				undefined,
				{ signal, timeout: connection.requestTimeoutMs }
			);

			return {
				content: this.formatToolResult(result as ToolCallResult),
				isError: (result as ToolCallResult).isError ?? false
			};
		} catch (error) {
			if (isAbortError(error)) {
				throw error;
			}

			// Let session-expired errors propagate unwrapped for reconnection handling
			if (this.isSessionExpiredError(error)) {
				throw error;
			}

			const message = error instanceof Error ? error.message : String(error);

			throw new Error(
				`Tool "${params.name}" execution failed on server "${connection.serverName}": ${message}`,
				{ cause: error instanceof Error ? error : undefined }
			);
		}
	}

	/**
	 * Format tool result content items to a single string.
	 * Handles text, image (base64 data URL), and embedded resource content types.
	 *
	 * @param result - Raw tool call result from MCP SDK
	 * @returns Concatenated string representation of all content items
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
		if (content.type === MCPContentType.TEXT && content.text) {
			return content.text;
		}

		if (content.type === MCPContentType.IMAGE && content.data) {
			return createBase64DataUrl(content.mimeType ?? DEFAULT_IMAGE_MIME_TYPE, content.data);
		}

		if (content.type === MCPContentType.RESOURCE && content.resource) {
			const resource = content.resource;

			if (resource.text) return resource.text;
			if (resource.blob) return resource.blob;

			return JSON.stringify(resource);
		}

		if (content.data && content.mimeType) {
			return createBase64DataUrl(content.mimeType, content.data);
		}

		return JSON.stringify(content);
	}

	/**
	 *
	 *
	 * Completions Operations
	 *
	 *
	 */

	/**
	 * Request completion suggestions from a server.
	 * Used for autocompleting prompt arguments or resource URI templates.
	 *
	 * @param connection - The MCP connection to use
	 * @param ref - Reference to the prompt or resource template
	 * @param argument - The argument being completed (name and current value)
	 * @returns Completion result with suggested values
	 */
	static async complete(
		connection: MCPConnection,
		ref: { type: MCPRefType.PROMPT; name: string } | { type: MCPRefType.RESOURCE; uri: string },
		argument: { name: string; value: string }
	): Promise<{ values: string[]; total?: number; hasMore?: boolean } | null> {
		try {
			const result = await connection.client.complete({
				ref,
				argument
			});

			return result.completion;
		} catch (error) {
			console.error(`[MCPService] Failed to get completions:`, error);

			return null;
		}
	}

	/**
	 *
	 *
	 * Resources Operations
	 *
	 *
	 */

	/**
	 * List resources from a connection.
	 * @param connection - The MCP connection to use
	 * @param cursor - Optional pagination cursor
	 * @returns Array of available resources and optional next cursor
	 */
	static async listResources(
		connection: MCPConnection,
		cursor?: string
	): Promise<{ resources: MCPResource[]; nextCursor?: string }> {
		try {
			const result = await connection.client.listResources(cursor ? { cursor } : undefined);

			return {
				resources: (result.resources ?? []) as MCPResource[],
				nextCursor: result.nextCursor
			};
		} catch (error) {
			if (this.isSessionExpiredError(error)) {
				throw error;
			}

			console.warn(`[MCPService][${connection.serverName}] Failed to list resources:`, error);

			return { resources: [] };
		}
	}

	/**
	 * List all resources from a connection (handles pagination automatically).
	 * @param connection - The MCP connection to use
	 * @returns Array of all available resources
	 */
	static async listAllResources(connection: MCPConnection): Promise<MCPResource[]> {
		const allResources: MCPResource[] = [];
		let cursor: string | undefined;

		do {
			const result = await this.listResources(connection, cursor);
			allResources.push(...result.resources);
			cursor = result.nextCursor;
		} while (cursor);

		return allResources;
	}

	/**
	 * List resource templates from a connection.
	 * @param connection - The MCP connection to use
	 * @param cursor - Optional pagination cursor
	 * @returns Array of available resource templates and optional next cursor
	 */
	static async listResourceTemplates(
		connection: MCPConnection,
		cursor?: string
	): Promise<{ resourceTemplates: MCPResourceTemplate[]; nextCursor?: string }> {
		try {
			const result = await connection.client.listResourceTemplates(cursor ? { cursor } : undefined);

			return {
				resourceTemplates: (result.resourceTemplates ?? []) as MCPResourceTemplate[],
				nextCursor: result.nextCursor
			};
		} catch (error) {
			if (this.isSessionExpiredError(error)) {
				throw error;
			}

			console.warn(
				`[MCPService][${connection.serverName}] Failed to list resource templates:`,
				error
			);

			return { resourceTemplates: [] };
		}
	}

	/**
	 * List all resource templates from a connection (handles pagination automatically).
	 * @param connection - The MCP connection to use
	 * @returns Array of all available resource templates
	 */
	static async listAllResourceTemplates(connection: MCPConnection): Promise<MCPResourceTemplate[]> {
		const allTemplates: MCPResourceTemplate[] = [];
		let cursor: string | undefined;

		do {
			const result = await this.listResourceTemplates(connection, cursor);
			allTemplates.push(...result.resourceTemplates);
			cursor = result.nextCursor;
		} while (cursor);

		return allTemplates;
	}

	/**
	 * Read the contents of a resource.
	 * @param connection - The MCP connection to use
	 * @param uri - The URI of the resource to read
	 * @returns The resource contents
	 */
	static async readResource(
		connection: MCPConnection,
		uri: string
	): Promise<MCPReadResourceResult> {
		try {
			const result = await connection.client.readResource({ uri });

			return {
				contents: (result.contents ?? []) as MCPResourceContent[],
				_meta: result._meta
			};
		} catch (error) {
			console.error(`[MCPService][${connection.serverName}] Failed to read resource:`, error);

			throw error;
		}
	}

	/**
	 * Subscribe to updates for a resource.
	 * The server will send notifications/resources/updated when the resource changes.
	 * @param connection - The MCP connection to use
	 * @param uri - The URI of the resource to subscribe to
	 */
	static async subscribeResource(connection: MCPConnection, uri: string): Promise<void> {
		try {
			await connection.client.subscribeResource({ uri });

			console.log(`[MCPService][${connection.serverName}] Subscribed to resource: ${uri}`);
		} catch (error) {
			console.error(
				`[MCPService][${connection.serverName}] Failed to subscribe to resource:`,
				error
			);

			throw error;
		}
	}

	/**
	 * Unsubscribe from updates for a resource.
	 * @param connection - The MCP connection to use
	 * @param uri - The URI of the resource to unsubscribe from
	 */
	static async unsubscribeResource(connection: MCPConnection, uri: string): Promise<void> {
		try {
			await connection.client.unsubscribeResource({ uri });

			console.log(`[MCPService][${connection.serverName}] Unsubscribed from resource: ${uri}`);
		} catch (error) {
			console.error(
				`[MCPService][${connection.serverName}] Failed to unsubscribe from resource:`,
				error
			);

			throw error;
		}
	}

	/**
	 * Check if a connection supports resources.
	 * Per MCP spec: presence of the `resources` key (even as empty object `{}`) indicates support.
	 * Empty object means resources are supported but no sub-features (subscribe, listChanged).
	 *
	 * @param connection - The MCP connection to check
	 * @returns Whether the server declares the resources capability
	 */
	static supportsResources(connection: MCPConnection): boolean {
		// Per MCP spec: "Servers that support resources MUST declare the resources capability"
		// The presence of the key indicates support, even if it's an empty object
		return connection.serverCapabilities?.resources !== undefined;
	}

	/**
	 * Check if a connection supports resource subscriptions.
	 * @param connection - The MCP connection to check
	 * @returns Whether the server supports resource subscriptions
	 */
	static supportsResourceSubscriptions(connection: MCPConnection): boolean {
		return !!connection.serverCapabilities?.resources?.subscribe;
	}
}
