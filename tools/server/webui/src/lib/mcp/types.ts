export type JsonRpcId = number | string;

export type JsonRpcRequest = {
	jsonrpc: '2.0';
	id: JsonRpcId;
	method: string;
	params?: Record<string, unknown>;
};

export type JsonRpcNotification = {
	jsonrpc: '2.0';
	method: string;
	params?: Record<string, unknown>;
};

export type JsonRpcError = {
	code: number;
	message: string;
	data?: unknown;
};

export type JsonRpcResponse = {
	jsonrpc: '2.0';
	id: JsonRpcId;
	result?: Record<string, unknown>;
	error?: JsonRpcError;
};

export type JsonRpcMessage = JsonRpcRequest | JsonRpcResponse | JsonRpcNotification;

export class MCPError extends Error {
	code: number;
	data?: unknown;

	constructor(message: string, code: number, data?: unknown) {
		super(message);
		this.name = 'MCPError';
		this.code = code;
		this.data = data;
	}
}

export type MCPToolInputSchema = Record<string, unknown>;

export type MCPToolDefinition = {
	name: string;
	description?: string;
	inputSchema?: MCPToolInputSchema;
};

export type MCPServerCapabilities = Record<string, unknown>;

export type MCPClientCapabilities = Record<string, unknown>;

export type MCPTransportType = 'websocket' | 'streamable_http';

export type MCPServerConfig = {
	/** MCP transport type. Defaults to `streamable_http`. */
	transport?: MCPTransportType;
	/** Remote MCP endpoint URL. */
	url: string;
	/** Optional WebSocket subprotocol(s). */
	protocols?: string | string[];
	/** Optional HTTP headers for environments that support them. */
	headers?: Record<string, string>;
	/** Optional credentials policy for fetch-based transports. */
	credentials?: RequestCredentials;
	/** Optional handshake timeout override (ms). */
	handshakeTimeoutMs?: number;
	/** Optional per-server request timeout override (ms). */
	requestTimeoutMs?: number;
	/** Optional per-server capability overrides. */
	capabilities?: MCPClientCapabilities;
	/** Optional pre-negotiated session identifier for Streamable HTTP transport. */
	sessionId?: string;
};

export type MCPClientInfo = {
	name: string;
	version?: string;
};

export type MCPClientConfig = {
	servers: Record<string, MCPServerConfig>;
	/** Defaults to `2025-06-18`. */
	protocolVersion?: string;
	/** Default capabilities advertised during initialize. */
	capabilities?: MCPClientCapabilities;
	/** Custom client info to advertise. */
	clientInfo?: MCPClientInfo;
	/** Request timeout when waiting for MCP responses (ms). Default: 30_000. */
	requestTimeoutMs?: number;
};

export type MCPToolCallArguments = Record<string, unknown>;

export type MCPToolCall = {
	id: string;
	function: {
		name: string;
		arguments: string | MCPToolCallArguments;
	};
};

export type MCPToolResultContent =
	| string
	| {
			type: 'text';
			text: string;
	  }
	| {
			type: 'image';
			data: string;
			mimeType?: string;
	  }
	| {
			type: 'resource';
			resource: Record<string, unknown>;
	  };

export type MCPToolsCallResult = {
	content?: MCPToolResultContent | MCPToolResultContent[];
	result?: unknown;
};
