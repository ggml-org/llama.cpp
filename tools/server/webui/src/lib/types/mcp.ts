// Re-export SDK types that we use
import type {
	ClientCapabilities as SDKClientCapabilities,
	Implementation as SDKImplementation,
	Tool,
	CallToolResult
} from '@modelcontextprotocol/sdk/types.js';

export type { Tool, CallToolResult };
export type ClientCapabilities = SDKClientCapabilities;
export type Implementation = SDKImplementation;

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
	capabilities?: ClientCapabilities;
	/** Optional pre-negotiated session identifier for Streamable HTTP transport. */
	sessionId?: string;
};

export type MCPClientConfig = {
	servers: Record<string, MCPServerConfig>;
	/** Defaults to `2025-06-18`. */
	protocolVersion?: string;
	/** Default capabilities advertised during initialize. */
	capabilities?: ClientCapabilities;
	/** Custom client info to advertise. */
	clientInfo?: Implementation;
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

/**
 * Raw MCP server configuration entry stored in settings.
 */
export type MCPServerSettingsEntry = {
	id: string;
	enabled: boolean;
	url: string;
	requestTimeoutSeconds: number;
};
