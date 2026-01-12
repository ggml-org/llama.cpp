import type {
	ClientCapabilities as SDKClientCapabilities,
	Implementation as SDKImplementation,
	Tool,
	CallToolResult
} from '@modelcontextprotocol/sdk/types.js';

export type { Tool, CallToolResult };
export type ClientCapabilities = SDKClientCapabilities;
export type Implementation = SDKImplementation;

export type MCPTransportType = 'websocket' | 'streamable_http';

export type MCPServerConfig = {
	transport?: MCPTransportType;
	url: string;
	protocols?: string | string[];
	headers?: Record<string, string>;
	credentials?: RequestCredentials;
	handshakeTimeoutMs?: number;
	requestTimeoutMs?: number;
	capabilities?: ClientCapabilities;
	sessionId?: string;
};

export type MCPClientConfig = {
	servers: Record<string, MCPServerConfig>;
	protocolVersion?: string;
	capabilities?: ClientCapabilities;
	clientInfo?: Implementation;
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

export type MCPServerSettingsEntry = {
	id: string;
	enabled: boolean;
	url: string;
	requestTimeoutSeconds: number;
	headers?: string;
	name?: string;
	iconUrl?: string;
};

export interface MCPHostManagerConfig {
	servers: MCPClientConfig['servers'];
	clientInfo?: Implementation;
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

export type McpServerUsageStats = Record<string, number>;

export interface MCPServerConnectionConfig {
	name: string;
	server: MCPServerConfig;
	clientInfo?: Implementation;
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
