import type { MCPConnectionPhase, MCPLogLevel } from '$lib/enums/mcp';
import type {
	Client,
	ClientCapabilities as SDKClientCapabilities,
	ServerCapabilities as SDKServerCapabilities,
	Implementation as SDKImplementation,
	Tool,
	CallToolResult,
	Prompt,
	GetPromptResult,
	PromptMessage,
	Transport
} from '@modelcontextprotocol/sdk';

export type { Tool, CallToolResult, Prompt, GetPromptResult, PromptMessage };
export type ClientCapabilities = SDKClientCapabilities;
export type ServerCapabilities = SDKServerCapabilities;
export type Implementation = SDKImplementation;

/**
 * Log entry for connection events
 */
export interface MCPConnectionLog {
	timestamp: Date;
	phase: MCPConnectionPhase;
	message: string;
	details?: unknown;
	level: MCPLogLevel;
}

/**
 * Server information returned after initialization
 */
export interface MCPServerInfo {
	name: string;
	version: string;
	title?: string;
	description?: string;
	websiteUrl?: string;
	icons?: Array<{ src: string; mimeType?: string; sizes?: string[] }>;
}

/**
 * Detailed capabilities information
 */
export interface MCPCapabilitiesInfo {
	server: {
		tools?: { listChanged?: boolean };
		prompts?: { listChanged?: boolean };
		resources?: { subscribe?: boolean; listChanged?: boolean };
		logging?: boolean;
		completions?: boolean;
		tasks?: boolean;
	};
	client: {
		roots?: { listChanged?: boolean };
		sampling?: boolean;
		elicitation?: { form?: boolean; url?: boolean };
		tasks?: boolean;
	};
}

/**
 * Tool information for display
 */
export interface MCPToolInfo {
	name: string;
	description?: string;
	title?: string;
}

/**
 * Prompt information for display
 */
export interface MCPPromptInfo {
	name: string;
	description?: string;
	title?: string;
	serverName: string;
	arguments?: Array<{
		name: string;
		description?: string;
		required?: boolean;
	}>;
}

/**
 * Full connection details for visualization
 */
export interface MCPConnectionDetails {
	phase: MCPConnectionPhase;
	transportType?: MCPTransportType;
	protocolVersion?: string;
	serverInfo?: MCPServerInfo;
	capabilities?: MCPCapabilitiesInfo;
	instructions?: string;
	tools: MCPToolInfo[];
	connectionTimeMs?: number;
	error?: string;
	logs: MCPConnectionLog[];
}

/**
 * Callback for connection phase changes
 */
export type MCPPhaseCallback = (
	phase: MCPConnectionPhase,
	log: MCPConnectionLog,
	details?: {
		transportType?: MCPTransportType;
		serverInfo?: MCPServerInfo;
		serverCapabilities?: ServerCapabilities;
		clientCapabilities?: ClientCapabilities;
		protocolVersion?: string;
		instructions?: string;
	}
) => void;

/**
 * Represents an active MCP server connection.
 * Returned by MCPService.connect() and used for subsequent operations.
 */
export interface MCPConnection {
	client: Client;
	transport: Transport;
	tools: Tool[];
	serverName: string;
	transportType: MCPTransportType;
	serverInfo?: MCPServerInfo;
	serverCapabilities?: ServerCapabilities;
	clientCapabilities?: ClientCapabilities;
	protocolVersion?: string;
	instructions?: string;
	connectionTimeMs: number;
}

/**
 * Extended health check state with detailed connection info
 */
export type HealthCheckState =
	| { status: import('$lib/enums/mcp').HealthCheckStatus.IDLE }
	| {
			status: import('$lib/enums/mcp').HealthCheckStatus.CONNECTING;
			phase: MCPConnectionPhase;
			logs: MCPConnectionLog[];
	  }
	| {
			status: import('$lib/enums/mcp').HealthCheckStatus.ERROR;
			message: string;
			phase?: MCPConnectionPhase;
			logs: MCPConnectionLog[];
	  }
	| {
			status: import('$lib/enums/mcp').HealthCheckStatus.SUCCESS;
			tools: MCPToolInfo[];
			serverInfo?: MCPServerInfo;
			capabilities?: MCPCapabilitiesInfo;
			transportType?: MCPTransportType;
			protocolVersion?: string;
			instructions?: string;
			connectionTimeMs?: number;
			logs: MCPConnectionLog[];
	  };

/**
 * Health check parameters
 */
export interface HealthCheckParams {
	id: string;
	url: string;
	requestTimeoutSeconds: number;
	headers?: string;
}

export type MCPServerConfig = {
	transport?: MCPTransportType;
	url: string;
	protocols?: string | string[];
	headers?: Record<string, string>;
	credentials?: RequestCredentials;
	handshakeTimeoutMs?: number;
	requestTimeoutMs?: number;
	capabilities?: ClientCapabilities;
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
