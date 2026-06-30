/**
 * Connection lifecycle phases for MCP protocol
 */
export enum MCPConnectionPhase {
	IDLE = 'idle',
	TRANSPORT_CREATING = 'transport_creating',
	TRANSPORT_READY = 'transport_ready',
	INITIALIZING = 'initializing',
	CAPABILITIES_EXCHANGED = 'capabilities_exchanged',
	LISTING_TOOLS = 'listing_tools',
	CONNECTED = 'connected',
	ERROR = 'error',
	DISCONNECTED = 'disconnected'
}

/**
 * Log level for connection events
 */
export enum MCPLogLevel {
	INFO = 'info',
	WARN = 'warn',
	ERROR = 'error'
}

/**
 * Transport types for MCP connections
 */
export enum MCPTransportType {
	WEBSOCKET = 'websocket',
	STREAMABLE_HTTP = 'streamable_http',
	SSE = 'sse'
}

/**
 * Health check status for MCP servers
 */
export enum HealthCheckStatus {
	IDLE = 'idle',
	CONNECTING = 'connecting',
	SUCCESS = 'success',
	ERROR = 'error'
}

/**
 * Content types for MCP tool results
 */
export enum MCPContentType {
	TEXT = 'text',
	IMAGE = 'image',
	RESOURCE = 'resource'
}

/**
 * JSON Schema types used in MCP tool definitions
 */
export enum JsonSchemaType {
	OBJECT = 'object',
	STRING = 'string',
	NUMBER = 'number'
}

/**
 * Reference types for MCP completions
 */
export enum MCPRefType {
	PROMPT = 'ref/prompt',
	RESOURCE = 'ref/resource'
}

/**
 * Prefix used to encode MCP prompt identity inside a SKILL
 * extra's skillId (`mcp:<serverName>:<promptName>`). System messages
 * created from MCP prompts carry this so the renderer can recover the
 * original MCP server/prompt without an extra attachment.
 */
export enum MCPPromptIdPrefix {
	PROMPT = 'mcp:'
}
