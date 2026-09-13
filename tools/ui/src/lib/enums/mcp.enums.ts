/**
 * Connection lifecycle phases for MCP protocol
 */
export enum MCPConnectionPhase {
	CAPABILITIES_EXCHANGED = 'capabilities_exchanged',
	CONNECTED = 'connected',
	DISCONNECTED = 'disconnected',
	ERROR = 'error',
	IDLE = 'idle',
	INITIALIZING = 'initializing',
	LISTING_TOOLS = 'listing_tools',
	TRANSPORT_CREATING = 'transport_creating',
	TRANSPORT_READY = 'transport_ready'
}

/**
 * Log level for connection events
 */
export enum MCPLogLevel {
	ERROR = 'error',
	INFO = 'info',
	WARN = 'warn'
}

/**
 * Transport types for MCP connections
 */
export enum MCPTransportType {
	SSE = 'sse',
	STREAMABLE_HTTP = 'streamable_http',
	WEBSOCKET = 'websocket'
}

/**
 * Health check status for MCP servers
 */
export enum HealthCheckStatus {
	CONNECTING = 'connecting',
	ERROR = 'error',
	IDLE = 'idle',
	SUCCESS = 'success'
}

/**
 * Content types for MCP tool results
 */
export enum MCPContentType {
	IMAGE = 'image',
	RESOURCE = 'resource',
	TEXT = 'text'
}

/**
 * JSON Schema types used in MCP tool definitions
 */
export enum JsonSchemaType {
	NUMBER = 'number',
	OBJECT = 'object',
	STRING = 'string'
}

/**
 * Reference types for MCP completions
 */
export enum MCPRefType {
	PROMPT = 'ref/prompt',
	RESOURCE = 'ref/resource'
}
