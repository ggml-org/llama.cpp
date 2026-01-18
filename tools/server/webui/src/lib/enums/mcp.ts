/**
 * Connection lifecycle phases for MCP protocol
 */
export enum MCPConnectionPhase {
	Idle = 'idle',
	TransportCreating = 'transport_creating',
	TransportReady = 'transport_ready',
	Initializing = 'initializing',
	CapabilitiesExchanged = 'capabilities_exchanged',
	ListingTools = 'listing_tools',
	Connected = 'connected',
	Error = 'error',
	Disconnected = 'disconnected'
}

/**
 * Log level for connection events
 */
export enum MCPLogLevel {
	Info = 'info',
	Warn = 'warn',
	Error = 'error'
}

/**
 * Transport types for MCP connections
 */
export enum MCPTransportType {
	Websocket = 'websocket',
	StreamableHttp = 'streamable_http',
	SSE = 'sse'
}

/**
 * Health check status for MCP servers
 */
export enum HealthCheckStatus {
	Idle = 'idle',
	Connecting = 'connecting',
	Success = 'success',
	Error = 'error'
}
