// New architecture exports
export { MCPHostManager } from './host-manager';
export type { MCPHostManagerConfig, OpenAIToolDefinition, ServerStatus } from './host-manager';
export { MCPServerConnection } from './server-connection';
export type {
	MCPServerConnectionConfig,
	ToolCallParams,
	ToolExecutionResult
} from './server-connection';

// Legacy client export (deprecated - use MCPHostManager instead)
export { MCPClient } from './client';

// Types
export { MCPError } from '$lib/types/mcp';
export type {
	MCPClientConfig,
	MCPServerConfig,
	MCPToolCall,
	MCPServerSettingsEntry
} from '$lib/types/mcp';
