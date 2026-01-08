// New architecture exports
export { MCPHostManager } from './host-manager';
export type { MCPHostManagerConfig, OpenAIToolDefinition, ServerStatus } from './host-manager';
export { MCPServerConnection } from './server-connection';
export type {
	MCPServerConnectionConfig,
	ToolCallParams,
	ToolExecutionResult
} from './server-connection';

// Errors
export { MCPError } from '$lib/errors';

// Types
export type {
	MCPClientConfig,
	MCPServerConfig,
	MCPToolCall,
	MCPServerSettingsEntry
} from '$lib/types/mcp';
