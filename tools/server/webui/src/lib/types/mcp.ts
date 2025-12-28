// [AI] MCP type definitions for Model Context Protocol integration
import type { ApiChatCompletionTool } from './api';

/**
 * MCP Server configuration
 */
export interface MCPServer {
	/** Unique identifier for the server */
	id: string;
	/** Display name */
	name: string;
	/** Command to execute (e.g., "python", "node") */
	command: string;
	/** Arguments to pass (e.g., ["path/to/server.py"]) */
	args: string[];
	/** Environment variables */
	env?: Record<string, string>;
	/** Whether the server is currently enabled */
	enabled: boolean;
	/** Current connection status */
	status: MCPServerStatus;
}

export enum MCPServerStatus {
	STOPPED = 'stopped',
	STARTING = 'starting',
	RUNNING = 'running',
	ERROR = 'error'
}

/**
 * MCP Tool (converted from MCP tool schema to OpenAI format)
 */
export interface MCPTool extends ApiChatCompletionTool {
	/** Server ID that provides this tool */
	serverId: string;
}

/**
 * MCP Tool execution request
 */
export interface MCPToolExecutionRequest {
	serverId: string;
	toolName: string;
	arguments: Record<string, unknown>;
}

/**
 * MCP Tool execution result
 */
export interface MCPToolExecutionResult {
	success: boolean;
	result?: unknown;
	error?: string;
}

/**
 * MCP Server list response from backend
 */
export interface MCPServerListResponse {
	servers: MCPServer[];
}

/**
 * MCP Tools list response from backend
 */
export interface MCPToolsListResponse {
	tools: MCPTool[];
}
