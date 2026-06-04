import type { ToolSource } from '$lib/enums';
import type { OpenAIToolDefinition } from './mcp';

export interface ToolEntry {
	source: ToolSource;
	/** For MCP tools, the server display name (used for UI grouping) */
	serverName?: string;
	/** For MCP tools, the server ID (used for permission keys) */
	serverId?: string;
	/** Stable selection identity: builtin:name, mcp-<serverId>:name, mcp:name, custom:name */
	key: string;
	definition: OpenAIToolDefinition;
}

export interface ToolGroup {
	source: ToolSource;
	label: string;
	/** For MCP groups, the server ID */
	serverId?: string;
	tools: ToolEntry[];
}
