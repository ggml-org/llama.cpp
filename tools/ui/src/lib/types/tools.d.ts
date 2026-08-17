import type { OpenAIToolDefinition } from './mcp';
import type { ToolSource } from '$lib/enums';
import type { Component } from 'svelte';

/**
 * UI metadata for a server or browser tool, keyed by its `BuiltInTool` id.
 */
export interface ToolUiEntry {
	icon: Component;
	label: string;
	source: ToolSource.SERVER | ToolSource.BROWSER;
}

export interface ToolEntry {
	source: ToolSource;
	/** For MCP tools, the server display name (used for UI grouping) */
	serverName?: string;
	/** For MCP tools, the server ID (used for permission keys) */
	serverId?: string;
	/** Stable selection identity: server:name, mcp-<serverId>:name, mcp:name, custom:name */
	key: string;
	definition: OpenAIToolDefinition;
}

export interface ToolGroup {
	source: ToolSource;
	/** Stable identity for keyed rendering and toggles, unique per group */
	key: string;
	label: string;
	/** For MCP groups, the server ID */
	serverId?: string;
	tools: ToolEntry[];
}
