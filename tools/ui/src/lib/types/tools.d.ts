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

/**
 * Settings-only metadata for one model-facing Skills adapter (`read_skill` /
 * `list_skill`), owned by the centralized Skills settings registry. Skills
 * are NOT ordinary model tools: these entries exist solely to render the
 * persistent exposure toggles in the Chat tool settings tab and must never
 * enter `allTools`, `toolGroups`, or `getEnabledToolsForLLM()`.
 */
export interface SkillToolSetting {
	/** Stable local selection key (e.g. `skill:read_skill`), distinct from generic builtin/MCP/custom keys. */
	key: string;
	/** Model-facing tool name. */
	toolName: string;
	/** Display label for the settings tab row. */
	label: string;
	/** Display description for the settings tab row. */
	description: string;
	/** Icon for the settings tab row. */
	icon: Component;
	/** Settings-only static definition; never sent to the model. */
	definition: OpenAIToolDefinition;
}
