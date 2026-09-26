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

export interface WriteFileMeta {
	fileName: string;
	filePath: string;
	language: string;
	content: string;
	bytesWritten?: number;
	resultMessage?: string;
	errorMessage?: string;
}

/** Everything the write_file block title and status pill show; the full meta
 *  ( with the embedded file content ) stays body-only so collapsed blocks
 *  never parse the content blob. */
export interface WriteFileTitleMeta {
	fileName: string;
	filePath: string;
	language: string;
	bytesWritten?: number;
	resultMessage?: string;
	errorMessage?: string;
}

export interface EditFileEdit {
	oldText: string;
	newText: string;
}

export interface EditFileMeta {
	fileName: string;
	filePath: string;
	edits: EditFileEdit[];
	resultMessage?: string;
	editsApplied?: number;
	errorMessage?: string;
}

/** Everything the edit_file block title and status pill show; the full meta
 *  ( with the embedded edit strings ) stays body-only so collapsed blocks
 *  never parse the args blob. */
export interface EditFileTitleMeta {
	fileName: string;
	filePath: string;
	resultMessage?: string;
	editsApplied?: number;
	errorMessage?: string;
}
