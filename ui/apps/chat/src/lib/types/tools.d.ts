import type { OpenAIToolDefinition } from './mcp';
import type { ToolCallSummaryKind, ToolSource } from '#lib/enums/index.ts';
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
	/** Undefined until the call streams a path; content renders without it. */
	fileName?: string;
	filePath?: string;
	language: string;
	/** Undefined until the args carry a content string; an empty write is ''. */
	content?: string;
	bytesWritten?: number;
	resultMessage?: string;
	errorMessage?: string;
}

/** Everything the write_file block title and status pill show; the full meta
 *  ( with the embedded file content ) stays body-only so collapsed blocks
 *  never parse the content blob. */
export interface WriteFileTitleMeta {
	fileName?: string;
	filePath?: string;
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
	/** Undefined until the call streams a path; the edits render without it. */
	fileName?: string;
	filePath?: string;
	edits: EditFileEdit[];
	resultMessage?: string;
	editsApplied?: number;
	errorMessage?: string;
}

/** Everything the edit_file block title and status pill show; the full meta
 *  ( with the embedded edit strings ) stays body-only so collapsed blocks
 *  never parse the args blob. */
export interface EditFileTitleMeta {
	fileName?: string;
	filePath?: string;
	resultMessage?: string;
	editsApplied?: number;
	errorMessage?: string;
}

/** How a tool call reads in the chat trace; see TOOL_CALL_PHRASING. */
export interface ToolCallPhrasing {
	/** Verb once the call is done, e.g. "Read". */
	past: string;
	/** Verb while the call runs, e.g. "Reading". */
	present: string;
	/** Words after the verb when the call has no target, e.g. "a file". */
	object: string;
	/** Words between the verb and the target, e.g. "for". */
	targetObject?: string;
	/** Show the target in quotes, for search terms. */
	quoteTarget?: boolean;
	summary: ToolCallSummaryKind;
}

/** A tool call as one plain sentence: label, then an optional target. */
export interface ToolCallDescription {
	/** Verb and object, e.g. "Read a file" or "Searched for". */
	label: string;
	/** What the call acted on, e.g. a file name or a search term. */
	target?: string;
	quoted: boolean;
	summary: ToolCallSummaryKind;
	/** How many things the call counts as in the group summary. */
	count: number;
	/** True when the label is the model's own description of the call. */
	described?: boolean;
}
