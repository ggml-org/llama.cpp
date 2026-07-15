// Registry of built-in and frontend (browser) tools whose renderer
// shows a recognizable icon and friendly label inline in the chat UI.
//
// To add a new built-in tool, add an entry to BUILTIN_TOOL_UI. To give a
// tool a custom title or body renderer, add a dedicated component under
// ChatMessageToolCall/ and route it in ChatMessageToolCallBlock.svelte
// (see ChatMessageToolCallBlockGetDatetime and
// ChatMessageToolCallBlockSearchResults for prior art).

import type { Component } from 'svelte';
import {
	Braces,
	Clock,
	FilePen,
	FilePlus,
	FileSearch,
	FileText,
	SearchCode,
	Terminal
} from '@lucide/svelte';
import { BuiltInTool, ToolSource } from '$lib/enums';

export interface BuiltinToolUiEntry {
	icon: Component;
	label: string;
	source: ToolSource.BUILTIN | ToolSource.FRONTEND;
}

export const BUILTIN_TOOL_UI: Readonly<Record<BuiltInTool, BuiltinToolUiEntry>> = {
	[BuiltInTool.READ_FILE]: { icon: FileText, label: 'Read file', source: ToolSource.BUILTIN },
	[BuiltInTool.EDIT_FILE]: { icon: FilePen, label: 'Edit file', source: ToolSource.BUILTIN },
	[BuiltInTool.WRITE_FILE]: { icon: FilePlus, label: 'Write file', source: ToolSource.BUILTIN },
	[BuiltInTool.FILE_GLOB_SEARCH]: {
		icon: FileSearch,
		label: 'Search files',
		source: ToolSource.BUILTIN
	},
	[BuiltInTool.GREP_SEARCH]: {
		icon: SearchCode,
		label: 'Search in files',
		source: ToolSource.BUILTIN
	},
	[BuiltInTool.GET_DATETIME]: { icon: Clock, label: 'Current time', source: ToolSource.BUILTIN },
	[BuiltInTool.EXEC_SHELL_COMMAND]: {
		icon: Terminal,
		label: 'Run command',
		source: ToolSource.BUILTIN
	},
	[BuiltInTool.RUN_JAVASCRIPT]: {
		icon: Braces,
		label: 'Run JavaScript',
		source: ToolSource.FRONTEND
	}
} as const;

export function getBuiltinToolUi(toolName: string | undefined): BuiltinToolUiEntry | null {
	if (!toolName) return null;
	return (BUILTIN_TOOL_UI as Record<string, BuiltinToolUiEntry>)[toolName] ?? null;
}
