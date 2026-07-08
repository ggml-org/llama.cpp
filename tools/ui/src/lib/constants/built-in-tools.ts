// Registry of built-in and frontend (browser) tools whose renderer
// shows a recognizable icon and friendly label inline in the chat UI.
//
// To add a new built-in tool, add an entry here. To give a tool a
// custom title (with file name, command, etc.) or a custom body
// renderer, extend the snippets and body branches in
// ChatMessageToolCallBlock.svelte — those are opt-in per tool.

import type { Component } from 'svelte';
import {
	Braces,
	Clock,
	FilePen,
	FilePlus,
	FileSearch,
	FileText,
	GitMerge,
	SearchCode,
	Terminal
} from '@lucide/svelte';
import { ToolSource } from '$lib/enums';

export interface BuiltinToolUiEntry {
	icon: Component;
	label: string;
	source: ToolSource.BUILTIN | ToolSource.FRONTEND;
}

export const BUILTIN_TOOL_UI: Readonly<Record<string, BuiltinToolUiEntry>> = {
	read_file: { icon: FileText, label: 'Read file', source: ToolSource.BUILTIN },
	edit_file: { icon: FilePen, label: 'Edit file', source: ToolSource.BUILTIN },
	write_file: { icon: FilePlus, label: 'Write file', source: ToolSource.BUILTIN },
	file_glob_search: {
		icon: FileSearch,
		label: 'Search files',
		source: ToolSource.BUILTIN
	},
	grep_search: {
		icon: SearchCode,
		label: 'Search in files',
		source: ToolSource.BUILTIN
	},
	apply_diff: { icon: GitMerge, label: 'Apply diff', source: ToolSource.BUILTIN },
	get_datetime: { icon: Clock, label: 'Current time', source: ToolSource.BUILTIN },
	exec_shell_command: {
		icon: Terminal,
		label: 'Run command',
		source: ToolSource.BUILTIN
	},
	run_javascript: {
		icon: Braces,
		label: 'Run JavaScript',
		source: ToolSource.FRONTEND
	}
} as const;

export function getBuiltinToolUi(toolName: string | undefined): BuiltinToolUiEntry | null {
	if (!toolName) return null;
	return BUILTIN_TOOL_UI[toolName] ?? null;
}
