import { ToolSource } from '$lib/enums/tools.enums';

/** HTTP header carrying the working directory a tool call runs in. The server resolves relative paths against it; the model cannot override it. */
export const X_TOOL_CWD_HEADER = 'x-tool-cwd';

export const TOOL_GROUP_LABELS = {
	[ToolSource.BUILTIN]: 'Built-in',
	[ToolSource.CUSTOM]: 'JSON Schema',
	[ToolSource.FRONTEND]: 'Browser'
} as const;

export const TOOL_SERVER_LABELS = {
	[ToolSource.BUILTIN]: 'Built-in Tools',
	[ToolSource.CUSTOM]: 'Custom Tools',
	[ToolSource.FRONTEND]: 'Browser Tools'
} as const;
