import { ToolSource } from '$lib/enums/tools.enums';

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
