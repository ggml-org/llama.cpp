import { ToolSource } from '$lib/enums/tools.enums';

/** HTTP header carrying the working directory a tool call runs in. The server resolves relative paths against it; the model cannot override it. */
export const X_TOOL_CWD_HEADER = 'x-tool-cwd';

/** HTTP header asking the server to encode a tool's output differently, e.g. read_file returning base64. Not a tool parameter, so it stays out of the definition the model sees. */
export const X_RESP_TYPE_HEADER = 'x-resp-type';

/** `X_RESP_TYPE_HEADER` value that makes read_file return the raw bytes as base64 instead of text. */
export const RESP_TYPE_BASE64 = 'base64';

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
