export enum ToolSource {
	BUILTIN = 'builtin',
	MCP = 'mcp',
	CUSTOM = 'custom'
}

export enum ToolPermissionDecision {
	ALWAYS = 'always',
	ALWAYS_SERVER = 'always_server',
	ONCE = 'once',
	DENY = 'deny'
}

export enum ToolResponseField {
	PLAIN_TEXT = 'plain_text_response',
	ERROR = 'error'
}
