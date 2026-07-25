export enum ToolSource {
	BUILTIN = 'builtin',
	MCP = 'mcp',
	CUSTOM = 'custom',
	FRONTEND = 'frontend'
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

/**
 * Wire-format identifiers for built-in and frontend tools. The string
 * value matches what the model emits in tool call names, so comparing
 * against `BuiltInTool.READ_FILE` is equivalent to comparing against the
 * raw `'read_file'` literal - the enum just keeps the two in lock-step
 * and gives TypeScript a single source of truth for autocomplete / rename
 * support.
 */
export enum BuiltInTool {
	READ_FILE = 'read_file',
	EDIT_FILE = 'edit_file',
	WRITE_FILE = 'write_file',
	GET_DATETIME = 'get_datetime',
	FILE_GLOB_SEARCH = 'file_glob_search',
	GREP_SEARCH = 'grep_search',
	EXEC_SHELL_COMMAND = 'exec_shell_command',
	RUN_JAVASCRIPT = 'run_javascript'
}
