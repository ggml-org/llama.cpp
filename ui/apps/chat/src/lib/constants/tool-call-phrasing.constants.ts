// Plain-language phrasing for tool calls in the chat trace: the line each
// call reads as ("Read a file", "Searched for 'issue'"), the live
// description while it runs ("Reading package.json") and the counts in a
// group summary ("Ran 3 commands, read 2 files").

import { BuiltInTool, ToolCallSummaryKind } from '#lib/enums/index.ts';
import type { ToolCallPhrasing } from '#lib/types/index.ts';

/**
 * A tool that loads tools on demand, and the `select:` prefix its query uses
 * to name tools to load rather than search for them. The UI reads the prefix
 * to count them; the convention lives with that tool.
 */
export const TOOL_SEARCH_PHRASING_NAME = 'tool_search';
export const TOOL_SEARCH_PHRASING_SELECT_PREFIX = 'select:';

/** Longest target shown after the verb before it is cut with "...". */
export const TOOL_CALL_TARGET_MAX_CHARS = 60;

/** Fewest consecutive tool calls that collapse into one group row. */
export const TOOL_CALL_GROUP_MIN_CALLS = 2;

/** Live title of a tool group while the model thinks inside it. */
export const TOOL_CALL_THINKING_LABEL = 'Thinking';

/** Status shown while a reply waits on the server and no finer status is known. */
export const REPLY_STATUS_FALLBACK = 'Processing...';

/** Args a generic (MCP or unknown) tool call is described by, first match wins. */
export const TOOL_CALL_MAIN_ARG_KEYS: readonly string[] = [
	'query',
	'q',
	'path',
	'file_path',
	'filePath',
	'url',
	'name',
	'command',
	'pattern',
	'prompt',
	'text',
	'id'
];

/** Args that count as "simple" stay below this length and on one line. */
export const TOOL_CALL_SIMPLE_ARG_MAX_CHARS = 200;

export const TOOL_CALL_PHRASING: Readonly<Record<string, ToolCallPhrasing>> = {
	// its own row reads "Current time is ...", so this entry serves group titles
	[BuiltInTool.BROWSER_GET_DATETIME]: {
		object: 'the time',
		past: 'Checked',
		present: 'Checking',
		summary: ToolCallSummaryKind.OTHER
	},
	[BuiltInTool.BROWSER_READ_MEDIA]: {
		object: 'media',
		past: 'Viewed',
		present: 'Viewing',
		summary: ToolCallSummaryKind.VIEW
	},
	[BuiltInTool.BROWSER_RUN_JAVASCRIPT]: {
		object: 'JavaScript',
		past: 'Ran',
		present: 'Running',
		summary: ToolCallSummaryKind.SCRIPT
	},
	[BuiltInTool.BROWSER_SEARCH_CHATS]: {
		object: 'chats',
		past: 'Searched',
		present: 'Searching',
		quoteTarget: true,
		summary: ToolCallSummaryKind.SEARCH,
		targetObject: 'chats for'
	},
	[BuiltInTool.SERVER_EDIT_FILE]: {
		object: 'a file',
		past: 'Edited',
		present: 'Editing',
		summary: ToolCallSummaryKind.EDIT
	},
	[BuiltInTool.SERVER_EXEC_SHELL_COMMAND]: {
		object: 'a command',
		past: 'Ran',
		present: 'Running',
		summary: ToolCallSummaryKind.COMMAND
	},
	[BuiltInTool.SERVER_FILE_GLOB_SEARCH]: {
		object: 'files',
		past: 'Listed',
		present: 'Listing',
		quoteTarget: true,
		summary: ToolCallSummaryKind.SEARCH,
		targetObject: 'files matching'
	},
	[BuiltInTool.SERVER_GET_INFO]: {
		object: 'runtime info',
		past: 'Checked',
		present: 'Checking',
		summary: ToolCallSummaryKind.OTHER
	},
	[BuiltInTool.SERVER_GREP_SEARCH]: {
		object: 'files',
		past: 'Searched',
		present: 'Searching',
		quoteTarget: true,
		summary: ToolCallSummaryKind.SEARCH,
		targetObject: 'for'
	},
	[BuiltInTool.SERVER_READ_FILE]: {
		object: 'a file',
		past: 'Read',
		present: 'Reading',
		summary: ToolCallSummaryKind.READ
	},
	[BuiltInTool.SERVER_WRITE_FILE]: {
		object: 'a file',
		past: 'Wrote',
		present: 'Writing',
		summary: ToolCallSummaryKind.WRITE
	}
};

export const TOOL_SEARCH_LOAD_PHRASING: ToolCallPhrasing = {
	object: 'tools',
	past: 'Loaded',
	present: 'Loading',
	summary: ToolCallSummaryKind.LOAD_TOOLS,
	targetObject: 'tools'
};

export const TOOL_SEARCH_QUERY_PHRASING: ToolCallPhrasing = {
	object: 'tools',
	past: 'Searched',
	present: 'Searching',
	quoteTarget: true,
	summary: ToolCallSummaryKind.SEARCH,
	targetObject: 'tools for'
};

export const WEB_SEARCH_PHRASING: ToolCallPhrasing = {
	object: 'the web',
	past: 'Searched',
	present: 'Searching',
	quoteTarget: true,
	summary: ToolCallSummaryKind.SEARCH,
	targetObject: 'the web for'
};

function counted(n: number, singular: string, plural: string): string {
	return `${n} ${n === 1 ? singular : plural}`;
}

/** Group summary clause per kind, in lower case; the summary capitalizes the first one. */
export const TOOL_CALL_SUMMARY: Readonly<Record<ToolCallSummaryKind, (n: number) => string>> = {
	[ToolCallSummaryKind.COMMAND]: (n) => `ran ${counted(n, 'command', 'commands')}`,
	[ToolCallSummaryKind.EDIT]: (n) => `edited ${counted(n, 'file', 'files')}`,
	[ToolCallSummaryKind.LOAD_TOOLS]: (n) => `loaded ${counted(n, 'tool', 'tools')}`,
	[ToolCallSummaryKind.OTHER]: (n) => `used ${counted(n, 'tool', 'tools')}`,
	[ToolCallSummaryKind.READ]: (n) => `read ${counted(n, 'file', 'files')}`,
	[ToolCallSummaryKind.SCRIPT]: (n) => `ran ${counted(n, 'script', 'scripts')}`,
	[ToolCallSummaryKind.SEARCH]: (n) => (n === 1 ? 'searched once' : `searched ${n} times`),
	[ToolCallSummaryKind.VIEW]: (n) => `viewed ${counted(n, 'media file', 'media files')}`,
	[ToolCallSummaryKind.WRITE]: (n) => `wrote ${counted(n, 'file', 'files')}`
};

/**
 * Arg a call's line shows by default when the call has it: tools that want a
 * readable line declare it in their own schema, and the line reads it as the
 * model wrote it. Settings can pick another arg, globally or per tool.
 */
export const TOOL_CALL_SHOWN_ARG_DEFAULT = 'description';

/** Per-tool choice that turns the shown arg off and keeps the sentence. */
export const TOOL_CALL_SHOWN_ARG_NONE = '';
