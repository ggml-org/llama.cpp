// Phrasing of Llama Chat's own tools in the chat trace: the browser tools, web
// search and tool_search. Llama Harness phrases llama-server's built-in tools
// and any other tool (see createTracePhraser).

import { type ToolCallPhrasing, ToolCallSummaryKind } from '@ggml-org/llama-harness';
import { BuiltInTool } from '#lib/enums/index.ts';

/**
 * A tool that loads tools on demand, and the `select:` prefix its query uses
 * to name tools to load rather than search for them. The UI reads the prefix
 * to count them; the convention lives with that tool.
 */
export const TOOL_SEARCH_PHRASING_NAME = 'tool_search';
export const TOOL_SEARCH_PHRASING_SELECT_PREFIX = 'select:';

/** Fewest consecutive tool calls that collapse into one group row. */
export const TOOL_CALL_GROUP_MIN_CALLS = 2;

/** Status shown while a reply waits on the server and no finer status is known. */
export const REPLY_STATUS_FALLBACK = 'Processing...';

/** Llama Chat's browser tools; llama-server's tools are phrased by Llama Harness. */
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

/** Per-tool choice that turns the shown arg off and keeps the sentence. */
export const TOOL_CALL_SHOWN_ARG_NONE = '';

/**
 * Picks of a tool's shown argument that are not an argument's name: the tool follows the global
 * setting, or shows none.
 */
export const TOOL_SHOWN_ARG_PICK = {
	DEFAULT: '__default__',
	NONE: '__none__'
} as const;

/** Text of the Tools settings, on the desktop's table and the phone sheet's pages. */
export const TOOL_SETTINGS_TEXT = {
	ALL_TOOLS: 'All tools',
	ALWAYS_ALLOW: 'Always allow',
	ALWAYS_ALLOW_NOTE: 'An always allowed tool runs without asking first.',
	ALWAYS_ALLOWED: 'Always allowed',
	ENABLED: 'Enabled',
	GROUP_TOOLS: (group: string) => `${group} tools`,
	NO_TOOLS: 'No tools available',
	SCOPE_CONVERSATION: 'Applies to this conversation.',
	SCOPE_NEW: 'Applies to new conversations. Tool picks inside a chat only affect that chat.',
	SHOWN_ARG: 'Shown argument',
	SHOWN_ARG_DEFAULT: 'Default',
	SHOWN_ARG_DEFAULT_OF: (arg: string) => `Default (${arg})`,
	SHOWN_ARG_NONE: 'None',
	SHOWN_ARG_NOTE: "The argument a call's line shows when the call has it."
} as const;
