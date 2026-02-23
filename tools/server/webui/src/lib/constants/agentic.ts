import type { AgenticConfig } from '$lib/types/agentic';

export const ATTACHMENT_SAVED_REGEX = /\[Attachment saved: ([^\]]+)\]/;

export const NEWLINE_SEPARATOR = '\n';

export const TURN_LIMIT_MESSAGE = '\n\n```\nTurn limit reached\n```\n';

export const LLM_ERROR_BLOCK_START = '\n\n```\nUpstream LLM error:\n';
export const LLM_ERROR_BLOCK_END = '\n```\n';

export const DEFAULT_AGENTIC_CONFIG: AgenticConfig = {
	enabled: true,
	maxTurns: 100,
	maxToolPreviewLines: 25
} as const;

// Agentic tool call tag markers
export const AGENTIC_TAGS = {
	TOOL_CALL_START: '<<<AGENTIC_TOOL_CALL_START>>>',
	TOOL_CALL_END: '<<<AGENTIC_TOOL_CALL_END>>>',
	TOOL_NAME_PREFIX: '<<<TOOL_NAME:',
	TOOL_ARGS_START: '<<<TOOL_ARGS_START>>>',
	TOOL_ARGS_END: '<<<TOOL_ARGS_END>>>',
	TAG_SUFFIX: '>>>'
} as const;

export const REASONING_TAGS = {
	START: '<<<reasoning_content_start>>>',
	END: '<<<reasoning_content_end>>>'
} as const;

// Regex for trimming leading/trailing newlines
export const TRIM_NEWLINES_REGEX = /^\n+|\n+$/g;

// Regex patterns for parsing agentic content
export const AGENTIC_REGEX = {
	// Matches completed tool calls (with END marker)
	COMPLETED_TOOL_CALL:
		/<<<AGENTIC_TOOL_CALL_START>>>\n<<<TOOL_NAME:(.+?)>>>\n<<<TOOL_ARGS_START>>>([\s\S]*?)<<<TOOL_ARGS_END>>>([\s\S]*?)<<<AGENTIC_TOOL_CALL_END>>>/g,
	// Matches pending tool call (has NAME and ARGS but no END)
	PENDING_TOOL_CALL:
		/<<<AGENTIC_TOOL_CALL_START>>>\n<<<TOOL_NAME:(.+?)>>>\n<<<TOOL_ARGS_START>>>([\s\S]*?)<<<TOOL_ARGS_END>>>([\s\S]*)$/,
	// Matches partial tool call (has START and NAME, ARGS still streaming)
	PARTIAL_WITH_NAME:
		/<<<AGENTIC_TOOL_CALL_START>>>\n<<<TOOL_NAME:(.+?)>>>\n<<<TOOL_ARGS_START>>>([\s\S]*)$/,
	// Matches early tool call (just START marker)
	EARLY_MATCH: /<<<AGENTIC_TOOL_CALL_START>>>([\s\S]*)$/,
	// Matches partial marker at end of content
	PARTIAL_MARKER: /<<<[A-Za-z_]*$/,
	// Matches reasoning content blocks (including tags)
	REASONING_BLOCK: /<<<reasoning_content_start>>>[\s\S]*?<<<reasoning_content_end>>>/g,
	// Matches an opening reasoning tag and any remaining content (unterminated)
	REASONING_OPEN: /<<<reasoning_content_start>>>[\s\S]*$/,
	// Matches tool name inside content
	TOOL_NAME_EXTRACT: /<<<TOOL_NAME:([^>]+)>>>/
} as const;
