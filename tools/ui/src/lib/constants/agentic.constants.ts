import type { AgenticConfig } from '$lib/types/agentic';

export const ATTACHMENT_SAVED_REGEX = /\[Attachment saved: ([^\]]+)\]/;

// JSON detection: trimmed content opens with an object or array literal.
export const TOOL_RESULT_JSON_OPEN_REGEX = /^[[{]/;

// Search-summary wire format used by file-glob and grep tools:
//   <matches>
//   ---
//   Total matches: N
export const SEARCH_SUMMARY = {
	SEPARATOR: '---\n',
	TOTAL_REGEX: /Total matches:\s*(\d+)/
} as const;

// Separator rendered between stats in the tool-result footer (e.g. between a
// result message and the byte/edit count). Plain ASCII spaces bracket a hyphen
// so the whole " - " sits on one visual line even when the surrounding text
// wraps mid-paragraph.
export const RESULT_STAT_SEPARATOR = ' - ';

// Separator between the assistant text sections of a grouped agentic
// session when they are joined for the clipboard.
export const AGENTIC_TEXT_COPY_SEPARATOR = '\n\n';

export const DEFAULT_AGENTIC_CONFIG: AgenticConfig = {
	enabled: true,
	maxTurns: 100
} as const;

export const REASONING_TAGS = {
	END: '</think>',
	START: '<think>'
} as const;

/**
 * @deprecated Legacy marker tags - only used for migration of old stored messages.
 * New messages use structured fields (reasoningContent, toolCalls, toolCallId).
 */
export const LEGACY_AGENTIC_TAGS = {
	TAG_SUFFIX: '>>>',
	TOOL_ARGS_END: '<<<TOOL_ARGS_END>>>',
	TOOL_ARGS_START: '<<<TOOL_ARGS_START>>>',
	TOOL_CALL_END: '<<<AGENTIC_TOOL_CALL_END>>>',
	TOOL_CALL_START: '<<<AGENTIC_TOOL_CALL_START>>>',
	TOOL_NAME_PREFIX: '<<<TOOL_NAME:'
} as const;

/**
 * @deprecated Legacy reasoning tags - only used for migration of old stored messages.
 * New messages use the dedicated reasoningContent field.
 */
export const LEGACY_REASONING_TAGS = {
	END: '<<<reasoning_content_end>>>',
	START: '<<<reasoning_content_start>>>'
} as const;

/**
 * @deprecated Legacy regex patterns - only used for migration of old stored messages.
 */
export const LEGACY_AGENTIC_REGEX = {
	AGENTIC_TOOL_CALL_BLOCK: /\n*<<<AGENTIC_TOOL_CALL_START>>>[\s\S]*?<<<AGENTIC_TOOL_CALL_END>>>/g,
	AGENTIC_TOOL_CALL_OPEN: /\n*<<<AGENTIC_TOOL_CALL_START>>>[\s\S]*$/,
	COMPLETED_TOOL_CALL:
		/<<<AGENTIC_TOOL_CALL_START>>>\n<<<TOOL_NAME:(.+?)>>>\n<<<TOOL_ARGS_START>>>([\s\S]*?)<<<TOOL_ARGS_END>>>([\s\S]*?)<<<AGENTIC_TOOL_CALL_END>>>/g,
	HAS_LEGACY_MARKERS: /<<<(?:AGENTIC_TOOL_CALL_START|reasoning_content_start)>>>/,
	REASONING_BLOCK: /<<<reasoning_content_start>>>[\s\S]*?<<<reasoning_content_end>>>/g,
	REASONING_EXTRACT: /<<<reasoning_content_start>>>([\s\S]*?)<<<reasoning_content_end>>>/,
	REASONING_OPEN: /<<<reasoning_content_start>>>[\s\S]*$/
} as const;
