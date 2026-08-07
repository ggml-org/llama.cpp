import type { AgenticConfig } from '$lib/types/agentic';

export const ATTACHMENT_SAVED_REGEX = /\[Attachment saved: ([^\]]+)\]/;

// JSON detection: trimmed content opens with an object or array literal.
export const TOOL_RESULT_JSON_OPEN_REGEX = /^[[{]/;

// Markdown structural markers used by `looksLikeMarkdown`. Inline / line-level.
export const MARKDOWN_CODE_FENCE_REGEX = /^(```|~~~)/m;
export const MARKDOWN_ATX_HEADING_REGEX = /^#{1,6}\s+\S/;
export const MARKDOWN_BLOCKQUOTE_REGEX = /^>\s+\S/;
export const MARKDOWN_LIST_BULLET_REGEX = /^\s*[-*+]\s+\S/;
export const MARKDOWN_LIST_NUMBERED_REGEX = /^\s*\d+[.)]\s+\S/;
export const MARKDOWN_LINK_REGEX = /\[[^\]\n]+\]\([^)\s]+\)/;
export const MARKDOWN_BOLD_REGEX = /\*\*[^*\n]+\*\*|__[^_\n]+__/;
export const MARKDOWN_TABLE_SEPARATOR_REGEX = /^\s*\|?[\s:|-]+\|?\s*$/;

// Search-summary wire format used by file-glob and grep tools:
//   <matches>
//   ---
//   Total matches: N
export const SEARCH_SUMMARY_SEPARATOR = '---\n';
export const SEARCH_SUMMARY_TOTAL_REGEX = /Total matches:\s*(\d+)/;

// Separator rendered between stats in the tool-result footer (e.g. between a
// result message and the byte/edit count). Plain ASCII spaces bracket a hyphen
// so the whole " - " sits on one visual line even when the surrounding text
// wraps mid-paragraph.
export const RESULT_STAT_SEPARATOR = ' - ';

export const DEFAULT_AGENTIC_CONFIG: AgenticConfig = {
	enabled: true,
	maxTurns: 100
} as const;

export const REASONING_TAGS = {
	START: '<think>',
	END: '</think>'
} as const;

/**
 * @deprecated Legacy marker tags - only used for migration of old stored messages.
 * New messages use structured fields (reasoningContent, toolCalls, toolCallId).
 */
export const LEGACY_AGENTIC_TAGS = {
	TOOL_CALL_START: '<<<AGENTIC_TOOL_CALL_START>>>',
	TOOL_CALL_END: '<<<AGENTIC_TOOL_CALL_END>>>',
	TOOL_NAME_PREFIX: '<<<TOOL_NAME:',
	TOOL_ARGS_START: '<<<TOOL_ARGS_START>>>',
	TOOL_ARGS_END: '<<<TOOL_ARGS_END>>>',
	TAG_SUFFIX: '>>>'
} as const;

/**
 * @deprecated Legacy reasoning tags - only used for migration of old stored messages.
 * New messages use the dedicated reasoningContent field.
 */
export const LEGACY_REASONING_TAGS = {
	START: '<<<reasoning_content_start>>>',
	END: '<<<reasoning_content_end>>>'
} as const;

/**
 * @deprecated Legacy regex patterns - only used for migration of old stored messages.
 */
export const LEGACY_AGENTIC_REGEX = {
	COMPLETED_TOOL_CALL:
		/<<<AGENTIC_TOOL_CALL_START>>>\n<<<TOOL_NAME:(.+?)>>>\n<<<TOOL_ARGS_START>>>([\s\S]*?)<<<TOOL_ARGS_END>>>([\s\S]*?)<<<AGENTIC_TOOL_CALL_END>>>/g,
	REASONING_BLOCK: /<<<reasoning_content_start>>>[\s\S]*?<<<reasoning_content_end>>>/g,
	REASONING_EXTRACT: /<<<reasoning_content_start>>>([\s\S]*?)<<<reasoning_content_end>>>/,
	REASONING_OPEN: /<<<reasoning_content_start>>>[\s\S]*$/,
	AGENTIC_TOOL_CALL_BLOCK: /\n*<<<AGENTIC_TOOL_CALL_START>>>[\s\S]*?<<<AGENTIC_TOOL_CALL_END>>>/g,
	AGENTIC_TOOL_CALL_OPEN: /\n*<<<AGENTIC_TOOL_CALL_START>>>[\s\S]*$/,
	HAS_LEGACY_MARKERS: /<<<(?:AGENTIC_TOOL_CALL_START|reasoning_content_start)>>>/
} as const;
