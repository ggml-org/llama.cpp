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
