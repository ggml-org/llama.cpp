export const IMAGE_NOT_ERROR_BOUND_SELECTOR = 'img:not([data-error-bound])';
export const DATA_ERROR_BOUND_ATTR = 'errorBound';
export const DATA_ERROR_HANDLED_ATTR = 'errorHandled';
export const BOOL_TRUE_STRING = 'true';
export const BOOL_FALSE_STRING = 'false';

/** Markdown structural markers used by `looksLikeMarkdown`. Inline / line-level. */
export const MARKDOWN = {
	ATX_HEADING_REGEX: /^#{1,6}\s+\S/,
	BLOCKQUOTE_REGEX: /^>\s+\S/,
	BOLD_REGEX: /\*\*[^*\n]+\*\*|__[^_\n]+__/,
	CODE_FENCE_REGEX: /^(```|~~~)/m,
	LINK_REGEX: /\[[^\]\n]+\]\([^)\s]+\)/,
	LIST_BULLET_REGEX: /^\s*[-*+]\s+\S/,
	LIST_NUMBERED_REGEX: /^\s*\d+[.)]\s+\S/,
	TABLE_SEPARATOR_REGEX: /^\s*\|?[\s:|-]+\|?\s*$/
} as const;
