export const IMAGE_NOT_ERROR_BOUND_SELECTOR = 'img:not([data-error-bound])';

/** Data attributes for the markdown renderer DOM contract. */
export const MARKDOWN_DATA_ATTRS = {
	BLOCK_ID: 'data-block-id',
	CODE_ID: 'data-code-id',
	ERROR_BOUND: 'data-error-bound',
	ERROR_HANDLED: 'data-error-handled',
	LISTENER_BOUND: 'data-listener-bound',
	ORIGINAL_SRC: 'data-original-src'
} as const;

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
