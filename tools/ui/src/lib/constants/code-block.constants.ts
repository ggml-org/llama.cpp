// Constants for the markdown code-block renderer: language/fence handling and CSS classes.

/** Parsing and escaping helpers for the markdown code-block renderer. */
export const CODE_BLOCK = {
	AMPERSAND_REGEX: /&/g,
	/** Language fallback used when no language is specified. */
	DEFAULT_LANGUAGE: 'text',
	/** Matches opening/closing markdown code fences. */
	FENCE_PATTERN: /^```|\n```/g,
	GT_REGEX: />/g,
	/** Matches the language specifier at the start of a code fence. */
	LANG_PATTERN: /^(\w*)\n?/,
	LT_REGEX: /</g,

	// Matches the `text:` prefix that file-type identifiers use to denote a
	// plain-text language (e.g. `text:typescript`). Used by tool-call renderers
	// to recover the underlying highlight.js language.
	TEXT_LANGUAGE_PREFIX_REGEX: /^text:/,
	// Whitespace-only empty lines (between start of string and first non-empty line).
	// Used by trimCodePadding to drop leading/trailing phantom blank rows from LLM
	// payload wrappers without touching internal blank lines.
	TRIM_LEADING_PADDING_REGEX: /^(?:[ \t]*\n)+/,

	TRIM_TRAILING_PADDING_REGEX: /(?:\n[ \t]*)+$/
} as const;

// Matches either Unix or Windows path separators so `String.split(REGEX)` can
// recover the trailing file-name segment from either `/foo/bar.txt` or
// `C:\foo\bar.txt`. Used wherever a parameter accepts a user-supplied path.
export const FILE_PATH_SEPARATOR_REGEX = /[\\/]/;

// Separates a file name from its extension, e.g. the '.' in `cover.png`.
export const FILE_EXTENSION_SEPARATOR = '.';

// Matches the `text:` prefix that file-type identifiers use to denote a
// plain-text language (e.g. `text:typescript`). Used by tool-call renderers
// to recover the underlying highlight.js language.
export const TEXT_LANGUAGE_PREFIX_REGEX = /^text:/;

/** CSS classes applied by the markdown code-block renderer. */
export const CODE_BLOCK_CLASS = {
	ACTIONS: 'code-block-actions',
	COPY_BTN: 'copy-code-btn',
	HEADER: 'code-block-header',
	LANGUAGE: 'code-language',
	PREVIEW_BTN: 'preview-code-btn',
	RELATIVE: 'relative',
	SCROLL_CONTAINER: 'code-block-scroll-container',
	WRAPPER: 'code-block-wrapper'
} as const;
