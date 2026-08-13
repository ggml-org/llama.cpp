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

	TRIM_TRAILING_PADDING_REGEX: /(?:\n[ \t]*)+$/,

	/** filter file name from MD code block, like ```python title="script.py" */
	FILE_NAME_REGEX: /(?:^|\s)(?:[a-zA-Z0-9_-]+=)?["'`]?([^"'`\s]+\.[a-zA-Z][a-zA-Z0-9]{0,6})["'`]?(?:\s|$)/i,

	/** filter file name from just above MD code block, like here is **script.py**: ```python... */
	FILE_NAME_BOUNDARY_REGEX: /`([^`]+)`|"([^"]+)"|'([^']+)'|\(([^)]+)\)|\*\*([^*]+)\*\*/g
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
	DOWNLOAD_BTN: 'download-code-btn',
	RELATIVE: 'relative',
	SCROLL_CONTAINER: 'code-block-scroll-container',
	WRAPPER: 'code-block-wrapper'
} as const;

/** Attributes applied by the markdown code-block renderer. */
export const CODE_BLOCK_ATTR = {
	CODE_ID: 'data-code-id',
	META_DATA: 'data-meta',
	FILE_NAME: 'data-filename'
} as const;

/** Language sensitive texts */
export const CODE_BLOCK_TEXT = {
	COPY_BTN_TITLE: 'Copy code',
	DOWNLOAD_BTN_TITLE: 'Download file',
	PREVIEW_TITLE: 'Preview code'
} as const;

/** Markdown code block type name to file extension mapping */
export const CODE_BLOCK_TYPE_TO_EXTENSION_MAP: Record<string, string> = {
	python: '.py', py: '.py',
	javascript: '.js', js: '.js',
	typescript: '.ts', ts: '.ts',
	bash: '.sh', sh: '.sh', shell: '.sh',
	'c++': '.cpp', cpp: '.cpp',
	yaml: '.yml', yml: '.yml',
	markdown: '.md', text: '.txt', plaintext: '.txt'
};
