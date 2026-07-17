export const NEWLINE = '\n';
export const TAB = '\t';
export const DEFAULT_LANGUAGE = 'text';
export const LANG_PATTERN = /^(\w*)\n?/;
export const AMPERSAND_REGEX = /&/g;
export const LT_REGEX = /</g;
export const GT_REGEX = />/g;
export const FENCE_PATTERN = /^```|\n```/g;

// Whitespace-only empty lines (between start of string and first non-empty line).
// Used by trimCodePadding to drop leading/trailing phantom blank rows from LLM
// payload wrappers without touching internal blank lines.
export const TRIM_LEADING_PADDING_REGEX = /^(?:[ \t]*\n)+/;
export const TRIM_TRAILING_PADDING_REGEX = /(?:\n[ \t]*)+$/;

// Matches either Unix or Windows path separators so `String.split(REGEX)` can
// recover the trailing file-name segment from either `/foo/bar.txt` or
// `C:\foo\bar.txt`. Used wherever a parameter accepts a user-supplied path.
export const FILE_PATH_SEPARATOR_REGEX = /[\\/]/;

// Matches the `text:` prefix that file-type identifiers use to denote a
// plain-text language (e.g. `text:typescript`). Used by tool-call renderers
// to recover the underlying highlight.js language.
export const TEXT_LANGUAGE_PREFIX_REGEX = /^text:/;
