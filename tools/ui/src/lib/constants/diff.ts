/** Number of unchanged lines kept before/after each change in a diff hunk. */
export const DIFF_CONTEXT_WINDOW = 3;

/** Single-character line prefixes used by unified-diff output. */
export const DIFF_LINE_PREFIX = {
	CONTEXT: ' ',
	ADD: '+',
	DEL: '-'
} as const;

/** Substring markers inside diff headers / separators. */
export const DIFF_HEADER_MARKER = {
	DEL: '---',
	ADD: '+++',
	HUNK: '@@'
} as const;
