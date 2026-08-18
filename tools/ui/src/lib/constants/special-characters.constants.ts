// Control / whitespace / formatting characters that appear literally inside rendered text.

/** Line feed. */
export const NEWLINE = '\n';

/** Horizontal tab. */
export const TAB = '\t';

/** Non-breaking space. */
export const NBSP = '\u00a0';

/** Non-breaking spaces used to render a tab stop that whitespace collapsing would otherwise squash. */
export const TAB_AS_SPACES = NBSP.repeat(4);

/** Matches a CR-terminated or bare LF line break. */
export const LINE_BREAK = /\r?\n/;
