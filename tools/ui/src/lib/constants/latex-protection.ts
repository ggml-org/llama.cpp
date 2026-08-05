/**
 * Matches common Markdown code blocks to exclude them from further processing (e.g. LaTeX).
 * - Fenced: ```...```
 * - Inline: `...` (does NOT support nested backticks or multi-backtick syntax)
 *
 * Note: This pattern does not handle advanced cases like:
 *       `` `code with `backticks` `` or \\``...\\``
 */
export const CODE_BLOCK_REGEXP = /(```[\s\S]*?```|`[^`\n]+`)/g;

/**
 * Matches LaTeX math delimiters \(...\) and \[...\] only when not preceded by a backslash (i.e., not escaped),
 * while also capturing code blocks (```, `...`) so they can be skipped during processing.
 *
 * Uses negative lookbehind `(?<!\\)` to avoid matching \\( or \\[.
 * Using the look‑behind pattern `(?<!\\)` we skip matches
 * that are preceded by a backslash, e.g.
 * `Definitions\\(also called macros)` (title of chapter 20 in The TeXbook)
 * or `\\[4pt]` (LaTeX line-break).
 *
 * group 1: code-block
 * group 2: square-bracket
 * group 3: round-bracket
 */
export const LATEX_MATH_AND_CODE_PATTERN =
	/(```[\S\s]*?```|`.*?`)|(?<!\\)\\\[([\S\s]*?[^\\])\\]|(?<!\\)\\\((.*?)\\\)/g;

/** Regex to capture the content of a $$...\\\\...$$ block (display-formula with line-break) */
export const LATEX_LINEBREAK_REGEXP = /\$\$([\s\S]*?\\\\[\s\S]*?)\$\$/;

/**
 * Matches the unescaped `\[...\]` display-math delimiter and surrounding
 * context so callers can insert line-breaks around the placeholder or convert
 * to inline when the formula has a non-empty trailing context (e.g. a table
 * cell that opens with `\[` and closes with content after `\]`).
 *
 * group 1: prefix before `\[`
 * group 2: formula body
 * group 3: trailing context after `\]`
 */
export const LATEX_DISPLAY_BLOCK_REGEXP = /([\S].*?)\\\[([\s\S]*?)\\\](.*)/g;

/**
 * Cheap gate for `preprocessLaTeX`. Every transformation it performs is triggered
 * by a `$` (inline/display math, currency escaping) or a backslash escape
 * (`\(`, `\[`, `\ce{`, `\pu{`). Text containing neither is returned untouched, so
 * this lets the caller skip the whole protect/restore pipeline.
 */
export const LATEX_TRIGGER_REGEXP = /[$\\]/;

/** Inline LaTeX math delimiter (the dollar sign). */
export const LATEX_INLINE_DELIMITER = '$';

/** Display LaTeX math delimiter (paired dollar signs). */
export const LATEX_DISPLAY_DELIMITER = '$$';

/** Matches a single non-whitespace character. */
export const LATEX_NON_WHITESPACE_REGEXP = /\S/;

/** Matches a character that may appear adjacent to `$`, indicating a non-TeX
 * context such as an identifier (`var$`, `$var`), currency ($5), or code. */
export const LATEX_NEIGHBOR_CHAR_REGEXP = /[A-Za-z0-9_$-]/;

/** Matches a single digit (used to detect currency-like `$5`). */
export const LATEX_DIGIT_REGEXP = /[0-9]/;

/** Matches the leading blockquote prefix (`> ` or `>`) on a markdown line. */
export const LATEX_BLOCKQUOTE_PREFIX_REGEXP = /^(>\s*)/;

/** Matches the placeholder inserted by the protect/restore pipeline for a
 * protected LaTeX expression. Group 1 is the index into `latexExpressions`. */
export const LATEX_PLACEHOLDER_REGEXP = /<<LATEX_(\d+)>>/g;

/** Matches the placeholder inserted by the protect/restore pipeline for a
 * protected code block. Group 1 is the index into `codeBlocks`. */
export const CODE_BLOCK_PLACEHOLDER_REGEXP = /<<CODE_BLOCK_(\d+)>>/g;

/** Matches a `$` immediately followed by a digit, which is treated as a
 * currency amount (e.g. `$5`) and escaped to `\$5` so it isn't parsed as math. */
export const LATEX_CURRENCY_DOLLAR_REGEXP = /\$(?=\d)/g;

/** Captures remaining `$$...$$`, `\[...\]`, `\(...\)` (only unescaped via
 * `(?<!\\)`) after the display-block pass has run. Group 1 holds the
 * matched formula. */
export const LATEX_PROTECT_REGEXP =
	/(\$\$[\s\S]*?\$\$|(?<!\\)\\\[[\s\S]*?\\\]|(?<!\\)\\\(.*?\\\))/g;

/** Matches unescaped inline `\(...\)` (at least one char inside) used to
 * convert `\(` → `$` after the protect pass. */
export const LATEX_INLINE_CONVERT_REGEXP = /(?<!\\)\\\((.+?)\\\)/g;

/** Matches unescaped display `\[...\]` used to convert `\[` → `$$`
 * after the protect pass. */
export const LATEX_DISPLAY_CONVERT_REGEXP = /(?<!\\)\\\[([\s\S]*?)\\\]/g;

/** `\(` — opens an inline LaTeX math block. */
export const LATEX_INLINE_OPEN = '\\(';

/** `\)` — closes an inline LaTeX math block. */
export const LATEX_INLINE_CLOSE = '\\)';

/** `\[` — opens a display LaTeX math block. */
export const LATEX_DISPLAY_OPEN = '\\[';

/** `\]` — closes a display LaTeX math block. */
export const LATEX_DISPLAY_CLOSE = '\\]';

/** `\` — the LaTeX escape character. */
export const LATEX_BACKSLASH = '\\';

/** `\$` — dollar sign escaped so it isn't parsed as math (used to disambiguate
 * currency amounts like `$5`). */
export const LATEX_CURRENCY_ESCAPE = '\\$';

/** `\ce{` — mhchem chemistry command prefix. */
export const LATEX_MHCHEM_CE = '\\ce{';

/** `\pu{` — mhchem physics-unit command prefix. */
export const LATEX_MHCHEM_PU = '\\pu{';

/** map from mchem-regexp to replacement */
export const MHCHEM_PATTERN_MAP: readonly [RegExp, string][] = [
	[/(\s)\$\\ce{/g, '$1$\\\\ce{'],
	[/(\s)\$\\pu{/g, '$1$\\\\pu{']
] as const;
