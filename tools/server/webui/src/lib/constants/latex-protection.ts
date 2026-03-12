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

/** map from mchem-regexp to replacement */
export const MHCHEM_PATTERN_MAP: readonly [RegExp, string][] = [
	[/(\s)\$\\ce{/g, '$1$\\\\ce{'],
	[/(\s)\$\\pu{/g, '$1$\\\\pu{']
] as const;
