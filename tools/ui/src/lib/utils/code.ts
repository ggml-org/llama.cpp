import { CODE_BLOCK, NEWLINE } from '$lib/constants';
import hljs from 'highlight.js';

export interface IncompleteCodeBlock {
	language: string;
	code: string;
	openingIndex: number;
}

// A fence line: up to 3 leading spaces (CommonMark), 3+ backticks, then
// whatever trails on the same line.
const FENCE_LINE_REGEX = /^ {0,3}(`{3,})(.*)$/;

/**
 * Splits text glued to a closing code fence onto its own line:
 *
 *     ```ts
 *     let foo = 'bar';
 *     ```create this file on ...
 *
 * A closing fence with trailing text is not a fence to the markdown
 * parser, so the block would swallow the text as code. The chat form
 * normally keeps the fence on its own line, but older messages and
 * hand-pasted content can carry the glued form.
 *
 * Only trailing text containing whitespace is split: a single word
 * after the backticks inside a fenced block is more likely nested
 * markdown (a ```python example inside a ```md block) than glued prose.
 */
export function splitGluedClosingCodeFences(markdown: string): string {
	if (!markdown.includes('```')) return markdown;

	const lines = markdown.split(NEWLINE);

	let inside = false;
	let changed = false;

	for (let i = 0; i < lines.length; i++) {
		const match = FENCE_LINE_REGEX.exec(lines[i]);

		if (!match) continue;

		if (!inside) {
			inside = true;

			continue;
		}

		inside = false;

		const trailing = match[2];

		if (trailing.includes('`') || !/\s/.test(trailing)) continue;

		lines[i] = lines[i].slice(0, lines[i].length - trailing.length);
		lines.splice(i + 1, 0, trailing.trim());
		i++;
		changed = true;
	}

	return changed ? lines.join(NEWLINE) : markdown;
}

/**
 * Strips empty lines (whitespace-only) from the start and end of code.
 *
 * Tool call payloads frequently arrive with surrounding whitespace from LLM
 * formatting (`"\nfunction ...\n"`). Preserving those newlines makes hljs emit
 * a leading/trailing empty line that `<pre>` then renders as a phantom row,
 * pushing real content away from the box edge. The trim keeps the body intact
 * so internal blank lines are still rendered as such.
 */
function trimCodePadding(code: string): string {
	return code
		.replace(CODE_BLOCK.TRIM_LEADING_PADDING_REGEX, '')
		.replace(CODE_BLOCK.TRIM_TRAILING_PADDING_REGEX, '');
}

function escapeCode(code: string): string {
	return code
		.replace(CODE_BLOCK.AMPERSAND_REGEX, '&amp;')
		.replace(CODE_BLOCK.LT_REGEX, '&lt;')
		.replace(CODE_BLOCK.GT_REGEX, '&gt;');
}

/** Bounded cache for highlightCode results. */
const HIGHLIGHT_CACHE_MAX_SIZE = 64;
const highlightCache = new Map<string, string>();

/**
 * Highlights code using highlight.js
 * @param code - The code to highlight
 * @param language - The programming language
 * @param autoDetect - Fall back to `highlightAuto` when `language` is unknown.
 *   Callers rendering a still-streaming block should pass false: auto-detection
 *   costs ~38ms per call and re-guesses on every chunk, so the language (and
 *   therefore the whole highlight) flickers as the block grows.
 * @returns HTML string with syntax highlighting
 */
export function highlightCode(code: string, language: string, autoDetect = true): string {
	if (!code) return '';

	// Cache key includes language and autoDetect flag since results differ.
	// During streaming, the same code string may be highlighted repeatedly
	// (e.g., when text after a code block changes but the code itself doesn't).
	const cacheKey = `${language}:${autoDetect}:${code}`;
	const cached = highlightCache.get(cacheKey);

	if (cached) return cached;

	const trimmed = trimCodePadding(code);

	let result: string;

	try {
		const lang = language.toLowerCase();
		const isSupported = hljs.getLanguage(lang);

		if (isSupported) {
			result = hljs.highlight(trimmed, { language: lang }).value;
		} else if (autoDetect) {
			result = hljs.highlightAuto(trimmed).value;
		} else {
			result = escapeCode(trimmed);
		}
	} catch {
		result = escapeCode(trimmed);
	}

	if (highlightCache.size >= HIGHLIGHT_CACHE_MAX_SIZE) {
		highlightCache.delete(highlightCache.keys().next().value!);
	}

	highlightCache.set(cacheKey, result);

	return result;
}

export { trimCodePadding };

/**
 * Detects if markdown ends with an incomplete code block (opened but not closed).
 * Returns the code block info if found, null otherwise.
 * @param markdown - The raw markdown string to check
 * @returns IncompleteCodeBlock info or null
 */
export function detectIncompleteCodeBlock(markdown: string): IncompleteCodeBlock | null {
	// Count all code fences in the markdown
	// A code block is incomplete if there's an odd number of ``` fences
	const fencePattern = new RegExp(CODE_BLOCK.FENCE_PATTERN.source, CODE_BLOCK.FENCE_PATTERN.flags);
	const fences: number[] = [];

	let fenceMatch;

	while ((fenceMatch = fencePattern.exec(markdown)) !== null) {
		// Store the position after the ```
		const pos = fenceMatch[0].startsWith(NEWLINE) ? fenceMatch.index + 1 : fenceMatch.index;

		fences.push(pos);
	}

	// If even number of fences (including 0), all code blocks are closed
	if (fences.length % 2 === 0) {
		return null;
	}

	// Odd number means last code block is incomplete
	// The last fence is the opening of the incomplete block
	const openingIndex = fences[fences.length - 1];
	const afterOpening = markdown.slice(openingIndex + 3);
	// Extract language and code content
	const langMatch = afterOpening.match(CODE_BLOCK.LANG_PATTERN);
	const language = langMatch?.[1] || CODE_BLOCK.DEFAULT_LANGUAGE;
	const codeStartIndex = openingIndex + 3 + (langMatch?.[0]?.length ?? 0);
	const code = markdown.slice(codeStartIndex);

	return {
		code,
		language,
		openingIndex
	};
}
