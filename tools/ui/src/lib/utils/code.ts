import hljs from 'highlight.js';
import {
	NEWLINE,
	DEFAULT_LANGUAGE,
	LANG_PATTERN,
	AMPERSAND_REGEX,
	LT_REGEX,
	GT_REGEX,
	FENCE_PATTERN,
	TRIM_LEADING_PADDING_REGEX,
	TRIM_TRAILING_PADDING_REGEX
} from '$lib/constants';

export interface IncompleteCodeBlock {
	language: string;
	code: string;
	openingIndex: number;
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
	return code.replace(TRIM_LEADING_PADDING_REGEX, '').replace(TRIM_TRAILING_PADDING_REGEX, '');
}

/**
 * Highlights code using highlight.js
 * @param code - The code to highlight
 * @param language - The programming language
 * @returns HTML string with syntax highlighting
 */
export function highlightCode(code: string, language: string): string {
	if (!code) return '';

	const trimmed = trimCodePadding(code);

	try {
		const lang = language.toLowerCase();
		const isSupported = hljs.getLanguage(lang);

		if (isSupported) {
			return hljs.highlight(trimmed, { language: lang }).value;
		} else {
			return hljs.highlightAuto(trimmed).value;
		}
	} catch {
		// Fallback to escaped plain text
		return trimmed
			.replace(AMPERSAND_REGEX, '&amp;')
			.replace(LT_REGEX, '&lt;')
			.replace(GT_REGEX, '&gt;');
	}
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
	const fencePattern = new RegExp(FENCE_PATTERN.source, FENCE_PATTERN.flags);
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
	const langMatch = afterOpening.match(LANG_PATTERN);
	const language = langMatch?.[1] || DEFAULT_LANGUAGE;
	const codeStartIndex = openingIndex + 3 + (langMatch?.[0]?.length ?? 0);
	const code = markdown.slice(codeStartIndex);

	return {
		language,
		code,
		openingIndex
	};
}
