import hljs from 'highlight.js';
import {
	NEWLINE,
	DEFAULT_LANGUAGE,
	LANG_PATTERN,
	AMPERSAND_REGEX,
	LT_REGEX,
	GT_REGEX,
	FENCE_PATTERN
} from '$lib/constants/code';

export interface IncompleteCodeBlock {
	language: string;
	code: string;
	openingIndex: number;
}

/**
 * Highlights code using highlight.js
 * @param code - The code to highlight
 * @param language - The programming language
 * @returns HTML string with syntax highlighting
 */
export function highlightCode(code: string, language: string): string {
	if (!code) return '';

	try {
		const lang = language.toLowerCase();
		const isSupported = hljs.getLanguage(lang);

		if (isSupported) {
			return hljs.highlight(code, { language: lang }).value;
		} else {
			return hljs.highlightAuto(code).value;
		}
	} catch {
		// Fallback to escaped plain text
		return code
			.replace(AMPERSAND_REGEX, '&amp;')
			.replace(LT_REGEX, '&lt;')
			.replace(GT_REGEX, '&gt;');
	}
}

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
