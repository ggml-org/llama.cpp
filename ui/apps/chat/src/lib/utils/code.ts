import { CODE_BLOCK, NEWLINE } from '#lib/constants/index.ts';

export {
	getLoadedHighlightLanguages,
	highlightCode,
	isHighlightLanguageLoaded,
	loadHighlightLanguages,
	resolveHighlightLanguage,
	trimCodePadding
} from '@ggml-org/llama-ui/code';

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
