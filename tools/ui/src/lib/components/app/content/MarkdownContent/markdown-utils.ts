/**
 * Utility functions for markdown processing in MarkdownContent component.
 */

import { CODE_BLOCK, MARKDOWN_DATA_ATTRS } from '$lib/constants';
import type { RootContent as HastRootContent } from 'hast';

/**
 * Generates a unique identifier for a HAST node based on its position.
 * Used for stable block identification during incremental rendering.
 * @param node - The HAST root content node
 * @param indexFallback - Fallback index if position is unavailable
 * @returns Unique string identifier for the node
 */
export function getHastNodeId(node: HastRootContent, indexFallback: number): string {
	const position = node.position;

	if (position?.start?.offset != null && position?.end?.offset != null) {
		return `hast-${position.start.offset}-${position.end.offset}`;
	}

	return `${node.type}-${indexFallback}`;
}

/**
 * Generates a hash for MDAST node based on its position.
 * Used for cache lookup during incremental rendering.
 */
export function getMdastNodeHash(node: unknown, index: number): string {
	const n = node as {
		type?: string;
		position?: { start?: { offset?: number }; end?: { offset?: number } };
	};

	if (n.position?.start?.offset != null && n.position?.end?.offset != null) {
		return `${n.type}-${n.position.start.offset}-${n.position.end.offset}`;
	}

	return `${n.type}-idx${index}`;
}

/**
 * Determines if the new content is an append (new content added to existing blocks).
 * This is used to optimize cache reuse during streaming updates.
 *
 * @param newContent - The new markdown content
 * @param previousContent - The previous markdown content to check against
 * @returns true if the content appears to be an append operation
 */
export function isAppendMode(newContent: string, previousContent: string): boolean {
	return previousContent.length > 0 && newContent.startsWith(previousContent);
}

export interface CodeInfo {
	rawCode: string;
	language: string;
}

/**
 * Extracts code information from a button click target within a code block.
 * @param target - The clicked button element
 * @returns Object with rawCode and language, or null if extraction fails
 */
export function getCodeInfoFromTarget(target: HTMLElement): CodeInfo | null {
	const wrapper = target.closest('.code-block-wrapper');

	if (!wrapper) {
		console.error('No wrapper found');

		return null;
	}

	const codeElement = wrapper.querySelector<HTMLElement>(`code[${MARKDOWN_DATA_ATTRS.CODE_ID}]`);

	if (!codeElement) {
		console.error('No code element found in wrapper');

		return null;
	}

	const rawCode = codeElement.textContent ?? '';
	const languageLabel = wrapper.querySelector<HTMLElement>('.code-language');
	const language = languageLabel?.textContent?.trim() || 'text';

	return { language, rawCode };
}

/**
 * Extracts filename from the text above a code block. Use heuristics to select the most likely filename.
 * Quoted names have higest priority, unquoted names only when code block has a type and matches the .ext
 * @param text - The text to search
 * @param extension - The expected file extension (from code block type) or empty
 * @returns The extracted filename, or null if it could not reliable extracted
 */
export function extractFilenameFromText(text: string, extension: string | null): string | null {
	if (!text) return null;

	const targetExt = extension ? extension.replace(/^\./, '').toLowerCase() : null;

	interface Candidate {
		name: string;
		isQuoted: boolean;
		index: number;
	}

	const candidates: Candidate[] = [];

	// Reset state on shared global regex
	CODE_BLOCK.FILE_NAME_BOUNDARY_REGEX.lastIndex = 0;

	let match: RegExpExecArray | null;

	while ((match = CODE_BLOCK.FILE_NAME_BOUNDARY_REGEX.exec(text)) !== null) {
		const raw = match[0];
		const startIndex = match.index;
		// Detect if candidate is explicitly styled/quoted
		const isQuoted = /^(`+.*`+|\*{1,2}.*\*{1,2}|["'].*["'])$/.test(raw);
		// 1. Strip surrounding quotes, markdown markers, brackets, colons, and line numbers
		// Applied twice to handle nested styling like **`filename`**
		const cleaned = raw
			.replace(/^[`*'"([{]+|[`*'"\])}:]+$/g, '')
			.replace(/^[`*'"([{]+|[`*'"\])}:]+$/g, '')
			.replace(/:\d+(-\d+)?$/, '')
			.replace(/:$/, '');
		// 2. Strip virtual paths (basename only)
		const basename = cleaned.split(/[/\\]/).pop() || '';

		// Reject if missing, no extension, contains double dots (traversal), or illegal OS chars
		if (
			!basename ||
			!basename.includes('.') ||
			basename.includes('..') ||
			CODE_BLOCK.FILE_NAME_ILLEGAL_CHARS_REGEX.test(basename)
		) {
			continue;
		}

		const parts = basename.split('.');
		const candExt = parts.pop()?.toLowerCase();
		const nameWithoutExt = parts.join('.');

		// Validate that the base name and extension only contain valid alphanumeric/dash/underscore chars
		if (!nameWithoutExt || !candExt || !CODE_BLOCK.FILE_NAME_VALID_REGEX.test(nameWithoutExt)) {
			continue;
		}

		// Constraint Check: Target extension matching vs strict quoting
		if (targetExt) {
			if (candExt !== targetExt) continue;
		} else {
			if (!isQuoted) continue;
		}

		candidates.push({
			index: startIndex,
			isQuoted,
			name: basename
		});
	}

	if (candidates.length === 0) {
		return null;
	}

	// Sort: Quoted first, then highest text index (closest to code block)
	candidates.sort((a, b) => {
		if (a.isQuoted !== b.isQuoted) {
			return a.isQuoted ? -1 : 1;
		}

		return b.index - a.index;
	});

	return candidates[0].name;
}
