/**
 * Utility functions for markdown processing in MarkdownContent component.
 */

import { CODE_BLOCK, CODE_BLOCK_CLASS, MARKDOWN_DATA_ATTRS } from '$lib/constants';
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
	const wrapper = target.closest(`.${CODE_BLOCK_CLASS.WRAPPER}`);

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
	const languageLabel = wrapper.querySelector<HTMLElement>(`.${CODE_BLOCK_CLASS.LANGUAGE}`);
	const language = languageLabel?.textContent?.trim() || CODE_BLOCK.DEFAULT_LANGUAGE;

	return { language, rawCode };
}

/**
 * Extracts the filename from the text above a code block. Like save **file.js**\n```javascript....
 * @param text - The MD text that contains the text + code block
 * @param position - The position of the code block (points to ```) to start search backwards
 * @param extension - The expected file extension (from code block type) or null/empty
 * @returns The extracted filename, or null if it could not be reliably extracted
 */
export function extractFilenameFromText(
	text: string,
	position: number,
	extension: string | null
): string | null {
	const isWrapMark = (c: string): boolean => CODE_BLOCK.DOWNLOAD_NAME_WRAP_MARKS.includes(c);
	const isStopMark = (c: string): boolean => CODE_BLOCK.DOWNLOAD_NAME_STOP_MARKS.includes(c);
	const isValidChar = (c: string): boolean => CODE_BLOCK.FILE_NAME_VALID_CHAR_REGEX.test(c);
	const isValidExtChar = (c: string): boolean => CODE_BLOCK.FILE_EXT_VALID_CHAR_REGEX.test(c);
	const isWhitespace = (c: string): boolean => c.charCodeAt(0) <= 32; // space and control are considered white

	if (!text) return null;

	let unwrappedCandidate: string | null = null;
	let lineCount = 0; // non empty lines counter
	let startPos = Math.min(text.length - 1, position);

	const stopPos = Math.max(position - CODE_BLOCK.DOWNLOAD_NAME_SCAN_BYTES_BACK, 0);

	if (extension)
		extension = extension.startsWith('.') ? extension.toLowerCase() : '.' + extension.toLowerCase();

	// set start pos to last character before fence, if not already the case
	if (startPos >= 1 && text[startPos] == '`' && text[startPos - 1] == '\n') startPos--;

	// Manual backwards scanning here in favor of complex, bug-prone and hard to maintain regex matching.
	// First scan for a wrapped filename (backticks, bold, parentheses, etc.). If the extension is known, it must match.
	// As fallback, take an unwrapped filename (breaks on whitespace), but only if the code block has a known type and the extension matches.
	let pos = startPos;

	while (pos > stopPos) {
		// stop when reaching max lines
		if (text[pos] == '\r' || text[pos] == '\n') {
			// skip all empty lines, they are only accounted once for the limited
			while (pos >= stopPos && (text[pos] == '\r' || text[pos] == '\n')) pos--;

			// as start position points to ``` the line above counts as the 1st line
			if (lineCount++ > CODE_BLOCK.DOWNLOAD_NAME_SCAN_LINES_BACK) break;
		}

		// stop when reaching previous fence
		if (pos >= 2 && text[pos] == '`' && text[pos - 1] == '`' && text[pos - 2] == '`') break;

		// Scan for possible file extension
		if (text[pos] !== '.') {
			pos--;

			continue;
		}

		let extEndPos = -1;
		let isWrapped = false;

		// Scan till extension ends
		for (let scan = pos + 1; scan <= startPos; scan++) {
			const c = text[scan];

			if (isWrapMark(c)) {
				isWrapped = true;

				break; // Valid end, wrapping mark detected like *file.ext*
			}

			if (isWhitespace(c) || isStopMark(c)) {
				isWrapped = c == ' ' && scan + 1 < text.length && isWrapMark(text[scan + 1]); // allow 1 space to wrap mark

				break; // Valid break on whitespace or 'here is file.txt:'
			}

			if (!isValidExtChar(c) || scan - pos > CODE_BLOCK.FILE_NAME_MAX_EXT_LENGTH) {
				extEndPos = -1;

				break; // Invalid end, abort this candidate
			}

			extEndPos = scan;
		}

		if (extEndPos === -1) {
			pos--;

			continue; // abort this candidate
		}

		// Scan back for filename start. re-use pos to advance main position
		let nameStartPos = -1;

		while (pos > stopPos) {
			pos--;
			const c = text[pos];

			if (isWrapped) {
				if (isStopMark(c) || isWrapMark(c)) {
					break; // Also cuts off path/file.ext
				}

				if (!isValidChar(c)) {
					nameStartPos = -1;

					break; // Invalid start (whitespace is valid in file names when wrapped)
				}
			} else {
				if (isStopMark(c) || isWhitespace(c)) {
					break; // Space characters not supported in unwrapped file names, so cut of here
				}

				if (!isValidChar(c)) {
					nameStartPos = -1;

					break; // Invalid start
				}
			}

			nameStartPos = pos;
		}

		if (nameStartPos === -1) {
			continue; // Abort this candidate. Pos is guarantueed to be decreased to the next unscanned character
		}

		const candidate = text.slice(nameStartPos, extEndPos + 1).trim(); // allow space before wrap mark

		if (!isValidCodeblockFilename(candidate)) continue; // illegal file name combinations

		if (!isWrapped) {
			if (!unwrappedCandidate && extension && candidate.toLowerCase().endsWith(extension)) {
				unwrappedCandidate = candidate;
			}

			continue; // Extension mandatory for unwrapped name
		}

		if (!extension || candidate.toLowerCase().endsWith(extension)) {
			return candidate; // Valid wrapped name, need to match extension if supplied
		}
	}

	return unwrappedCandidate;
}

/**
 * Checks if the file name (to download) from a code block is valid 'enough' to be portable
 * @param filename - The file name with (mandatory) extension
 * @returns true if valid
 */
export function isValidCodeblockFilename(filename: string): boolean {
	const dotPos = !filename ? -1 : filename.lastIndexOf('.');

	return (
		dotPos > 0 &&
		!filename.endsWith('.') &&
		!filename.includes('..') &&
		!filename.includes(' .') &&
		!filename.includes('. ') &&
		!filename.startsWith('-') &&
		filename.length <= CODE_BLOCK.FILE_NAME_MAX_LENGTH &&
		filename.length - dotPos - 1 <= CODE_BLOCK.FILE_NAME_MAX_EXT_LENGTH &&
		CODE_BLOCK.FILE_NAME_VALID_CHAR_REGEX.test(filename.substring(0, dotPos)) &&
		CODE_BLOCK.FILE_EXT_VALID_CHAR_REGEX.test(filename.substring(dotPos + 1))
	);
}
