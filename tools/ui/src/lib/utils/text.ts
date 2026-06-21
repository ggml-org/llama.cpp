import { NEWLINE_SEPARATOR } from '$lib/constants';

/**
 * Returns a shortened preview of the provided content capped at the given length.
 * Appends an ellipsis when the content exceeds the maximum.
 */
export function getPreviewText(content: string, max = 150): string {
	return content.length > max ? content.slice(0, max) + '...' : content;
}

/**
 * Generates a single-line title from a potentially multi-line prompt.
 * Uses the first non-empty line if `useFirstLine` is true.
 */
export function generateConversationTitle(content: string, useFirstLine: boolean = false): string {
	if (useFirstLine) {
		const firstLine = content.split(NEWLINE_SEPARATOR).find((line) => line.trim().length > 0);
		return firstLine ? firstLine.trim() : content.trim();
	}

	return content.trim();
}
