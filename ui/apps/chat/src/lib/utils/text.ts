import { stripMentionLinks } from './mention-badge';
import { NEWLINE } from '#lib/constants/index.ts';

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
 *
 * `@`-mentions travel as `[name](file://path)` links, which would otherwise
 * dominate a short title, so each one collapses to its `name`.
 */
export function generateConversationTitle(content: string, useFirstLine: boolean = false): string {
	const stripped = stripMentionLinks(content);

	if (useFirstLine) {
		const firstLine = stripped.split(NEWLINE).find((line) => line.trim().length > 0);

		return firstLine ? firstLine.trim() : stripped.trim();
	}

	return stripped.trim();
}
