import { THINKING_FORMATS } from '$lib/constants/thinking-formats';

/**
 * Parses thinking content from a message that may contain thinking tags
 * Returns an object with thinking content and cleaned message content
 * Handles both complete blocks and incomplete blocks (streaming)
 * Supports formats: <think>...</think>, [THINK]...[/THINK], and ◁think▷...◁/think▷
 * @param content - The message content to parse
 * @returns An object containing the extracted thinking content and the cleaned message content
 */
export function parseThinkingContent(content: string): {
	thinking: string | null;
	cleanContent: string;
} {
	// Check for incomplete blocks (streaming case)
	for (const format of THINKING_FORMATS) {
		const hasStart = content.includes(format.startTag);
		const hasEnd = content.includes(format.endTag);

		if (hasStart && !hasEnd) {
			const cleanContent = content.split(format.endTag)?.[1]?.trim();
			const thinkingContent = content.split(format.startTag)?.[1]?.trim();

			return {
				cleanContent,
				thinking: thinkingContent
			};
		}
	}

	// Check for complete blocks
	for (const format of THINKING_FORMATS) {
		const match = content.match(format.regex);

		if (match) {
			const thinkingContent = match[1]?.trim() ?? '';
			const cleanContent = `${content.slice(0, match.index ?? 0)}${content.slice(
				(match.index ?? 0) + match[0].length
			)}`.trim();

			return {
				thinking: thinkingContent,
				cleanContent
			};
		}
	}

	return {
		thinking: null,
		cleanContent: content
	};
}

/**
 * Checks if content contains an opening thinking tag (for streaming)
 * @param content - The message content to check
 * @returns True if the content contains an opening thinking tag
 */
export function hasThinkingStart(content: string): boolean {
	return (
		THINKING_FORMATS.some((format) => content.includes(format.startTag)) ||
		content.includes('<|channel|>analysis')
	);
}

/**
 * Checks if content contains a closing thinking tag (for streaming)
 * @param content - The message content to check
 * @returns True if the content contains a closing thinking tag
 */
export function hasThinkingEnd(content: string): boolean {
	return THINKING_FORMATS.some((format) => content.includes(format.endTag));
}

/**
 * Extracts partial thinking content during streaming
 * Used when we have opening tag but not yet closing tag
 * @param content - The message content to extract partial thinking from
 * @returns An object containing the extracted partial thinking content and the remaining content
 */
export function extractPartialThinking(content: string): {
	thinking: string | null;
	remainingContent: string;
} {
	// Find all format positions and determine which appears first
	const formatPositions = THINKING_FORMATS.map((format) => ({
		...format,
		startIndex: content.indexOf(format.startTag),
		endIndex: content.indexOf(format.endTag)
	}))
		.filter((format) => format.startIndex !== -1)
		.sort((a, b) => a.startIndex - b.startIndex);

	const firstFormat = formatPositions[0];

	if (firstFormat && firstFormat.endIndex === -1) {
		// We have an opening tag but no closing tag (streaming case)
		const thinkingStart = firstFormat.startIndex + firstFormat.startTag.length;

		return {
			thinking: content.substring(thinkingStart),
			remainingContent: content.substring(0, firstFormat.startIndex)
		};
	}

	if (!firstFormat) {
		return { thinking: null, remainingContent: content };
	}

	// If we have both start and end tags, use the main parsing function
	const parsed = parseThinkingContent(content);

	return {
		thinking: parsed.thinking,
		remainingContent: parsed.cleanContent
	};
}
