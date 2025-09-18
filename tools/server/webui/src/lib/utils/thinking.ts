/**
 * Parses thinking content from a message that may contain <think> tags
 * Returns an object with thinking content and cleaned message content
 * Handles both complete <think>...</think> blocks and incomplete <think> blocks (streaming)
 * @param content - The message content to parse
 * @returns An object containing the extracted thinking content and the cleaned message content
 */
export function parseThinkingContent(content: string): {
	thinking: string | null;
	cleanContent: string;
} {
	const incompleteMatch = content.includes('<think>') && !content.includes('</think>');

	if (incompleteMatch) {
		// Remove the entire <think>... part from clean content
		const cleanContent = content.split('</think>')?.[1]?.trim();
		// Extract everything after <think> as thinking content
		const thinkingContent = content.split('<think>')?.[1]?.trim();

		return {
			cleanContent,
			thinking: thinkingContent
		};
	}

	const completeMatch = content.includes('</think>');

	if (completeMatch) {
		return {
			thinking: content.split('</think>')?.[0]?.trim(),
			cleanContent: content.split('</think>')?.[1]?.trim()
		};
	}

	return {
		thinking: null,
		cleanContent: content
	};
}

/**
 * Checks if content contains an opening <think> tag (for streaming)
 * @param content - The message content to check
 * @returns True if the content contains an opening <think> tag
 */
export function hasThinkingStart(content: string): boolean {
	return content.includes('<think>') || content.includes('<|channel|>analysis');
}

/**
 * Checks if content contains a closing </think> tag (for streaming)
 * @param content - The message content to check
 * @returns True if the content contains a closing </think> tag
 */
export function hasThinkingEnd(content: string): boolean {
	return content.includes('</think>');
}

/**
 * Extracts partial thinking content during streaming
 * Used when we have <think> but not yet </think>
 * @param content - The message content to extract partial thinking from
 * @returns An object containing the extracted partial thinking content and the remaining content
 */
export function extractPartialThinking(content: string): {
	thinking: string | null;
	remainingContent: string;
} {
	const startIndex = content.indexOf('<think>');
	if (startIndex === -1) {
		return { thinking: null, remainingContent: content };
	}

	const endIndex = content.indexOf('</think>');
	if (endIndex === -1) {
		// Still streaming thinking content
		const thinkingStart = startIndex + '<think>'.length;
		return {
			thinking: content.substring(thinkingStart),
			remainingContent: content.substring(0, startIndex)
		};
	}

	// Complete thinking block found
	const parsed = parseThinkingContent(content);
	return {
		thinking: parsed.thinking,
		remainingContent: parsed.cleanContent
	};
}
