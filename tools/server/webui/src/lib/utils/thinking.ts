/**
 * Parses thinking content from a message that may contain <think> tags
 * Returns an object with thinking content and cleaned message content
 */
export function parseThinkingContent(content: string): {
	thinking: string | null;
	cleanContent: string;
} {
	const thinkRegex = /<think>([\s\S]*?)<\/think>/;
	const match = content.match(thinkRegex);
	
	if (match) {
		return {
			thinking: match[1].trim(),
			cleanContent: content.replace(thinkRegex, '').trim()
		};
	}
	
	return {
		thinking: null,
		cleanContent: content
	};
}

/**
 * Checks if content contains an opening <think> tag (for streaming)
 */
export function hasThinkingStart(content: string): boolean {
	return content.includes('<think>');
}

/**
 * Checks if content contains a closing </think> tag (for streaming)
 */
export function hasThinkingEnd(content: string): boolean {
	return content.includes('</think>');
}

/**
 * Extracts partial thinking content during streaming
 * Used when we have <think> but not yet </think>
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
