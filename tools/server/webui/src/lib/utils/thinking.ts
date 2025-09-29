/**
 * Parses thinking content from a message that may contain <think> tags or [THINK] tags
 * Returns an object with thinking content and cleaned message content
 * Handles both complete blocks and incomplete blocks (streaming)
 * Supports formats: <think>...</think> and [THINK]...[/THINK]
 * @param content - The message content to parse
 * @returns An object containing the extracted thinking content and the cleaned message content
 */
export function parseThinkingContent(content: string): {
	thinking: string | null;
	cleanContent: string;
} {
	const incompleteThinkMatch = content.includes('<think>') && !content.includes('</think>');
	const incompleteThinkBracketMatch = content.includes('[THINK]') && !content.includes('[/THINK]');

	if (incompleteThinkMatch) {
		const cleanContent = content.split('</think>')?.[1]?.trim();
		const thinkingContent = content.split('<think>')?.[1]?.trim();

		return {
			cleanContent,
			thinking: thinkingContent
		};
	}

	if (incompleteThinkBracketMatch) {
		const cleanContent = content.split('[/THINK]')?.[1]?.trim();
		const thinkingContent = content.split('[THINK]')?.[1]?.trim();

		return {
			cleanContent,
			thinking: thinkingContent
		};
	}

	const completeThinkMatch = content.match(/<think>([\s\S]*?)<\/think>/);
	const completeThinkBracketMatch = content.match(/\[THINK\]([\s\S]*?)\[\/THINK\]/);

	if (completeThinkMatch) {
		const thinkingContent = completeThinkMatch[1]?.trim() ?? '';
		const cleanContent = `${content.slice(0, completeThinkMatch.index ?? 0)}${content.slice(
			(completeThinkMatch.index ?? 0) + completeThinkMatch[0].length
		)}`.trim();

		return {
			thinking: thinkingContent,
			cleanContent
		};
	}

	if (completeThinkBracketMatch) {
		const thinkingContent = completeThinkBracketMatch[1]?.trim() ?? '';
		const cleanContent = `${content.slice(0, completeThinkBracketMatch.index ?? 0)}${content.slice(
			(completeThinkBracketMatch.index ?? 0) + completeThinkBracketMatch[0].length
		)}`.trim();

		return {
			thinking: thinkingContent,
			cleanContent
		};
	}

	return {
		thinking: null,
		cleanContent: content
	};
}

/**
 * Checks if content contains an opening thinking tag (for streaming)
 * Supports both <think> and [THINK] formats
 * @param content - The message content to check
 * @returns True if the content contains an opening thinking tag
 */
export function hasThinkingStart(content: string): boolean {
	return (
		content.includes('<think>') ||
		content.includes('[THINK]') ||
		content.includes('<|channel|>analysis')
	);
}

/**
 * Checks if content contains a closing thinking tag (for streaming)
 * Supports both </think> and [/THINK] formats
 * @param content - The message content to check
 * @returns True if the content contains a closing thinking tag
 */
export function hasThinkingEnd(content: string): boolean {
	return content.includes('</think>') || content.includes('[/THINK]');
}

/**
 * Extracts partial thinking content during streaming
 * Supports both <think> and [THINK] formats
 * Used when we have opening tag but not yet closing tag
 * @param content - The message content to extract partial thinking from
 * @returns An object containing the extracted partial thinking content and the remaining content
 */
export function extractPartialThinking(content: string): {
	thinking: string | null;
	remainingContent: string;
} {
	const thinkStartIndex = content.indexOf('<think>');
	const thinkEndIndex = content.indexOf('</think>');

	const bracketStartIndex = content.indexOf('[THINK]');
	const bracketEndIndex = content.indexOf('[/THINK]');

	const useThinkFormat =
		thinkStartIndex !== -1 && (bracketStartIndex === -1 || thinkStartIndex < bracketStartIndex);
	const useBracketFormat =
		bracketStartIndex !== -1 && (thinkStartIndex === -1 || bracketStartIndex < thinkStartIndex);

	if (useThinkFormat) {
		if (thinkEndIndex === -1) {
			const thinkingStart = thinkStartIndex + '<think>'.length;

			return {
				thinking: content.substring(thinkingStart),
				remainingContent: content.substring(0, thinkStartIndex)
			};
		}
	} else if (useBracketFormat) {
		if (bracketEndIndex === -1) {
			const thinkingStart = bracketStartIndex + '[THINK]'.length;

			return {
				thinking: content.substring(thinkingStart),
				remainingContent: content.substring(0, bracketStartIndex)
			};
		}
	} else {
		return { thinking: null, remainingContent: content };
	}

	const parsed = parseThinkingContent(content);

	return {
		thinking: parsed.thinking,
		remainingContent: parsed.cleanContent
	};
}
