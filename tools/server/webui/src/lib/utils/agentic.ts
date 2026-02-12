import { AgenticSectionType } from '$lib/enums';
import {
	AGENTIC_TAGS,
	AGENTIC_REGEX,
	REASONING_TAGS,
	TRIM_NEWLINES_REGEX
} from '$lib/constants/agentic';

/**
 * Represents a parsed section of agentic content
 */
export interface AgenticSection {
	type: AgenticSectionType;
	content: string;
	toolName?: string;
	toolArgs?: string;
	toolResult?: string;
}

/**
 * Represents a segment of content that may contain reasoning blocks
 */
type ReasoningSegment = {
	type:
		| AgenticSectionType.TEXT
		| AgenticSectionType.REASONING
		| AgenticSectionType.REASONING_PENDING;
	content: string;
};

/**
 * Parses agentic content into structured sections
 *
 * Main parsing function that processes content containing:
 * - Tool calls (completed, pending, or streaming)
 * - Reasoning blocks (completed or streaming)
 * - Regular text content
 *
 * The parser handles chronological display of agentic flow output, maintaining
 * the order of operations and properly identifying different states of tool calls
 * and reasoning blocks during streaming.
 *
 * @param rawContent - The raw content string to parse
 * @returns Array of structured agentic sections ready for display
 *
 * @example
 * ```typescript
 * const content = "Some text <<<AGENTIC_TOOL_CALL>>>tool_name...";
 * const sections = parseAgenticContent(content);
 * // Returns: [{ type: 'text', content: 'Some text' }, { type: 'tool_call_streaming', ... }]
 * ```
 */
export function parseAgenticContent(rawContent: string): AgenticSection[] {
	if (!rawContent) return [];

	const segments = splitReasoningSegments(rawContent);
	const sections: AgenticSection[] = [];

	for (const segment of segments) {
		if (segment.type === AgenticSectionType.TEXT) {
			sections.push(...parseToolCallContent(segment.content));
			continue;
		}

		if (segment.type === AgenticSectionType.REASONING) {
			if (segment.content.trim()) {
				sections.push({ type: AgenticSectionType.REASONING, content: segment.content });
			}
			continue;
		}

		sections.push({
			type: AgenticSectionType.REASONING_PENDING,
			content: segment.content
		});
	}

	return sections;
}

/**
 * Parses content containing tool call markers
 *
 * Identifies and extracts tool calls from content, handling:
 * - Completed tool calls with name, arguments, and results
 * - Pending tool calls (execution in progress)
 * - Streaming tool calls (arguments being received)
 * - Early-stage tool calls (just started)
 *
 * @param rawContent - The raw content string to parse
 * @returns Array of agentic sections representing tool calls and text
 */
function parseToolCallContent(rawContent: string): AgenticSection[] {
	if (!rawContent) return [];

	const sections: AgenticSection[] = [];

	const completedToolCallRegex = new RegExp(AGENTIC_REGEX.COMPLETED_TOOL_CALL.source, 'g');

	let lastIndex = 0;
	let match;

	while ((match = completedToolCallRegex.exec(rawContent)) !== null) {
		if (match.index > lastIndex) {
			const textBefore = rawContent.slice(lastIndex, match.index).trim();
			if (textBefore) {
				sections.push({ type: AgenticSectionType.TEXT, content: textBefore });
			}
		}

		const toolName = match[1];
		const toolArgs = match[2];
		const toolResult = match[3].replace(TRIM_NEWLINES_REGEX, '');

		sections.push({
			type: AgenticSectionType.TOOL_CALL,
			content: toolResult,
			toolName,
			toolArgs,
			toolResult
		});

		lastIndex = match.index + match[0].length;
	}

	const remainingContent = rawContent.slice(lastIndex);

	const pendingMatch = remainingContent.match(AGENTIC_REGEX.PENDING_TOOL_CALL);
	const partialWithNameMatch = remainingContent.match(AGENTIC_REGEX.PARTIAL_WITH_NAME);
	const earlyMatch = remainingContent.match(AGENTIC_REGEX.EARLY_MATCH);

	if (pendingMatch) {
		const pendingIndex = remainingContent.indexOf(AGENTIC_TAGS.TOOL_CALL_START);
		if (pendingIndex > 0) {
			const textBefore = remainingContent.slice(0, pendingIndex).trim();
			if (textBefore) {
				sections.push({ type: AgenticSectionType.TEXT, content: textBefore });
			}
		}

		const toolName = pendingMatch[1];
		const toolArgs = pendingMatch[2];
		const streamingResult = (pendingMatch[3] || '').replace(TRIM_NEWLINES_REGEX, '');

		sections.push({
			type: AgenticSectionType.TOOL_CALL_PENDING,
			content: streamingResult,
			toolName,
			toolArgs,
			toolResult: streamingResult || undefined
		});
	} else if (partialWithNameMatch) {
		const pendingIndex = remainingContent.indexOf(AGENTIC_TAGS.TOOL_CALL_START);
		if (pendingIndex > 0) {
			const textBefore = remainingContent.slice(0, pendingIndex).trim();
			if (textBefore) {
				sections.push({ type: AgenticSectionType.TEXT, content: textBefore });
			}
		}

		const partialArgs = partialWithNameMatch[2] || '';

		sections.push({
			type: AgenticSectionType.TOOL_CALL_STREAMING,
			content: '',
			toolName: partialWithNameMatch[1],
			toolArgs: partialArgs || undefined,
			toolResult: undefined
		});
	} else if (earlyMatch) {
		const pendingIndex = remainingContent.indexOf(AGENTIC_TAGS.TOOL_CALL_START);
		if (pendingIndex > 0) {
			const textBefore = remainingContent.slice(0, pendingIndex).trim();
			if (textBefore) {
				sections.push({ type: AgenticSectionType.TEXT, content: textBefore });
			}
		}

		const nameMatch = earlyMatch[1]?.match(AGENTIC_REGEX.TOOL_NAME_EXTRACT);

		sections.push({
			type: AgenticSectionType.TOOL_CALL_STREAMING,
			content: '',
			toolName: nameMatch?.[1],
			toolArgs: undefined,
			toolResult: undefined
		});
	} else if (lastIndex < rawContent.length) {
		let remainingText = rawContent.slice(lastIndex).trim();

		const partialMarkerMatch = remainingText.match(AGENTIC_REGEX.PARTIAL_MARKER);
		if (partialMarkerMatch) {
			remainingText = remainingText.slice(0, partialMarkerMatch.index).trim();
		}

		if (remainingText) {
			sections.push({ type: AgenticSectionType.TEXT, content: remainingText });
		}
	}

	if (sections.length === 0 && rawContent.trim()) {
		sections.push({ type: AgenticSectionType.TEXT, content: rawContent });
	}

	return sections;
}

/**
 * Strips partial marker from text content
 *
 * Removes incomplete agentic markers (e.g., "<<<", "<<<AGENTIC") that may appear
 * at the end of streaming content.
 *
 * @param text - The text content to process
 * @returns Text with partial markers removed
 */
function stripPartialMarker(text: string): string {
	const partialMarkerMatch = text.match(AGENTIC_REGEX.PARTIAL_MARKER);
	if (partialMarkerMatch) {
		return text.slice(0, partialMarkerMatch.index).trim();
	}
	return text;
}

/**
 * Splits raw content into segments based on reasoning blocks
 *
 * Identifies and extracts reasoning content wrapped in REASONING_TAGS.START/END markers,
 * separating it from regular text content. Handles both complete and incomplete
 * (streaming) reasoning blocks.
 *
 * @param rawContent - The raw content string to parse
 * @returns Array of reasoning segments with their types and content
 */
function splitReasoningSegments(rawContent: string): ReasoningSegment[] {
	if (!rawContent) return [];

	const segments: ReasoningSegment[] = [];
	let cursor = 0;

	while (cursor < rawContent.length) {
		const startIndex = rawContent.indexOf(REASONING_TAGS.START, cursor);
		if (startIndex === -1) {
			const remainingText = rawContent.slice(cursor);
			if (remainingText) {
				segments.push({ type: AgenticSectionType.TEXT, content: remainingText });
			}
			break;
		}

		if (startIndex > cursor) {
			const textBefore = rawContent.slice(cursor, startIndex);
			if (textBefore) {
				segments.push({ type: AgenticSectionType.TEXT, content: textBefore });
			}
		}

		const contentStart = startIndex + REASONING_TAGS.START.length;
		const endIndex = rawContent.indexOf(REASONING_TAGS.END, contentStart);

		if (endIndex === -1) {
			const pendingContent = rawContent.slice(contentStart);
			segments.push({
				type: AgenticSectionType.REASONING_PENDING,
				content: stripPartialMarker(pendingContent)
			});
			break;
		}

		const reasoningContent = rawContent.slice(contentStart, endIndex);
		segments.push({ type: AgenticSectionType.REASONING, content: reasoningContent });
		cursor = endIndex + REASONING_TAGS.END.length;
	}

	return segments;
}
