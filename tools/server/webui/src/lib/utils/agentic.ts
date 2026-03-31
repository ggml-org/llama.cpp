import { AgenticSectionType, MessageRole } from '$lib/enums';
import { ATTACHMENT_SAVED_REGEX, NEWLINE_SEPARATOR } from '$lib/constants';
import type { ApiChatCompletionToolCall } from '$lib/types/api';
import type {
	DatabaseMessage,
	DatabaseMessageExtra,
	DatabaseMessageExtraImageFile
} from '$lib/types/database';
import { AttachmentType } from '$lib/enums';

/**
 * Represents a parsed section of agentic content for display
 */
export interface AgenticSection {
	type: AgenticSectionType;
	content: string;
	toolName?: string;
	toolArgs?: string;
	toolResult?: string;
	toolResultExtras?: DatabaseMessageExtra[];
}

/**
 * Represents a tool result line that may reference an image attachment
 */
export type ToolResultLine = {
	text: string;
	image?: DatabaseMessageExtraImageFile;
};

/**
 * Derives display sections from a single assistant message and its direct tool results.
 *
 * @param message - The assistant message
 * @param toolMessages - Tool result messages for this assistant's tool_calls
 * @param streamingToolCalls - Partial tool calls during streaming (not yet persisted)
 */
function deriveSingleTurnSections(
	message: DatabaseMessage,
	toolMessages: DatabaseMessage[] = [],
	streamingToolCalls: ApiChatCompletionToolCall[] = []
): AgenticSection[] {
	const sections: AgenticSection[] = [];

	// 1. Reasoning content (from dedicated field)
	if (message.reasoningContent) {
		sections.push({
			type: AgenticSectionType.REASONING,
			content: message.reasoningContent
		});
	}

	// 2. Text content
	if (message.content?.trim()) {
		sections.push({
			type: AgenticSectionType.TEXT,
			content: message.content
		});
	}

	// 3. Persisted tool calls (from message.toolCalls field)
	const toolCalls = parseToolCalls(message.toolCalls);
	for (const tc of toolCalls) {
		const resultMsg = toolMessages.find((m) => m.toolCallId === tc.id);
		sections.push({
			type: resultMsg ? AgenticSectionType.TOOL_CALL : AgenticSectionType.TOOL_CALL_PENDING,
			content: resultMsg?.content || '',
			toolName: tc.function?.name,
			toolArgs: tc.function?.arguments,
			toolResult: resultMsg?.content,
			toolResultExtras: resultMsg?.extra
		});
	}

	// 4. Streaming tool calls (not yet persisted - currently being received)
	for (const tc of streamingToolCalls) {
		// Skip if already in persisted tool calls
		if (tc.id && toolCalls.find((t) => t.id === tc.id)) continue;
		sections.push({
			type: AgenticSectionType.TOOL_CALL_STREAMING,
			content: '',
			toolName: tc.function?.name,
			toolArgs: tc.function?.arguments
		});
	}

	return sections;
}

/**
 * Derives display sections from structured message data.
 *
 * Handles both single-turn (one assistant + its tool results) and multi-turn
 * agentic sessions (multiple assistant + tool messages grouped together).
 *
 * When `toolMessages` contains continuation assistant messages (from multi-turn
 * agentic flows), they are processed in order to produce sections across all turns.
 *
 * @param message - The first/anchor assistant message
 * @param toolMessages - Tool result messages and continuation assistant messages
 * @param streamingToolCalls - Partial tool calls during streaming (not yet persisted)
 * @param isStreaming - Whether the message is currently being streamed
 */
export function deriveAgenticSections(
	message: DatabaseMessage,
	toolMessages: DatabaseMessage[] = [],
	streamingToolCalls: ApiChatCompletionToolCall[] = []
): AgenticSection[] {
	const hasAssistantContinuations = toolMessages.some((m) => m.role === MessageRole.ASSISTANT);

	if (!hasAssistantContinuations) {
		return deriveSingleTurnSections(message, toolMessages, streamingToolCalls);
	}

	const sections: AgenticSection[] = [];

	const firstTurnToolMsgs = collectToolMessages(toolMessages, 0);
	sections.push(...deriveSingleTurnSections(message, firstTurnToolMsgs));

	let i = firstTurnToolMsgs.length;

	while (i < toolMessages.length) {
		const msg = toolMessages[i];

		if (msg.role === MessageRole.ASSISTANT) {
			const turnToolMsgs = collectToolMessages(toolMessages, i + 1);
			const isLastTurn = i + 1 + turnToolMsgs.length >= toolMessages.length;

			sections.push(
				...deriveSingleTurnSections(msg, turnToolMsgs, isLastTurn ? streamingToolCalls : [])
			);

			i += 1 + turnToolMsgs.length;
		} else {
			i++;
		}
	}

	return sections;
}

/**
 * Collect consecutive tool messages starting at `startIndex`.
 */
function collectToolMessages(messages: DatabaseMessage[], startIndex: number): DatabaseMessage[] {
	const result: DatabaseMessage[] = [];

	for (let i = startIndex; i < messages.length; i++) {
		if (messages[i].role === MessageRole.TOOL) {
			result.push(messages[i]);
		} else {
			break;
		}
	}

	return result;
}

/**
 * Parse tool result text into lines, matching image attachments by name.
 */
export function parseToolResultWithImages(
	toolResult: string,
	extras?: DatabaseMessageExtra[]
): ToolResultLine[] {
	const lines = toolResult.split(NEWLINE_SEPARATOR);
	return lines.map((line) => {
		const match = line.match(ATTACHMENT_SAVED_REGEX);
		if (!match || !extras) return { text: line };

		const attachmentName = match[1];
		const image = extras.find(
			(e): e is DatabaseMessageExtraImageFile =>
				e.type === AttachmentType.IMAGE && e.name === attachmentName
		);

		return { text: line, image };
	});
}

/**
 * Safely parse the toolCalls JSON string from a DatabaseMessage.
 */
function parseToolCalls(toolCallsJson?: string): ApiChatCompletionToolCall[] {
	if (!toolCallsJson) return [];

	try {
		const parsed = JSON.parse(toolCallsJson);

		return Array.isArray(parsed) ? parsed : [];
	} catch {
		return [];
	}
}

/**
 * Check if a message has agentic content (tool calls or is part of an agentic flow).
 */
export function hasAgenticContent(
	message: DatabaseMessage,
	toolMessages: DatabaseMessage[] = []
): boolean {
	if (message.toolCalls) {
		const tc = parseToolCalls(message.toolCalls);

		if (tc.length > 0) return true;
	}

	return toolMessages.length > 0;
}
