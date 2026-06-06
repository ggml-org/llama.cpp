import { AgenticSectionType, ContinueIntentKind, MessageRole } from '$lib/enums';
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
	wasInterrupted?: boolean;
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
	streamingToolCalls: ApiChatCompletionToolCall[] = [],
	isStreaming: boolean = false
): AgenticSection[] {
	const sections: AgenticSection[] = [];

	// 1. Reasoning content (from dedicated field)
	if (message.reasoningContent) {
		const toolCalls = parseToolCalls(message.toolCalls);
		const hasContentAfterReasoning =
			!!message.content?.trim() || toolCalls.length > 0 || streamingToolCalls.length > 0;
		const isPending = isStreaming && !hasContentAfterReasoning;
		sections.push({
			type: isPending ? AgenticSectionType.REASONING_PENDING : AgenticSectionType.REASONING,
			content: message.reasoningContent,
			wasInterrupted: !isStreaming && !hasContentAfterReasoning
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
		// Only show as pending/loading if we're actively streaming; otherwise it's just a tool call without result
		const type = resultMsg
			? AgenticSectionType.TOOL_CALL
			: isStreaming
				? AgenticSectionType.TOOL_CALL_PENDING
				: AgenticSectionType.TOOL_CALL;
		sections.push({
			type,
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
	streamingToolCalls: ApiChatCompletionToolCall[] = [],
	isStreaming: boolean = false
): AgenticSection[] {
	const hasAssistantContinuations = toolMessages.some((m) => m.role === MessageRole.ASSISTANT);

	if (!hasAssistantContinuations) {
		return deriveSingleTurnSections(message, toolMessages, streamingToolCalls, isStreaming);
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
				...deriveSingleTurnSections(
					msg,
					turnToolMsgs,
					isLastTurn ? streamingToolCalls : [],
					isLastTurn && isStreaming
				)
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

/**
 * Classification of how a Continue click on an assistant message should resume
 * generation. The caller dispatches the resume path based on this value.
 *
 *   append_text  -> the target is a plain text turn, resume with
 *                   continue_final_message and rehydrate the persisted
 *                   tool_calls and attachments through the regular DB to API
 *                   message converter.
 *   rerun_turn   -> the target carries tool_calls that were never resolved by
 *                   tool result messages. The agentic stream was cut mid turn,
 *                   so we drop the target and rerun the loop from the previous
 *                   history. truncateAfter is the last kept index, inclusive.
 *   next_turn    -> the target's tool_calls were already resolved by trailing
 *                   tool results. Hand the history up to and including the
 *                   last consecutive tool result back to the agentic loop so it
 *                   starts the next turn naturally. truncateAfter points at
 *                   that last tool result.
 */
export type ContinueIntent =
	| { kind: ContinueIntentKind.APPEND_TEXT }
	| { kind: ContinueIntentKind.RERUN_TURN; truncateAfter: number }
	| { kind: ContinueIntentKind.NEXT_TURN; truncateAfter: number };

/**
 * Decide how a Continue click on messages[idx] should resume generation.
 * Pure function over the persisted history snapshot.
 */
export function classifyContinueIntent(messages: DatabaseMessage[], idx: number): ContinueIntent {
	const target = messages[idx];

	// Defensive default: callers already filter by role, stay deterministic.
	if (!target || target.role !== MessageRole.ASSISTANT) {
		return { kind: ContinueIntentKind.APPEND_TEXT };
	}

	const hasToolCalls = parseToolCalls(target.toolCalls).length > 0;
	if (!hasToolCalls) {
		return { kind: ContinueIntentKind.APPEND_TEXT };
	}

	// Walk consecutive trailing tool results. The agentic loop only emits tool
	// messages directly after the assistant turn that owns them, so the first
	// non tool message marks the boundary.
	let lastTrailingTool = idx;
	for (let i = idx + 1; i < messages.length; i++) {
		if (messages[i].role === MessageRole.TOOL) {
			lastTrailingTool = i;
		} else {
			break;
		}
	}

	if (lastTrailingTool > idx) {
		return { kind: ContinueIntentKind.NEXT_TURN, truncateAfter: lastTrailingTool };
	}

	return { kind: ContinueIntentKind.RERUN_TURN, truncateAfter: idx - 1 };
}
