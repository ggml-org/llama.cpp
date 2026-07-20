import {
	AgenticSectionType,
	AttachmentType,
	ContinueIntentKind,
	MessageRole,
	ToolResultKind
} from '$lib/enums';
import {
	ATTACHMENT_SAVED_REGEX,
	MARKDOWN_ATX_HEADING_REGEX,
	MARKDOWN_BOLD_REGEX,
	MARKDOWN_BLOCKQUOTE_REGEX,
	MARKDOWN_CODE_FENCE_REGEX,
	MARKDOWN_LINK_REGEX,
	MARKDOWN_LIST_BULLET_REGEX,
	MARKDOWN_LIST_NUMBERED_REGEX,
	MARKDOWN_TABLE_SEPARATOR_REGEX,
	NEWLINE,
	REASONING_TAGS,
	SEARCH_SUMMARY_SEPARATOR,
	SEARCH_SUMMARY_TOTAL_REGEX,
	TOOL_RESULT_JSON_OPEN_REGEX
} from '$lib/constants';
import type { ApiChatCompletionToolCall } from '$lib/types/api';
import type {
	DatabaseMessage,
	DatabaseMessageExtra,
	DatabaseMessageExtraImageFile
} from '$lib/types/database';

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
	/** ID of the model-side tool call (matches tool_calls[i].id). Lets
	 *  downstream consumers correlate a section with the agentic loop's
	 *  currently-executing tool, e.g. to drive live-streaming UI state
	 *  by matching against agenticStore.executingToolCallId. */
	toolCallId?: string;
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
			toolResultExtras: resultMsg?.extra,
			toolCallId: tc.id
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
			toolArgs: tc.function?.arguments,
			toolCallId: tc.id
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
 * Build the raw text representation shown in the "raw output" view of an
 * assistant message. Each section is formatted as it would appear in the
 * model-facing transcript, joined by blank lines.
 */
export function buildAssistantRawOutput(sections: AgenticSection[]): string {
	const parts: string[] = [];

	for (const section of sections) {
		switch (section.type) {
			case AgenticSectionType.REASONING:
			case AgenticSectionType.REASONING_PENDING:
				parts.push(`${REASONING_TAGS.START}${NEWLINE}${section.content}${REASONING_TAGS.END}`);
				break;

			case AgenticSectionType.TEXT:
				parts.push(section.content);
				break;

			case AgenticSectionType.TOOL_CALL:
			case AgenticSectionType.TOOL_CALL_PENDING:
			case AgenticSectionType.TOOL_CALL_STREAMING: {
				const callObj: Record<string, unknown> = { name: section.toolName };

				if (section.toolArgs) {
					try {
						callObj.arguments = JSON.parse(section.toolArgs);
					} catch {
						callObj.arguments = section.toolArgs;
					}
				}

				parts.push(JSON.stringify(callObj, null, 2));

				if (section.toolResult) {
					parts.push(`${NEWLINE}${section.toolResult}`);
				}

				break;
			}
		}
	}

	return parts.join(`${NEWLINE}${NEWLINE}`);
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
 * Split a tool-result blob into a list and an optional "Total matches: N"
 * summary. Both file-glob and grep tools emit this format on the server:
 *
 *   <matches>
 *   ---
 *   Total matches: 42
 *
 * Returns the lines and exposes a callback for capturing the total so each
 * caller can stash it on its own meta type without taking a return-tuple.
 */
export function splitSearchSummaryList(
	text: string,
	captureTotal: (n: number) => void
): { lines: string[] } {
	const separatorIndex = text.indexOf(SEARCH_SUMMARY_SEPARATOR);
	const matchesText = separatorIndex === -1 ? text : text.slice(0, separatorIndex);
	const summaryText =
		separatorIndex === -1 ? '' : text.slice(separatorIndex + SEARCH_SUMMARY_SEPARATOR.length);

	const totalMatch = summaryText.match(SEARCH_SUMMARY_TOTAL_REGEX);
	if (totalMatch) {
		captureTotal(parseInt(totalMatch[1], 10));
	}

	const lines = matchesText
		.split(NEWLINE)
		.map((line) => line.trim())
		.filter((line) => line.length > 0);

	return { lines };
}

/**
 * Parse tool result text into lines, matching image attachments by name.
 */
export function parseToolResultWithImages(
	toolResult: string,
	extras?: DatabaseMessageExtra[]
): ToolResultLine[] {
	const lines = toolResult.split(NEWLINE);
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
 * Pick a renderer tier for a tool's result content.
 *
 *   json     - trimmed content starts with `{` or `[` and parses cleanly.
 *   markdown - content shows structural markdown markers (headers, code
 *              fences, links, lists, blockquotes, tables) and should render
 *              through MarkdownContent for proper formatting.
 *   text     - everything else, rendered as plain text lines (with image
 *              attachment resolution as a side effect).
 */
export function classifyToolResult(content: string | undefined): ToolResultKind {
	if (!content) return ToolResultKind.TEXT;
	const trimmed = content.trim();
	if (!trimmed) return ToolResultKind.TEXT;

	// Strongest signal: JSON object/array round-trips through JSON.parse.
	if (TOOL_RESULT_JSON_OPEN_REGEX.test(trimmed)) {
		try {
			JSON.parse(trimmed);
			return ToolResultKind.JSON;
		} catch (error) {
			console.error('[agentic] tool result looked like JSON but failed to parse:', error);
		}
	}

	if (looksLikeMarkdown(trimmed)) return ToolResultKind.MARKDOWN;

	return ToolResultKind.TEXT;
}

/**
 * Heuristic detector for "is this content a markdown document rather than
 * plain text?". True when at least one well-known structural marker shows
 * up - headers, code fences, links, bold, lists, blockquotes, tables.
 * Each marker is specific enough that plain tool-output prose rarely
 * trips it, but plain text starting with `# 5` will - acceptable false
 * positive for the gain in formatting for tool results like search
 * summaries that come back already-mardown.
 */
function looksLikeMarkdown(content: string): boolean {
	// Code fences are unambiguous - triple backticks or tildes at line start.
	if (MARKDOWN_CODE_FENCE_REGEX.test(content)) return true;

	const lines = content.split(NEWLINE);

	for (const line of lines) {
		if (MARKDOWN_ATX_HEADING_REGEX.test(line)) return true;
		if (MARKDOWN_BLOCKQUOTE_REGEX.test(line)) return true;
		if (MARKDOWN_LIST_BULLET_REGEX.test(line)) return true;
		if (MARKDOWN_LIST_NUMBERED_REGEX.test(line)) return true;
	}

	// Inline structural markers anywhere in the body.
	if (MARKDOWN_LINK_REGEX.test(content)) return true;
	if (MARKDOWN_BOLD_REGEX.test(content)) return true;

	// Tables: a pipe-bearing header line followed by a separator row.
	if (lines.length >= 2) {
		const head = lines[0];
		const sep = lines[1];

		if (head.includes('|') && MARKDOWN_TABLE_SEPARATOR_REGEX.test(sep)) return true;
	}

	return false;
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
