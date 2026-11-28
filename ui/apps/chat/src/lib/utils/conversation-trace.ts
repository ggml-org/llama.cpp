/**
 * Builds the trace a share uploads: one branch of a conversation, from its start to a message.
 */

import {
	HF_PREVIEW_LINES_MAX,
	HF_PREVIEW_TEXT_MAX,
	type HfTracePreviewLine
} from '@ggml-org/llama-huggingface';
import { MessageRole, MessageType } from '#lib/enums/index.ts';
import { filterByLeafNodeId } from '#lib/utils/branching.ts';

const WHITESPACE = /\s+/g;
/** A fenced code block, or what is left of one the preview cut short. */
const FENCED_CODE = /```[\s\S]*?(?:```|$)/g;
const BACKTICKS = /`/g;

export interface ConversationTraceOptions {
	/** Last message of the trace; the conversation's current message when omitted. */
	upToMessageId?: string;
	includeAttachments?: boolean;
}

/**
 * Picks the branch that ends at `upToMessageId` out of the conversation's message tree.
 * The trace viewer reads messages as one line, so other branches are left out.
 * Local details (working directories, sidebar group) stay on this device.
 * @returns The trace as an exported conversation, ready for the JSONL serializer
 */
export function buildConversationTrace(
	data: ExportedConversation,
	options: ConversationTraceOptions = {}
): ExportedConversation {
	const { conv, messages } = data;
	const leafId = options.upToMessageId ?? conv.currNode ?? '';
	const path = filterByLeafNodeId(messages, leafId);
	const pathIds = new Set(path.map((message) => message.id));
	const traceMessages = path.map((message) => {
		const { extra, toolCwd: _toolCwd, ...rest } = message;
		const traceMessage: DatabaseMessage = {
			...rest,
			children: message.children.filter((id: string) => pathIds.has(id))
		};

		if (options.includeAttachments && extra?.length) traceMessage.extra = extra;

		return traceMessage;
	});
	const { cwd: _cwd, groupId: _groupId, ...sharedConv } = conv;

	return {
		conv: { ...sharedConv, currNode: traceMessages.at(-1)?.id ?? null },
		messages: traceMessages
	};
}

/**
 * The first messages of a trace, as the dialog shows them faded once the trace is shared:
 * what the user and the assistant said, in one line each, without code blocks. Tool calls,
 * reasoning, a compaction and messages the app made itself have no place in it.
 */
export function buildTracePreview(trace: ExportedConversation): HfTracePreviewLine[] {
	const lines: HfTracePreviewLine[] = [];

	for (const message of trace.messages) {
		if (lines.length >= HF_PREVIEW_LINES_MAX) break;

		if (message.isSynthetic || message.type !== MessageType.TEXT) continue;

		const role =
			message.role === MessageRole.USER
				? 'user'
				: message.role === MessageRole.ASSISTANT
					? 'assistant'
					: null;
		const text = message.content
			.replace(FENCED_CODE, ' ')
			.replace(BACKTICKS, '')
			.replace(WHITESPACE, ' ')
			.trim()
			.slice(0, HF_PREVIEW_TEXT_MAX);

		if (role && text) lines.push({ role, text });
	}

	return lines;
}
