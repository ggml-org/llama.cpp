/**
 * Builds the trace a share uploads: one branch of a conversation, from its start to a message.
 */

import { filterByLeafNodeId } from './branching';

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
