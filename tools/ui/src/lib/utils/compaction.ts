import { COMPACTION } from '$lib/constants';
import { MessageRole, MessageType } from '$lib/enums';
import type { DatabaseMessage } from '$lib/types';

/**
 * Applies the compaction boundary to a branch message list before it is
 * sent to the model. Messages before the last compaction node are dropped,
 * except system messages, which carry conversation-wide instructions. The
 * compaction node itself is emitted as a copy whose content frames the
 * stored summary, so the display row stays clean while the model receives
 * the framing.
 *
 * A list without a compaction node is returned as-is.
 */
export function sliceAtLastCompaction(messages: DatabaseMessage[]): DatabaseMessage[] {
	let boundary = -1;

	for (let i = messages.length - 1; i >= 0; i--) {
		// An empty compaction node is an interrupted compaction and carries
		// no summary, so it never acts as a boundary.
		if (messages[i].type === MessageType.COMPACTION && messages[i].content.trim().length > 0) {
			boundary = i;

			break;
		}
	}

	if (boundary === -1) return messages;

	const systemPrefix = messages
		.slice(0, boundary)
		.filter((m) => m.role === MessageRole.SYSTEM || m.type === MessageType.ROOT);
	const compactionNode = messages[boundary];
	const framedNode: DatabaseMessage = {
		...compactionNode,
		content: COMPACTION.CONTEXT_TEMPLATE.replace(
			COMPACTION.SUMMARY_PLACEHOLDER,
			compactionNode.content
		)
	};

	return [...systemPrefix, framedNode, ...messages.slice(boundary + 1)];
}

/**
 * A branch is compactable when at least one assistant turn with content
 * exists after the last compaction boundary: anything less has nothing
 * worth summarizing.
 */
export function canCompactMessages(messages: DatabaseMessage[]): boolean {
	for (let i = messages.length - 1; i >= 0; i--) {
		const m = messages[i];

		if (m.type === MessageType.COMPACTION && m.content.trim().length > 0) return false;

		if (m.role === MessageRole.ASSISTANT && m.content.trim().length > 0) return true;
	}

	return false;
}

/**
 * Returns the message carrying the current context size of the branch: the
 * latest assistant turn or compaction node holding timings. A compaction
 * node is stamped with the prompt timings of the reduced context by the
 * post-compaction pre-encode, so it supersedes earlier assistant turns.
 */
export function lastContextBearingMessage(
	messages: DatabaseMessage[]
): DatabaseMessage | undefined {
	for (let i = messages.length - 1; i >= 0; i--) {
		const m = messages[i];

		if (m.role === MessageRole.ASSISTANT && m.timings) return m;

		if (m.type === MessageType.COMPACTION && m.timings) return m;
	}

	return undefined;
}
