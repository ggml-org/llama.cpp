import { COMPACTION } from '$lib/constants';
import { MessageRole, MessageType } from '$lib/enums';
import type { DatabaseMessage } from '$lib/types';

/** Whether a message belongs to the prefix every request keeps first. */
function isPrefix(message: DatabaseMessage): boolean {
	return message.role === MessageRole.SYSTEM || message.type === MessageType.ROOT;
}

/** Whether a message is a compaction node holding its summary. */
function isBoundary(message: DatabaseMessage): boolean {
	return message.type === MessageType.COMPACTION && message.content.trim().length > 0;
}

/**
 * Where each step of a message list starts. A step is what the model reads
 * as one unit: a user message, or an assistant message with the tool
 * results that follow it, so a cut never separates a tool call from its
 * result. The prefix and the compaction nodes are no steps.
 */
function stepStarts(messages: DatabaseMessage[]): number[] {
	const starts: number[] = [];

	messages.forEach((m, i) => {
		if (isPrefix(m) || m.type === MessageType.COMPACTION || m.role === MessageRole.TOOL) return;

		starts.push(i);
	});

	return starts;
}

/**
 * Applies the compaction boundary to a branch message list before it is
 * sent to the model: the system prefix, then the last compaction node as a
 * copy whose content frames its summary, then the steps it kept right
 * before it, then everything after it. Older compaction nodes and the steps
 * they summarized are dropped. The display row stays clean while the model
 * receives the framing.
 *
 * A list without a compaction node is returned as-is.
 */
export function sliceAtLastCompaction(messages: DatabaseMessage[]): DatabaseMessage[] {
	let boundary = -1;

	for (let i = messages.length - 1; i >= 0; i--) {
		// An empty compaction node is an interrupted compaction and carries
		// no summary, so it never acts as a boundary.
		if (isBoundary(messages[i])) {
			boundary = i;

			break;
		}
	}

	if (boundary === -1) return messages;

	const before = messages.slice(0, boundary);
	const compactionNode = messages[boundary];
	const framedNode: DatabaseMessage = {
		...compactionNode,
		// a replacement function keeps the summary literal: a string would read
		// $& or $$ in it as patterns
		content: COMPACTION.CONTEXT_TEMPLATE.replace(
			COMPACTION.SUMMARY_PLACEHOLDER,
			() => compactionNode.content
		)
	};
	const steps = before.filter((m) => !isPrefix(m) && m.type !== MessageType.COMPACTION);
	const starts = stepStarts(steps);
	const kept = Math.min(compactionNode.kept ?? 0, starts.length);
	const from = kept > 0 ? starts[starts.length - kept] : steps.length;

	return [
		...before.filter(isPrefix),
		framedNode,
		...steps.slice(from),
		...messages.slice(boundary + 1)
	];
}

/**
 * Where a compaction of the list the model reads cuts it: the summary
 * request sends what comes before the cut, the very prefix the model last
 * read, so the server reuses its cache, and the preserved steps after it
 * stay as written. Null when no step lies before the preserved ones.
 */
export function compactionCut(view: DatabaseMessage[], preserve: number): number | null {
	const starts = stepStarts(view);
	const folded = starts.length - preserve;

	if (folded < 1) return null;

	return preserve > 0 ? starts[folded] : view.length;
}

/**
 * A branch is compactable when the list the model reads holds at least one
 * step before the preserved ones: anything less has nothing worth
 * summarizing.
 */
export function canCompactMessages(messages: DatabaseMessage[], preserve: number): boolean {
	return compactionCut(sliceAtLastCompaction(messages), preserve) !== null;
}

/**
 * The context size the compaction shrank: the timings of the last stamped
 * message ahead of the node, i.e. what the branch carried when the summary
 * was requested. Null while nothing ahead of the node is stamped.
 */
export function compactionTokensBefore(
	messages: DatabaseMessage[],
	node: DatabaseMessage
): number | null {
	const nodeIndex = messages.findIndex((m) => m.id === node.id);
	const timings =
		nodeIndex > 0 ? lastContextBearingMessage(messages.slice(0, nodeIndex))?.timings : undefined;

	if (!timings) return null;

	const tokens = (timings.prompt_n ?? 0) + (timings.cache_n ?? 0) + (timings.predicted_n ?? 0);

	return tokens > 0 ? tokens : null;
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
