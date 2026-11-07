/**
 * Pure operations over the conversation fork tree and the message tree rows.
 *
 * Conversations form a fork tree through `forkedFromConversationId`; messages
 * form a tree through `parent` / `children`. These helpers compute the rows to
 * write, so any storage backend can persist them without re-implementing the
 * tree rules.
 */

import { filterByLeafNodeId } from './branching';
import { uuid } from './uuid';
import type { McpServerOverride } from '$lib/types/database';

/**
 * Collects the ids of every fork descending from `rootId`, at any depth.
 * `rootId` itself is not included.
 */
export function collectForkDescendantIds(
	conversations: readonly DatabaseConversation[],
	rootId: string
): string[] {
	const ids: string[] = [];
	const queue = [rootId];

	while (queue.length > 0) {
		const parentId = queue.pop()!;

		for (const conv of conversations) {
			if (conv.forkedFromConversationId !== parentId) continue;

			ids.push(conv.id);
			queue.push(conv.id);
		}
	}

	return ids;
}

/**
 * Computes the rows that reparent the direct forks of `parentId` to its
 * nearest ancestor not listed in `excludeIds` (or promote them to top-level).
 * Forks whose own id is in `excludeIds` are left out, since they are deleted
 * in the same batch.
 */
export function reparentDirectForks(
	conversations: readonly DatabaseConversation[],
	parentId: string,
	excludeIds: ReadonlySet<string> = new Set()
): DatabaseConversation[] {
	const byId = new Map(conversations.map((conv) => [conv.id, conv]));
	const conv = byId.get(parentId);

	if (!conv) return [];

	let newParent = conv.forkedFromConversationId;

	const visited = new Set<string>([parentId]);

	while (newParent && excludeIds.has(newParent)) {
		if (visited.has(newParent)) {
			newParent = undefined;

			break;
		}

		visited.add(newParent);
		const next = byId.get(newParent);

		if (!next) {
			newParent = undefined;

			break;
		}

		newParent = next.forkedFromConversationId;
	}

	const updates: DatabaseConversation[] = [];

	for (const child of conversations) {
		if (child.forkedFromConversationId !== parentId || excludeIds.has(child.id)) continue;

		updates.push({ ...child, forkedFromConversationId: newParent });
	}

	return updates;
}

/**
 * Builds a fork of `sourceConv` holding the path from the root to
 * `atMessageId`, with fresh ids for the conversation and every message.
 *
 * @throws Error if no path to `atMessageId` can be resolved
 */
export function buildForkedConversation(
	sourceConv: DatabaseConversation,
	messages: readonly DatabaseMessage[],
	atMessageId: string,
	options: { name: string; includeAttachments: boolean }
): { conv: DatabaseConversation; messages: DatabaseMessage[] } {
	const pathMessages = filterByLeafNodeId(messages, atMessageId, true) as DatabaseMessage[];

	if (pathMessages.length === 0) {
		throw new Error(`Could not resolve message path to ${atMessageId}`);
	}

	const idMap = new Map<string, string>();

	for (const msg of pathMessages) {
		idMap.set(msg.id, uuid());
	}

	const newConvId = uuid();
	const clonedMessages: DatabaseMessage[] = pathMessages.map((msg) => {
		const newId = idMap.get(msg.id)!;
		const newParent = msg.parent ? (idMap.get(msg.parent) ?? null) : null;
		const newChildren = msg.children
			.filter((childId: string) => idMap.has(childId))
			.map((childId: string) => idMap.get(childId)!);

		return {
			...msg,
			children: newChildren,
			convId: newConvId,
			extra: options.includeAttachments ? msg.extra : undefined,
			id: newId,
			parent: newParent
		};
	});
	const lastClonedMessage = clonedMessages[clonedMessages.length - 1];
	const conv: DatabaseConversation = {
		currNode: lastClonedMessage.id,
		cwd: sourceConv.cwd,
		forkedFromConversationId: sourceConv.id,
		id: newConvId,
		lastModified: Date.now(),
		mcpServerOverrides: sourceConv.mcpServerOverrides
			? sourceConv.mcpServerOverrides.map((o: McpServerOverride) => ({
					enabled: o.enabled,
					serverId: o.serverId
				}))
			: undefined,
		name: options.name
	};

	return { conv, messages: clonedMessages };
}

/**
 * Groups messages by conversation, each group sorted by timestamp (oldest first).
 */
export function groupMessagesByConversation(
	messages: readonly DatabaseMessage[]
): Map<string, DatabaseMessage[]> {
	const groups = new Map<string, DatabaseMessage[]>();

	for (const msg of messages) {
		const bucket = groups.get(msg.convId);

		if (bucket) bucket.push(msg);
		else groups.set(msg.convId, [msg]);
	}

	for (const bucket of groups.values()) {
		bucket.sort((a, b) => a.timestamp - b.timestamp);
	}

	return groups;
}
