/**
 * Utility functions for conversation data manipulation
 */
import { MessageRole } from '$lib/enums';
import type { DatabaseMessage } from '$lib/types';

/**
 * Model that generated the conversation's latest assistant message, or null
 * when no assistant message carries one.
 */
export function getConversationModel(messages: readonly DatabaseMessage[]): string | null {
	for (let i = messages.length - 1; i >= 0; i--) {
		const message = messages[i];

		if (message.role === MessageRole.ASSISTANT && message.model) return message.model;
	}

	return null;
}

/**
 * Creates a map of conversation IDs to their message counts from exported conversation data
 * @param exportedData - Array of exported conversations with their messages
 * @returns Map of conversation ID to message count
 */
export function createMessageCountMap(
	exportedData: Array<{ conv: DatabaseConversation; messages: DatabaseMessage[] }>
): Map<string, number> {
	const countMap = new Map<string, number>();

	for (const item of exportedData) {
		countMap.set(item.conv.id, item.messages.length);
	}

	return countMap;
}

/**
 * Gets the message count for a specific conversation from the count map
 * @param conversationId - The ID of the conversation
 * @param countMap - Map of conversation IDs to message counts
 * @returns The message count, or 0 if not found
 */
export function getMessageCount(conversationId: string, countMap: Map<string, number>): number {
	return countMap.get(conversationId) ?? 0;
}

export interface ConversationTreeItem {
	conversation: DatabaseConversation;
	depth: number;
}

// Pinned conversations first, then by lastModified descending
const comparePinnedThenRecent = (a: DatabaseConversation, b: DatabaseConversation) => {
	if (a.pinned && !b.pinned) return -1;

	if (!a.pinned && b.pinned) return 1;

	return b.lastModified - a.lastModified;
};

/**
 * Builds a flat tree of conversations with depth levels for nested forks.
 * Accepts a pre-filtered list so search filtering stays in the component.
 *
 * Output order matches the sidebar render exactly: pinned first, then
 * unpinned by lastModified desc, with forks interleaved under their parents.
 * Range-select / marquee in the sidebar rely on this alignment.
 */
export function buildConversationTree(convs: DatabaseConversation[]): ConversationTreeItem[] {
	const childrenByParent = new Map<string, DatabaseConversation[]>();
	const forkIds = new Set<string>();

	for (const conv of convs) {
		if (conv.forkedFromConversationId) {
			forkIds.add(conv.id);

			const siblings = childrenByParent.get(conv.forkedFromConversationId) || [];

			siblings.push(conv);
			childrenByParent.set(conv.forkedFromConversationId, siblings);
		}
	}

	const result: ConversationTreeItem[] = [];
	const visited = new Set<string>();

	function walk(conv: DatabaseConversation, depth: number) {
		visited.add(conv.id);
		result.push({ conversation: conv, depth });

		const children = childrenByParent.get(conv.id);

		if (children) {
			children.sort(comparePinnedThenRecent);

			for (const child of children) {
				walk(child, depth + 1);
			}
		}
	}

	const roots = convs.filter((c) => !forkIds.has(c.id)).sort(comparePinnedThenRecent);

	for (const root of roots) {
		walk(root, 0);
	}

	for (const conv of convs) {
		if (!visited.has(conv.id)) {
			walk(conv, 1);
		}
	}

	return result;
}
