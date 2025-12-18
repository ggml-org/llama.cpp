/**
 * Utility functions for conversation data manipulation
 */

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
