/**
 * Utility functions for conversation data manipulation
 */
import type { ExportedConversations, DatabaseMessage } from '$lib/types';

// Conversation filename constants

// Length of the trimmed conversation ID in the filename
export const CONV_ID_TRIM_LENGTH = 8;
// Maximum length of the sanitized conversation name snippet
export const CONV_NAME_SUFFIX_MAX_LENGTH = 20;
// Characters to keep in the ISO timestamp. 19 keeps 2026-01-01T00:00:00
export const ISO_TIMESTAMP_SLICE_LENGTH = 19;
// Regex
export const NON_ALPHANUMERIC_REGEX = /[^a-z0-9]/gi;
export const MULTIPLE_UNDERSCORE_REGEX = /_+/g;

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

/**
 * Generates a sanitized filename for a conversation export
 * @param conversation - The conversation metadata
 * @param msgs - Optional array of messages belonging to the conversation
 * @returns The generated filename string
 */
export function generateConversationFilename(
	conversation: { id?: string; name?: string },
	msgs?: DatabaseMessage[]
): string {
	const conversationName = (conversation.name ?? '').trim().toLowerCase();

	const sanitizedName = conversationName
		.replace(NON_ALPHANUMERIC_REGEX, '_')
		.replace(MULTIPLE_UNDERSCORE_REGEX, '_')
		.substring(0, CONV_NAME_SUFFIX_MAX_LENGTH);

	// If we have messages, use the timestamp of the newest message
	const referenceDate = msgs?.length
		? new Date(Math.max(...msgs.map((m) => m.timestamp)))
		: new Date();

	const iso = referenceDate.toISOString().slice(0, ISO_TIMESTAMP_SLICE_LENGTH);
	const formattedDate = iso.replace('T', '_').replaceAll(':', '-');
	const trimmedConvId = conversation.id?.slice(0, CONV_ID_TRIM_LENGTH) ?? '';
	return `${formattedDate}_conv_${trimmedConvId}_${sanitizedName}.json`;
}

/**
 * Triggers a browser download of the provided exported conversation data
 * @param data - The exported conversation payload (either a single conversation or array of them)
 * @param filename - Filename; if omitted, a deterministic name is generated
 */
export function downloadConversationFile(data: ExportedConversations, filename?: string): void {
	// Choose the first conversation or message
	const conversation = 'conv' in data ? data.conv : Array.isArray(data) ? data[0]?.conv : undefined;
	const msgs =
		'messages' in data ? data.messages : Array.isArray(data) ? data[0]?.messages : undefined;

	if (!conversation) {
		console.error('Invalid data: missing conversation');
		return;
	}

	const downloadFilename = filename ?? generateConversationFilename(conversation, msgs);

	const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
	const url = URL.createObjectURL(blob);
	const a = document.createElement('a');
	a.href = url;
	a.download = downloadFilename;
	document.body.appendChild(a);
	a.click();
	document.body.removeChild(a);
	URL.revokeObjectURL(url);
}
