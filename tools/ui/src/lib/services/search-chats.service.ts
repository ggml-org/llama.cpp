/**
 * SearchChatsService - executor for the search_chats browser tool
 *
 * Reads conversations and messages in readonly transactions; matching and formatting live in utils/search-chats.
 */

import { DatabaseService } from './database.service';
import { IDXDB_TABLES } from '$lib/constants';
import type { ToolExecutionResult } from '$lib/types';
import type { DatabaseConversation, DatabaseMessage } from '$lib/types/database';
import { parseSearchChatsParams, searchChats, SearchChatsError } from '$lib/utils/search-chats';

function loadConversations(ids?: string[]): Promise<DatabaseConversation[]> {
	return DatabaseService.readOnly(async (table) => {
		const conversations = table(IDXDB_TABLES.conversations);

		if (!ids) return (await conversations.toArray()) as DatabaseConversation[];

		const found = (await conversations.bulkGet(ids)) as (DatabaseConversation | undefined)[];

		return found.filter((c): c is DatabaseConversation => !!c);
	});
}

function loadMessages(convIds: string[] | null): Promise<DatabaseMessage[]> {
	return DatabaseService.readOnly(async (table) => {
		const messages = table(IDXDB_TABLES.messages);
		const rows = convIds
			? await messages.where('convId').anyOf(convIds).toArray()
			: await messages.toArray();

		return rows as DatabaseMessage[];
	});
}

export class SearchChatsService {
	/** `currentConversationId` is hidden from results unless the model sets include_current */
	static async executeTool(
		args: Record<string, unknown>,
		currentConversationId?: string
	): Promise<ToolExecutionResult> {
		try {
			const params = parseSearchChatsParams(args);
			const conversations = await loadConversations(params.byIds);
			const content = await searchChats(conversations, params, loadMessages, currentConversationId);

			return { content, isError: false };
		} catch (err) {
			if (err instanceof SearchChatsError)
				return { content: `Error: ${err.message}`, isError: true };

			throw err;
		}
	}
}
