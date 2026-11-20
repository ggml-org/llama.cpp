/**
 * ConversationSearchService - the search dialog's message search
 *
 * Scans the messages of each conversation, newest conversation first, and stops at the first
 * matching message of a conversation and once enough conversations matched, so a common word
 * does not read the whole database. What a message offers to the search lives in utils/conversation-search.
 */

import { DatabaseService } from './database.service';
import type { DatabaseConversation } from '#lib/types/database.js';
import {
	type ConversationSearchSnippet,
	findMessageMatch,
	matchSnippet
} from '#lib/utils/conversation-search.ts';

export interface ConversationContentMatch {
	conversation: DatabaseConversation;
	messageId: string;
	snippet: ConversationSearchSnippet;
}

export interface ConversationContentSearchOptions {
	/** Matches to find before the scan stops */
	limit: number;
	/** Checked between conversations; false stops the scan, its result is then dropped */
	isCurrent?: () => boolean;
}

export class ConversationSearchService {
	/**
	 * Conversations whose messages contain `query`, case-insensitively, in the order given.
	 * Pass `conversations` newest first: the scan stops after `limit` matches.
	 */
	static async searchContent(
		conversations: DatabaseConversation[],
		query: string,
		{ isCurrent = () => true, limit }: ConversationContentSearchOptions
	): Promise<ConversationContentMatch[]> {
		const needle = query.trim().toLowerCase();

		if (!needle || limit <= 0) return [];

		return DatabaseService.readOnly(async ({ messages }) => {
			const found: ConversationContentMatch[] = [];

			for (const conversation of conversations) {
				if (found.length >= limit || !isCurrent()) break;

				const message = await messages
					.where('convId')
					.equals(conversation.id)
					.filter((row) => findMessageMatch(row, needle) !== null)
					.first();
				// only the one message that matched is read twice
				const match = message && findMessageMatch(message, needle);

				if (!message || !match) continue;

				found.push({
					conversation,
					messageId: message.id,
					snippet: matchSnippet(match.text, match.at, needle.length)
				});
			}

			return found;
		});
	}
}
