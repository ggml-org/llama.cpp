// IndexedDB side of the search dialog's message search: order, the stop at `limit` and staleness.
// Matching and snippets are covered by tests/unit/conversation-search.test.ts.

import { MessageRole, MessageType } from '#lib/enums/index.ts';
import { ConversationSearchService } from '#lib/services/conversation-search.service.ts';
import { DatabaseService } from '#lib/services/database.service.ts';
import type { DatabaseConversation, ExportedConversation } from '#lib/types/database.js';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

function makeConversation(
	id: string,
	turns: [MessageRole, string][],
	lastModified: number
): ExportedConversation {
	const messages = turns.map(([role, content], i) => ({
		children: i + 1 < turns.length ? [`${id}-m${i + 1}`] : [],
		content,
		convId: id,
		id: `${id}-m${i}`,
		parent: i === 0 ? null : `${id}-m${i - 1}`,
		role,
		timestamp: lastModified + i,
		type: MessageType.TEXT
	}));

	return {
		conv: { currNode: messages.at(-1)!.id, id, lastModified, name: id },
		messages
	} as unknown as ExportedConversation;
}

let newestFirst: DatabaseConversation[] = [];

beforeEach(async () => {
	await DatabaseService.importConversations([
		makeConversation(
			'old',
			[
				[MessageRole.USER, 'what is the weather?'],
				[MessageRole.TOOL, 'Lisbon: sunny'],
				[MessageRole.ASSISTANT, 'It is sunny in Lisbon.']
			],
			1000
		),
		makeConversation('new', [[MessageRole.USER, 'Is it sunny tomorrow?']], 2000),
		makeConversation('other', [[MessageRole.USER, 'recipe for bread']], 3000)
	]);

	newestFirst = (await DatabaseService.getAllConversations()).sort(
		(a, b) => b.lastModified - a.lastModified
	);
});

afterEach(async () => {
	const conversations = await DatabaseService.getAllConversations();

	await DatabaseService.bulkDeleteConversations(conversations.map((conv) => conv.id));
});

describe('ConversationSearchService.searchContent', () => {
	it('returns each matching chat once, in the order given', async () => {
		const found = await ConversationSearchService.searchContent(newestFirst, 'SUNNY', {
			limit: 10
		});

		expect(found.map((match) => match.conversation.id)).toEqual(['new', 'old']);
		expect(found[0].snippet.match).toBe('sunny');
	});

	it('stops at the limit', async () => {
		const found = await ConversationSearchService.searchContent(newestFirst, 'sunny', { limit: 1 });

		expect(found.map((match) => match.conversation.id)).toEqual(['new']);
	});

	it('stops once the search is stale', async () => {
		const found = await ConversationSearchService.searchContent(newestFirst, 'sunny', {
			isCurrent: () => false,
			limit: 10
		});

		expect(found).toEqual([]);
	});

	it('finds tool output', async () => {
		const found = await ConversationSearchService.searchContent(newestFirst, 'lisbon: sun', {
			limit: 10
		});

		expect(found.map((match) => match.conversation.id)).toEqual(['old']);
	});
});
