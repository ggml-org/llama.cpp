// IndexedDB side of search_chats: loading, the current-chat filter and the readonly transaction.
// Matching and formatting are covered by tests/unit/search-chats.test.ts.

import { MessageRole, MessageType } from '$lib/enums';
import { DatabaseService } from '$lib/services/database.service';
import { SearchChatsService } from '$lib/services/search-chats.service';
import type { ExportedConversation } from '$lib/types/database';
import { formatLocalIso } from '$lib/utils/formatters';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

function makeConversation(
	id: string,
	name: string,
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
		conv: { currNode: messages.at(-1)!.id, id, lastModified, name },
		messages
	} as unknown as ExportedConversation;
}

beforeEach(async () => {
	await DatabaseService.importConversations([
		makeConversation(
			'paris',
			"let's think about paris",
			[
				[MessageRole.USER, 'plan a weekend in Paris'],
				[MessageRole.ASSISTANT, 'Day 1: Louvre'],
				[MessageRole.USER, 'how much is a metro ticket?'],
				[MessageRole.ASSISTANT, 'About 2 euros.']
			],
			1000
		),
		makeConversation(
			'search',
			'list my chats',
			[
				[MessageRole.USER, 'list my chats'],
				[MessageRole.TOOL, `=> [{"name":"let's think about paris"}]`],
				[MessageRole.ASSISTANT, 'You have two chats.']
			],
			2000
		)
	]);
});

afterEach(async () => {
	const conversations = await DatabaseService.getAllConversations();

	await DatabaseService.bulkDeleteConversations(conversations.map((conv) => conv.id));
});

describe('search_chats service', () => {
	it('searches every stored message and hides the current chat', async () => {
		const reply = await SearchChatsService.executeTool(
			{ by_matches: ['%paris%'], limit_messages: 1, message_selection: 'latest' },
			'search'
		);

		expect(reply.isError).toBe(false);
		expect(reply.content).toContain('1 chat(s) matching ["%paris%"]');
		expect(reply.content).toContain('#4 assistant');
		expect(reply.content).toContain('About 2 euros.');
	});

	it('loads chats by id and reports missing ones', async () => {
		const reply = await SearchChatsService.executeTool(
			{ by_conv_ids: ['paris', 'missing'], limit_messages: 1, message_selection: 'earliest' },
			'search'
		);

		expect(reply.content).toContain('not found: missing');
		expect(reply.content).toContain('plan a weekend in Paris');
	});

	it('finds chats with messages in a time range through the timestamp index', async () => {
		const reply = await SearchChatsService.executeTool(
			{ from_timestamp: formatLocalIso(new Date(2000)), limit_messages: 1 },
			'other'
		);

		expect(reply.content).toContain('1 chat(s) from');
		expect(reply.content).toContain('conv_id: search');
		expect(reply.content).toContain('#1 user');
	});

	it('returns parameter errors to the model', async () => {
		const reply = await SearchChatsService.executeTool({ order_timestamp: 'esc' });

		expect(reply.isError).toBe(true);
		expect(reply.content).toBe('Error: order_timestamp must be one of: asc, desc');
	});

	it('reads through a transaction that refuses writes', async () => {
		await expect(
			DatabaseService.readOnly(({ messages }) => messages.delete('paris-m0'))
		).rejects.toThrow();
		expect(await DatabaseService.getConversationMessages('paris')).toHaveLength(4);
	});
});
