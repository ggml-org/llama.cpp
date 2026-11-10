import { SEARCH_CHATS_OUTPUT_MAX_CHARS, SEARCH_CHATS_PATTERN_MAX_CHARS } from '$lib/constants';
import { MessageRole, MessageType } from '$lib/enums';
import type { DatabaseConversation, DatabaseMessage } from '$lib/types/database';
import { localIsoString } from '$lib/utils/get-datetime';
import { parseSearchChatsParams, searchChats, SearchChatsError } from '$lib/utils/search-chats';
import { describe, expect, it } from 'vitest';

type Turn = [role: 'user' | 'assistant' | 'tool', content: string];

const conversations: DatabaseConversation[] = [];
const messages: DatabaseMessage[] = [];
const at = (ms: number) => localIsoString(new Date(ms));

/** Add a linear chat whose messages are timestamped lastModified + i */
function addChat(id: string, name: string, lastModified: number, turns: Turn[]): string[] {
	const root = `${id}-root`;
	const ids = turns.map((_, i) => `${id}-${i + 1}`);

	messages.push({
		children: ids.slice(0, 1),
		content: '',
		convId: id,
		id: root,
		parent: null,
		role: MessageRole.SYSTEM,
		timestamp: lastModified - 100,
		type: MessageType.ROOT
	} as DatabaseMessage);

	turns.forEach(([role, content], i) => {
		messages.push({
			children: ids.slice(i + 1, i + 2),
			content,
			convId: id,
			id: ids[i],
			parent: i === 0 ? root : ids[i - 1],
			role: role as MessageRole,
			timestamp: lastModified + i,
			type: MessageType.TEXT
		} as DatabaseMessage);
	});

	conversations.push({ currNode: ids.at(-1)!, id, lastModified, name } as DatabaseConversation);

	return ids;
}

const parisIds = addChat('paris', "let's think about paris", 3000, [
	['user', 'plan a weekend in Paris'],
	['assistant', 'Day 1: Louvre. Day 2: Montmartre.'],
	['user', 'is the eiffel tower worth it?'],
	['assistant', 'Yes, book tickets early.'],
	['user', 'how much is a metro ticket?'],
	['assistant', 'About 2 euros.']
]);

// edited-away branch off the first assistant reply, never visible
messages.push({
	children: [],
	content: 'old edit about versailles',
	convId: 'paris',
	id: 'paris-edit',
	parent: parisIds[1],
	role: MessageRole.USER,
	timestamp: 2955,
	type: MessageType.TEXT
} as DatabaseMessage);

// an earlier search session: its tool output names other chats
addChat('search', 'list my chats', 4000, [
	['user', 'list my chats please'],
	['tool', `=> [{"name":"let's think about paris"},{"name":"what time is it in bulgaria"}]`],
	['assistant', 'You have a few chats.']
]);

addChat('bulgaria', 'what time is it in bulgaria', 2000, [
	['user', 'what time is it in bulgaria'],
	['assistant', 'It is 14:00 in Sofia.']
]);

addChat('current', 'this chat', 5000, [['user', 'find the chat where I talk about paris']]);

const loads: (string[] | null)[] = [];

async function loadMessages(ids: string[] | null) {
	loads.push(ids);

	return ids ? messages.filter((m) => ids.includes(m.convId)) : messages;
}

async function search(args: Record<string, unknown>, current = 'current') {
	const params = parseSearchChatsParams(args);
	const candidates = params.byIds
		? conversations.filter((c) => params.byIds!.includes(c.id))
		: conversations;

	return searchChats(candidates, params, loadMessages, current);
}

describe('search_chats: finding chats', () => {
	it('answers "last question in the paris chat" in one call', async () => {
		const out = await search({
			by_matches: ['%paris%'],
			limit_messages: 4,
			message_selection: 'latest'
		});

		expect(out).toContain('1 chat(s) matching ["%paris%"] (newest first, showing 1-1 of 1)');
		expect(out).toContain(`## Chat 1: "let's think about paris"`);
		expect(out).toContain(`#5 user | ${at(3004)}: how much is a metro ticket?`);
		expect(out).toContain('#6 assistant');
		expect(out).not.toContain('#2 assistant');
	});

	it('ignores tool output and the current chat', async () => {
		const out = await search({ by_matches: ['bulgaria'] });

		// the search session mentions bulgaria in tool output only
		expect(out).toContain('1 chat(s) matching');
		expect(out).toContain('id: bulgaria');
		expect(out).not.toContain('id: search');
	});

	it('shows the matching messages by default', async () => {
		const out = await search({ by_matches: ['%eiffel%', 'metro'] });

		expect(out).toContain('matched in: 2 message(s)');
		expect(out).toContain('#3 user');
		expect(out).toContain('#5 user');
		expect(out).not.toContain('#1 user');
	});

	it('reports a title-only match and falls back to the latest messages', async () => {
		const out = await search({ by_matches: ['%think about%'], limit_messages: 1 });

		expect(out).toContain('matched in: title only, showing the latest messages');
		expect(out).toContain('#6 assistant');
	});

	it('does not search edited-away branches', async () => {
		expect(await search({ by_matches: ['versailles'] })).toContain('0 chat(s) matching');
	});

	it('supports _ wildcards and plain words as "contains"', async () => {
		expect(await search({ by_matches: ['m_tro'] })).toContain('id: paris');
		expect(await search({ by_matches: 'SOFIA' })).toContain('id: bulgaria');
	});

	it('moves the excerpt of a long message to the match', async () => {
		const long = `  ${'x'.repeat(2000)} the secret word is paris ${'y'.repeat(2000)}`;

		addChat('long', 'long chat', 1000, [['user', long]]);

		const out = await search({
			by_ids: ['long'],
			by_matches: ['secret word'],
			max_message_chars: 100
		});

		expect(out).toMatch(/: \.\.\.x+ the secret word is paris y+\.\.\.\[\+\d+ chars\]/);

		conversations.pop();
		messages.splice(-2, 2);
	});
});

describe('search_chats: listing and reading', () => {
	it('lists chats as one line each without reading messages', async () => {
		loads.length = 0;

		const out = await search({ limit_chats: 999, limit_messages: 0 });

		expect(out.split('\n')).toEqual([
			'3 chat(s) (newest first, showing 1-3 of 3)',
			'',
			`- "list my chats" | id: search | updated ${at(4000)}`,
			`- "let's think about paris" | id: paris | updated ${at(3000)}`,
			`- "what time is it in bulgaria" | id: bulgaria | updated ${at(2000)}`
		]);
		expect(loads).toEqual([]);
	});

	it('reads messages only for the chats on the page', async () => {
		loads.length = 0;
		await search({ limit_chats: 2 });
		expect(loads).toEqual([['search', 'paris']]);
	});

	it('pages and orders oldest first', async () => {
		const out = await search({
			limit_chats: 1,
			limit_messages: 0,
			offset_chats: 1,
			order_timestamp: 'asc'
		});

		expect(out).toContain('(oldest first, showing 2-2 of 3)');
		expect(out).toContain('id: paris');
	});

	it('reads a chat by id from the start', async () => {
		const out = await search({
			by_ids: ['paris'],
			limit_messages: 2,
			message_selection: 'earliest'
		});

		expect(out).toContain(`#1 user | ${at(3000)}: plan a weekend in Paris`);
		expect(out).toContain('#2 assistant');
		expect(out).not.toContain('#3 user');
	});

	it('reports unknown ids and hides the current chat unless asked', async () => {
		const hidden = await search({ by_ids: ['current', 'nope'] });

		expect(hidden).toContain('not found: nope');
		expect(hidden).toContain('the current chat is hidden');
		expect(hidden).toContain('0 chat(s) by id');

		const shown = await search({ by_ids: ['current'], include_current: true });

		expect(shown).toContain(`#1 user | ${at(5000)}: find the chat where I talk about paris`);
	});

	it('stays within the output budget with a paging hint', async () => {
		for (let i = 0; i < 20; i++) {
			addChat(`big${i}`, `big ${i}`, 100 + i, [['user', 'z'.repeat(1000)]]);
		}

		const out = await search({ by_matches: ['zzz'], limit_chats: 20, max_message_chars: 1000 });

		expect(out).toMatch(/\[\d+ more chat\(s\) on this page not shown: use offset_chats=\d+/);
		expect(out.length).toBeLessThanOrEqual(SEARCH_CHATS_OUTPUT_MAX_CHARS);

		conversations.splice(-20, 20);
		messages.splice(-40, 40);
	});
});

describe('search_chats: parameters', () => {
	it.each([
		[{ message_selection: 'matching' }, 'needs by_matches'],
		[{ limit_chats: 0 }, 'limit_chats must be an integer from 1 to 1000'],
		[{ limit_messages: 1.5 }, 'limit_messages must be an integer'],
		[{ order_timestamp: 'esc' }, 'order_timestamp must be one of: asc, desc'],
		[{ by_matches: [1] }, 'by_matches must be an array of non-empty strings'],
		[
			{ by_matches: ['x'.repeat(SEARCH_CHATS_PATTERN_MAX_CHARS + 1)] },
			`by_matches entries must be at most ${SEARCH_CHATS_PATTERN_MAX_CHARS} characters`
		],
		[{ message_selection: 'all' }, 'message_selection must be one of']
	])('rejects %j', (args, message) => {
		expect(() => parseSearchChatsParams(args)).toThrow(SearchChatsError);
		expect(() => parseSearchChatsParams(args)).toThrow(message);
	});

	it('accepts numbers sent as strings', () => {
		expect(parseSearchChatsParams({ limit_chats: '3' }).limitChats).toBe(3);
	});
});
