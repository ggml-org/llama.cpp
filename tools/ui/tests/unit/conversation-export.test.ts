import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/services/database.service', () => ({
	DatabaseService: { getConversationsWithMessages: vi.fn() }
}));

import { MessageRole, MessageType } from '$lib/enums';
import { ConversationTransferService } from '$lib/services/conversation-transfer.service';
import { DatabaseService } from '$lib/services/database.service';
import { conversationsStore } from '$lib/stores/conversations/index.svelte';
import type { DatabaseConversation, DatabaseMessage } from '$lib/types/database';
import { filterByLeafNodeId } from '$lib/utils/branching';

/**
 * Reproduces the exported-conversation bug:
 *
 * A conversation created in the current page session keeps `currNode: ''` in the
 * sidebar list, because that list is only loaded at init while IndexedDB is stamped
 * on every message insert.
 *
 * Exporting from the cached record resulted in no branch pointer, and importing
 * the file showed every branch at once.
 */

const fetchMock = vi.mocked(DatabaseService.getConversationsWithMessages);

beforeEach(() => {
	fetchMock.mockReset();
});

const CONV_ID = 'c1';

function message(
	id: string,
	parent: string | null,
	timestamp: number,
	role: MessageRole,
	type: MessageType = MessageType.TEXT
): DatabaseMessage {
	return {
		children: [],
		content: id,
		convId: CONV_ID,
		id,
		parent,
		role,
		timestamp,
		toolCalls: '',
		type
	} as DatabaseMessage;
}

/** root -> u1 -> a1 -> { u2a -> a2a (older) | u2b -> a2b (newer) } */
function branchedMessages(): DatabaseMessage[] {
	const messages = [
		message('root', null, 10, MessageRole.USER, MessageType.ROOT),
		message('u1', 'root', 20, MessageRole.USER),
		message('a1', 'u1', 30, MessageRole.ASSISTANT),
		message('u2a', 'a1', 40, MessageRole.USER),
		message('a2a', 'u2a', 50, MessageRole.ASSISTANT),
		message('u2b', 'a1', 60, MessageRole.USER),
		message('a2b', 'u2b', 70, MessageRole.ASSISTANT)
	];

	for (const m of messages) {
		m.children = messages.filter((c) => c.parent === m.id).map((c) => c.id);
	}

	return messages;
}

/** A second conversation with a single linear path: root -> u1 -> a1. */
function linearMessages(convId: string): DatabaseMessage[] {
	return [
		{ ...message('root', null, 10, MessageRole.USER, MessageType.ROOT), children: ['u1'], convId },
		{ ...message('u1', 'root', 20, MessageRole.USER), children: ['a1'], convId },
		{ ...message('a1', 'u1', 30, MessageRole.ASSISTANT), convId }
	];
}

function conversation(currNode: string, id: string = CONV_ID): DatabaseConversation {
	return { currNode, id, lastModified: 100, name: `Chat ${id}` };
}

/** Mirrors `conversationsStore.loadConversation` */
function displayedIds(imported: { conv: DatabaseConversation; messages: DatabaseMessage[] }) {
	if (imported.conv.currNode) {
		return filterByLeafNodeId(imported.messages, imported.conv.currNode, false).map((m) => m.id);
	}

	return imported.messages.map((m) => m.id);
}

/** Export then re-import */
function roundTrip(conv: DatabaseConversation) {
	const jsonl = ConversationTransferService.serializeSessionToJsonl({
		conv,
		messages: branchedMessages()
	});
	const [imported] = ConversationTransferService.parseSessionsJsonl(jsonl);

	return { imported, sessionLine: JSON.parse(jsonl.split('\n')[0]) };
}

describe('conversation export source', () => {
	it('reads the database record rather than the stale sidebar list', async () => {
		conversationsStore.conversations = [conversation('')];

		fetchMock.mockResolvedValue(
			new Map([[CONV_ID, { conv: conversation('a2a'), messages: branchedMessages() }]])
		);

		const [exported] = await conversationsStore.getConversationsForExport([CONV_ID]);

		expect(exported.conv.currNode).toBe('a2a');
		expect(conversationsStore.conversations[0].currNode).toBe('');
	});

	it('reads every selected conversation from the database on bulk export', async () => {
		conversationsStore.conversations = [conversation(''), conversation('', 'c2')];
		conversationsStore.activeConversation = conversation('');

		fetchMock.mockResolvedValue(
			new Map([
				['c2', { conv: conversation('a1', 'c2'), messages: linearMessages('c2') }],
				[CONV_ID, { conv: conversation('a2a'), messages: branchedMessages() }]
			])
		);

		const archive = vi
			.spyOn(ConversationTransferService, 'downloadConversationsArchive')
			.mockImplementation(() => {});

		await conversationsStore.bulkExportConversations([CONV_ID, 'c2']);

		expect(fetchMock).toHaveBeenCalledWith([CONV_ID, 'c2']);
		expect(archive).toHaveBeenCalledTimes(1);

		const payload = archive.mock.calls[0][0];

		expect(payload.map((entry) => entry.conv.id)).toEqual([CONV_ID, 'c2']);
		// Each entry carries its own database currNode.
		expect(payload.map((entry) => entry.conv.currNode)).toEqual(['a2a', 'a1']);
		expect(payload[1].messages.map((m: DatabaseMessage) => m.id)).toEqual(['root', 'u1', 'a1']);

		archive.mockRestore();
	});
});

describe('exported conversation branch pointer', () => {
	it('carries the database currNode, so the import restores the current branch', () => {
		// The user regenerated to create a2b, then switched back to the a2a branch,
		// so the stored leaf is NOT the newest message.
		const { imported, sessionLine } = roundTrip(conversation('a2a'));

		expect(sessionLine.currNode).toBe('a2a');
		expect(displayedIds(imported)).toEqual(['u1', 'a1', 'u2a', 'a2a']);
		expect(imported.messages.map((m: DatabaseMessage) => m.id).sort()).toEqual([
			'a1',
			'a2a',
			'a2b',
			'root',
			'u1',
			'u2a',
			'u2b'
		]);
	});

	it('shows every branch on import when the cache entry exported an empty currNode', () => {
		const { imported, sessionLine } = roundTrip(conversation(''));

		expect(sessionLine.currNode).toBe('');
		expect(displayedIds(imported)).toEqual(['root', 'u1', 'a1', 'u2a', 'a2a', 'u2b', 'a2b']);
	});
});
