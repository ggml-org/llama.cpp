// Field updates to the active conversation must keep the object identity
// stable: effects that track the identity ( the chat screen's sibling-info
// refresh ) refire on every identity change, which used to trigger a full
// message refetch on every send and tool result.

import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/services/database.service', () => ({
	DatabaseService: {
		getConversation: vi.fn(),
		getConversationMessages: vi.fn(),
		updateConversation: vi.fn(),
		updateCurrentNode: vi.fn()
	}
}));

import { DatabaseService } from '$lib/services/database.service';
import { conversationsStore } from '$lib/stores/conversations/index.svelte';
import type { DatabaseConversation, DatabaseMessage } from '$lib/types/database';

const getConversationMock = vi.mocked(DatabaseService.getConversation);
const getMessagesMock = vi.mocked(DatabaseService.getConversationMessages);
const updateCurrentNodeMock = vi.mocked(DatabaseService.updateCurrentNode);

function makeConversation(overrides: Partial<DatabaseConversation> = {}): DatabaseConversation {
	return {
		currNode: 'node-1',
		id: 'conv-1',
		lastModified: 1000,
		name: 'conversation',
		...overrides
	};
}

async function loadActive(conversation: DatabaseConversation, messages: DatabaseMessage[]) {
	getConversationMock.mockResolvedValue(conversation);
	getMessagesMock.mockResolvedValue(messages);

	expect(await conversationsStore.loadConversation(conversation.id)).toBe(true);
}

beforeEach(() => {
	getConversationMock.mockReset();
	getMessagesMock.mockReset();
	updateCurrentNodeMock.mockReset();
	updateCurrentNodeMock.mockResolvedValue(undefined);
	vi.mocked(DatabaseService.updateConversation).mockReset();
	vi.mocked(DatabaseService.updateConversation).mockResolvedValue(undefined);
});

describe('active conversation identity', () => {
	it('hands the load read off exactly once', async () => {
		await loadActive(makeConversation(), []);

		expect(conversationsStore.consumeLastLoadedMessages('conv-1')).toEqual([]);
		// a second consume is a miss: branch actions must fall back to a refetch
		expect(conversationsStore.consumeLastLoadedMessages('conv-1')).toBeNull();
	});

	it('writes currNode in place on updateCurrentNode', async () => {
		await loadActive(makeConversation(), []);
		const before = conversationsStore.activeConversation;

		await conversationsStore.updateCurrentNode('node-2');

		expect(conversationsStore.activeConversation).toBe(before);
		expect(conversationsStore.activeConversation?.currNode).toBe('node-2');
	});

	it('writes renamed and pinned fields in place on applyConversationUpdate', async () => {
		await loadActive(makeConversation(), []);
		const before = conversationsStore.activeConversation;

		conversationsStore.applyConversationUpdate('conv-1', { name: 'renamed', pinned: true });

		expect(conversationsStore.activeConversation).toBe(before);
		expect(conversationsStore.activeConversation?.name).toBe('renamed');
		expect(conversationsStore.activeConversation?.pinned).toBe(true);
	});

	it('writes lastModified in place on updateConversationTimestamp', async () => {
		await loadActive(makeConversation(), []);
		const before = conversationsStore.activeConversation;

		conversationsStore.updateConversationTimestamp('conv-1');

		expect(conversationsStore.activeConversation).toBe(before);
		expect(conversationsStore.activeConversation?.lastModified).toBeGreaterThan(1000);
	});
});
