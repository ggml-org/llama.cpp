// A message queued (steering or pending) while a conversation is busy belongs
// to that conversation: when its turn ends while another chat is open, it must
// stay queued there instead of being sent into the chat the user is viewing.

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
import { chatStore } from '$lib/stores/chat/index.svelte';
import { conversationsStore } from '$lib/stores/conversations/index.svelte';

type QueuedMessage = { content: string };
type QueueAccess = { sendQueuedMessage(convId: string, msg: QueuedMessage): Promise<void> };

const sendQueuedMessage = (convId: string, msg: QueuedMessage) =>
	(chatStore as unknown as QueueAccess).sendQueuedMessage(convId, msg);

async function openConversation(id: string) {
	vi.mocked(DatabaseService.getConversation).mockResolvedValue({
		currNode: null,
		id,
		lastModified: 1000,
		name: id
	});
	vi.mocked(DatabaseService.getConversationMessages).mockResolvedValue([]);

	expect(await conversationsStore.loadConversation(id)).toBe(true);
}

beforeEach(() => {
	vi.restoreAllMocks();
	chatStore.clearPendingMessage('conv-a');
	chatStore.clearPendingMessage('conv-b');
});

describe('queued message isolation', () => {
	it('keeps the message on its own conversation when another one is active', async () => {
		const send = vi.spyOn(chatStore, 'sendMessage').mockResolvedValue();

		await openConversation('conv-b');
		await sendQueuedMessage('conv-a', { content: 'steer a' });

		expect(send).not.toHaveBeenCalled();
		expect(chatStore.getPendingMessageContent('conv-a')).toBe('steer a');
		expect(chatStore.getPendingMessageContent('conv-b')).toBeNull();
	});

	it('sends right away when its conversation is still active', async () => {
		const send = vi.spyOn(chatStore, 'sendMessage').mockResolvedValue();

		await openConversation('conv-a');
		await sendQueuedMessage('conv-a', { content: 'steer a' });

		expect(send).toHaveBeenCalledWith('steer a', undefined);
		expect(chatStore.hasPendingMessage('conv-a')).toBe(false);
	});

	it('flushes the left-behind message only once its conversation is active again', async () => {
		const send = vi.spyOn(chatStore, 'sendMessage').mockResolvedValue();

		chatStore.injectPendingMessage('conv-a', 'steer a');
		await openConversation('conv-b');
		await chatStore.flushPendingMessage('conv-a');

		expect(send).not.toHaveBeenCalled();

		await openConversation('conv-a');
		await chatStore.flushPendingMessage('conv-a');

		expect(send).toHaveBeenCalledWith('steer a', undefined);
		expect(chatStore.hasPendingMessage('conv-a')).toBe(false);
	});
});
