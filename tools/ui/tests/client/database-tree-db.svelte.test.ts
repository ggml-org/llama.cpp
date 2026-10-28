import { MessageRole, MessageType } from '$lib/enums';
import { DatabaseService } from '$lib/services/database.service';
import { afterEach, describe, expect, it } from 'vitest';

afterEach(async () => {
	const conversations = await DatabaseService.getAllConversations();

	await DatabaseService.bulkDeleteConversations(conversations.map((conv) => conv.id));
});

async function createChat(name: string, forkedFromConversationId?: string) {
	const conv = await DatabaseService.createConversation(name, { forkedFromConversationId });
	const rootId = await DatabaseService.createRootMessage(conv.id);

	return { conv, rootId };
}

function userMessage(convId: string, content: string, timestamp: number) {
	return {
		children: [],
		content,
		convId,
		parent: null,
		role: MessageRole.USER,
		timestamp: Date.now() + timestamp,
		type: MessageType.TEXT
	};
}

/**
 * Tree rules applied by DatabaseService on top of the IndexedDB backend.
 */
describe('DatabaseService tree operations', () => {
	it('links a new branch to its parent and moves the current node', async () => {
		const { conv, rootId } = await createChat('a');
		const first = await DatabaseService.createMessageBranch(userMessage(conv.id, 'one', 1), rootId);
		const second = await DatabaseService.createMessageBranch(
			userMessage(conv.id, 'two', 2),
			rootId
		);
		const messages = await DatabaseService.getConversationMessages(conv.id);
		const root = messages.find((m) => m.id === rootId);

		expect(root?.children).toEqual([first.id, second.id]);
		expect((await DatabaseService.getConversation(conv.id))?.currNode).toBe(second.id);
	});

	it('cascades a message delete and unlinks it from its parent', async () => {
		const { conv, rootId } = await createChat('a');
		const user = await DatabaseService.createMessageBranch(userMessage(conv.id, 'u', 1), rootId);
		const reply = await DatabaseService.createMessageBranch(userMessage(conv.id, 'r', 2), user.id);
		const deleted = await DatabaseService.deleteMessageCascading(conv.id, user.id);
		const messages = await DatabaseService.getConversationMessages(conv.id);

		expect(deleted.sort()).toEqual([user.id, reply.id].sort());
		expect(messages.map((m) => m.id)).toEqual([rootId]);
		expect(messages[0].children).toEqual([]);
	});

	it('forks the path to a message into a new conversation', async () => {
		const { conv, rootId } = await createChat('a');
		const user = await DatabaseService.createMessageBranch(userMessage(conv.id, 'u', 1), rootId);

		await DatabaseService.createMessageBranch(userMessage(conv.id, 'later', 2), user.id);

		const fork = await DatabaseService.forkConversation(conv.id, user.id, {
			includeAttachments: true,
			name: 'fork'
		});
		const forkMessages = await DatabaseService.getConversationMessages(fork.id);

		expect(fork.forkedFromConversationId).toBe(conv.id);
		expect(forkMessages.map((m) => m.content)).toEqual(['', 'u']);
		expect(fork.currNode).toBe(forkMessages[1].id);
	});

	it('reparents forks of a deleted conversation to its parent', async () => {
		const { conv: a } = await createChat('a');
		const { conv: b } = await createChat('b', a.id);
		const { conv: c } = await createChat('c', b.id);

		await DatabaseService.deleteConversation(b.id);

		expect(await DatabaseService.getConversation(b.id)).toBeUndefined();
		expect(await DatabaseService.getConversationMessages(b.id)).toEqual([]);
		expect((await DatabaseService.getConversation(c.id))?.forkedFromConversationId).toBe(a.id);
	});

	it('deletes forks along with the conversation when asked to', async () => {
		const { conv: a } = await createChat('a');
		const { conv: b } = await createChat('b', a.id);
		const { conv: c } = await createChat('c', b.id);

		await DatabaseService.deleteConversation(a.id, { deleteWithForks: true });

		expect(await DatabaseService.getAllConversations()).toEqual([]);
		expect(await DatabaseService.getConversationMessages(c.id)).toEqual([]);
	});

	it('reparents past every conversation deleted in the same batch', async () => {
		const { conv: a } = await createChat('a');
		const { conv: b } = await createChat('b', a.id);
		const { conv: c } = await createChat('c', b.id);
		const { conv: d } = await createChat('d', c.id);

		await DatabaseService.bulkDeleteConversations([b.id, c.id]);

		const remaining = await DatabaseService.getAllConversations();

		expect(remaining.map((conv) => conv.id).sort()).toEqual([a.id, d.id].sort());
		expect((await DatabaseService.getConversation(d.id))?.forkedFromConversationId).toBe(a.id);
	});
});
