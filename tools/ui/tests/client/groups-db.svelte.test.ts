import { DatabaseService } from '$lib/services/database.service';
import type { Group } from '$lib/types';
import { afterEach, describe, expect, it } from 'vitest';

function makeGroup(id: string): Group {
	return { createdAt: 0, id, name: `Group ${id}`, updatedAt: 0 };
}

afterEach(async () => {
	const groups = await DatabaseService.getAllGroups();

	await Promise.all(groups.map((group) => DatabaseService.deleteGroup(group.id)));

	const conversations = await DatabaseService.getAllConversations();

	await Promise.all(
		conversations.map((conversation) => DatabaseService.deleteConversation(conversation.id))
	);
});

/**
 * Groups live in their own table; deleting one must leave its chats in place,
 * out of any group.
 */
describe('DatabaseService groups', () => {
	it('round-trips a group', async () => {
		await DatabaseService.createGroup(makeGroup('a'));

		expect(await DatabaseService.getAllGroups()).toEqual([makeGroup('a')]);
	});

	it('applies partial updates without dropping other fields', async () => {
		await DatabaseService.createGroup(makeGroup('a'));

		await DatabaseService.updateGroup('a', { name: 'Renamed', updatedAt: 5 });

		expect(await DatabaseService.getAllGroups()).toEqual([
			{ ...makeGroup('a'), name: 'Renamed', updatedAt: 5 }
		]);
	});

	it('returns groups newest first by creation, whatever their updates', async () => {
		await DatabaseService.createGroup({ ...makeGroup('a'), createdAt: 1, updatedAt: 9 });
		await DatabaseService.createGroup({ ...makeGroup('b'), createdAt: 2, updatedAt: 3 });

		expect((await DatabaseService.getAllGroups()).map((group) => group.id)).toEqual(['b', 'a']);
	});

	it('takes the chats out of a deleted group and keeps them', async () => {
		await DatabaseService.createGroup(makeGroup('a'));
		const inGroup = await DatabaseService.createConversation('in', { groupId: 'a' });
		const elsewhere = await DatabaseService.createConversation('out', { groupId: 'b' });

		await DatabaseService.deleteGroup('a');

		expect(await DatabaseService.getAllGroups()).toEqual([]);
		expect((await DatabaseService.getConversation(inGroup.id))?.groupId).toBeUndefined();
		expect((await DatabaseService.getConversation(elsewhere.id))?.groupId).toBe('b');
	});

	it("sorts imported chats into the file's group, by id, then by name, else a new one", async () => {
		await DatabaseService.createGroup(makeGroup('a'));
		await DatabaseService.createGroup({ ...makeGroup('b'), name: 'Research' });

		const chat = (id: string, groupId?: string) => ({
			conv: { currNode: '', groupId, id, lastModified: 0, name: id },
			messages: []
		});

		await DatabaseService.importConversations([
			{ ...chat('same-id', 'a'), group: { id: 'a', name: 'Other name' } },
			{ ...chat('same-name', 'x'), group: { id: 'x', name: 'Research' } },
			{ ...chat('new-group', 'y'), group: { id: 'y', name: 'Travel' } },
			chat('dangling', 'gone')
		]);

		const groupOf = async (id: string) => (await DatabaseService.getConversation(id))?.groupId;

		expect(await groupOf('same-id')).toBe('a');
		expect(await groupOf('same-name')).toBe('b');
		expect(await groupOf('new-group')).toBe('y');
		expect(await groupOf('dangling')).toBeUndefined();
		expect((await DatabaseService.getAllGroups()).map((group) => group.name)).toContain('Travel');
	});

	it('keeps a forked chat in the group of the chat it was forked from', async () => {
		await DatabaseService.createGroup(makeGroup('a'));
		const source = await DatabaseService.createConversation('source', { groupId: 'a' });
		const rootId = await DatabaseService.createRootMessage(source.id);
		const fork = await DatabaseService.forkConversation(source.id, rootId, {
			includeAttachments: false,
			name: 'fork'
		});

		expect(fork.groupId).toBe('a');
		expect((await DatabaseService.getConversation(fork.id))?.groupId).toBe('a');
	});
});
