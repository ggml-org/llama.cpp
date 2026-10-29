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

	it('returns groups newest first', async () => {
		await DatabaseService.createGroup({ ...makeGroup('a'), updatedAt: 1 });
		await DatabaseService.createGroup({ ...makeGroup('b'), updatedAt: 2 });

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
});
