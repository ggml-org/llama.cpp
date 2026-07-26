import { afterEach, describe, expect, it } from 'vitest';
import { DatabaseService } from '$lib/services/database.service';
import { MessageRole, MessageType } from '$lib/enums';
import type { ExportedConversation } from '$lib/types/database';

function makeSession(id: string): ExportedConversation {
	return {
		conv: { id, currNode: `${id}-msg`, lastModified: 0, name: `Chat ${id}` },
		messages: [
			{
				id: `${id}-msg`,
				convId: id,
				type: MessageType.TEXT,
				timestamp: 0,
				role: MessageRole.USER,
				content: `hello from ${id}`,
				parent: null,
				children: []
			}
		]
	} as unknown as ExportedConversation;
}

afterEach(async () => {
	const conversations = await DatabaseService.getAllConversations();
	await DatabaseService.bulkDeleteConversations(conversations.map((conv) => conv.id));
});

/**
 * An import leaves a conversation already in the database untouched, so the
 * caller needs to know what was written to report it instead of echoing the
 * selection back at the user.
 */
describe('DatabaseService.importConversations', () => {
	it('reports the conversations it wrote', async () => {
		const { imported, skipped } = await DatabaseService.importConversations([
			makeSession('a'),
			makeSession('b')
		]);

		expect(imported.map((conv) => conv.id)).toEqual(['a', 'b']);
		expect(skipped).toEqual([]);
		expect(await DatabaseService.getConversationMessages('a')).toHaveLength(1);
	});

	it('reports an existing conversation as skipped and leaves it untouched', async () => {
		await DatabaseService.importConversations([makeSession('a')]);
		await DatabaseService.updateConversation('a', { name: 'Renamed locally' });

		const { imported, skipped } = await DatabaseService.importConversations([makeSession('a')]);

		expect(imported).toEqual([]);
		expect(skipped.map((conv) => conv.id)).toEqual(['a']);
		expect((await DatabaseService.getConversation('a'))?.name).toBe('Renamed locally');
	});

	it('imports the new conversations of a partially known selection', async () => {
		await DatabaseService.importConversations([makeSession('a')]);

		const { imported, skipped } = await DatabaseService.importConversations([
			makeSession('a'),
			makeSession('b')
		]);

		expect(imported.map((conv) => conv.id)).toEqual(['b']);
		expect(skipped.map((conv) => conv.id)).toEqual(['a']);
	});
});
