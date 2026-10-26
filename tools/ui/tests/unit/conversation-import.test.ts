import { NEWLINE } from '$lib/constants';
import { MessageRole, MessageType } from '$lib/enums';
import { ConversationTransferService } from '$lib/services/conversation-transfer.service';
import type { ExportedConversation } from '$lib/types/database';
import { strToU8, zipSync } from 'fflate';
import { describe, expect, it } from 'vitest';

function makeSession(id: string): ExportedConversation {
	return {
		conv: { currNode: `${id}-msg`, id, lastModified: 0, name: `Chat ${id}` },
		messages: [
			{
				children: [],
				content: `hello from ${id}`,
				convId: id,
				id: `${id}-msg`,
				parent: null,
				role: MessageRole.USER,
				timestamp: 0,
				type: MessageType.TEXT
			}
		]
	} as unknown as ExportedConversation;
}

/**
 * `parseImportFile` detects the format from the file contents. iOS has no UTI
 * for `.jsonl`, so the picker cannot filter on it and the filename carries no
 * guarantee: a JSONL export must import under any name.
 */
describe('ConversationTransferService.parseImportFile', () => {
	it('imports a JSONL export whose name has no meaningful extension', async () => {
		const jsonl = ConversationTransferService.serializeSessionToJsonl(makeSession('a'));
		const sessions = await ConversationTransferService.parseImportFile(new File([jsonl], 'export'));

		expect(sessions).toHaveLength(1);
		expect(sessions[0].conv.id).toBe('a');
		expect(sessions[0].messages[0].content).toBe('hello from a');
	});

	it('imports several sessions from one JSONL file', async () => {
		const jsonl = [makeSession('a'), makeSession('b')]
			.map((session) => ConversationTransferService.serializeSessionToJsonl(session))
			.join(NEWLINE);
		const sessions = await ConversationTransferService.parseImportFile(
			new File([jsonl], 'export.txt')
		);

		expect(sessions.map((session) => session.conv.id)).toEqual(['a', 'b']);
	});

	it('imports a ZIP archive whose name has no meaningful extension', async () => {
		const zipped = zipSync({
			'a.jsonl': strToU8(ConversationTransferService.serializeSessionToJsonl(makeSession('a'))),
			'b.jsonl': strToU8(ConversationTransferService.serializeSessionToJsonl(makeSession('b'))),
			'notes.txt': strToU8('ignored')
		});
		const sessions = await ConversationTransferService.parseImportFile(
			new File([zipped], 'archive')
		);

		expect(sessions.map((session) => session.conv.id).sort()).toEqual(['a', 'b']);
	});

	it('imports the legacy JSON array format', async () => {
		const json = JSON.stringify([makeSession('a')], null, 2);
		const sessions = await ConversationTransferService.parseImportFile(
			new File([json], 'export.jsonl')
		);

		expect(sessions).toHaveLength(1);
		expect(sessions[0].conv.id).toBe('a');
	});

	it('imports the legacy JSON single object format', async () => {
		const json = JSON.stringify(makeSession('a'));
		const sessions = await ConversationTransferService.parseImportFile(new File([json], 'export'));

		expect(sessions).toHaveLength(1);
		expect(sessions[0].conv.id).toBe('a');
	});

	it('rejects a file that holds neither format', async () => {
		await expect(
			ConversationTransferService.parseImportFile(new File(['not an export'], 'export.jsonl'))
		).rejects.toThrow();
	});
});
