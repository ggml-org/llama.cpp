import { afterEach, describe, expect, it } from 'vitest';
import { DatabaseService } from '$lib/services/database.service';
import { EncryptionService } from '$lib/services/encryption.service';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import { ENCRYPTION_META_LOCALSTORAGE_KEY } from '$lib/constants';
import { MessageRole, MessageType } from '$lib/enums';

afterEach(async () => {
	EncryptionService.disable();
	localStorage.removeItem(ENCRYPTION_META_LOCALSTORAGE_KEY);
	const conversations = await DatabaseService.getAllConversations();
	await DatabaseService.bulkDeleteConversations(conversations.map((conv) => conv.id));
});

describe('encrypted conversation export', () => {
	it('writes the encryption meta into the session header and ciphertext into records', async () => {
		await EncryptionService.setupWithPassphrase('pw');

		const conv = await DatabaseService.createConversation('secret chat');
		const rootId = await DatabaseService.createRootMessage(conv.id);
		await DatabaseService.createMessageBranch(
			{
				convId: conv.id,
				type: MessageType.TEXT,
				timestamp: Date.now(),
				role: MessageRole.USER,
				content: 'secret body',
				parent: null,
				children: []
			} as Omit<DatabaseMessage, 'id'>,
			rootId
		);

		const fetched = await DatabaseService.getConversationsWithMessages([conv.id]);
		const session = [...fetched.values()][0];
		const [encrypted] = await DatabaseService.encryptForExport([session]);
		const meta = EncryptionService.getPersistedMeta();
		expect(meta).not.toBeNull();

		const jsonl = conversationsStore.serializeSessionToJsonl(encrypted, {
			encryptionMeta: meta ?? undefined
		});
		const [headerLine, ...messageLines] = jsonl.split('\n');
		const header = JSON.parse(headerLine);

		expect(header.encryption).toEqual(meta);
		expect(EncryptionService.isEncryptedValue(header.name)).toBe(true);

		const contents = messageLines.map((line) => JSON.parse(line).message.content as string);
		const rootIndex = contents.indexOf('');
		expect(rootIndex).not.toBe(-1);
		const bodies = contents.filter((_, index) => index !== rootIndex);
		expect(bodies).toHaveLength(1);
		expect(EncryptionService.isEncryptedValue(bodies[0])).toBe(true);

		// the plaintext export path is untouched
		const plainJsonl = conversationsStore.serializeSessionToJsonl(session);
		const plainHeader = JSON.parse(plainJsonl.split('\n')[0]);
		expect(plainHeader.encryption).toBeUndefined();
		expect(plainHeader.name).toBe('secret chat');
	});

	it('restores an encrypted export with its passphrase, as on another machine', async () => {
		await EncryptionService.setupWithPassphrase('pw');

		const conv = await DatabaseService.createConversation('secret chat');
		const rootId = await DatabaseService.createRootMessage(conv.id);
		await DatabaseService.createMessageBranch(
			{
				convId: conv.id,
				type: MessageType.TEXT,
				timestamp: Date.now(),
				role: MessageRole.USER,
				content: 'secret body',
				parent: null,
				children: []
			} as Omit<DatabaseMessage, 'id'>,
			rootId
		);

		const fetched = await DatabaseService.getConversationsWithMessages([conv.id]);
		const [encrypted] = await DatabaseService.encryptForExport([...fetched.values()]);
		const meta = EncryptionService.getPersistedMeta();
		expect(meta).not.toBeNull();

		const jsonl = conversationsStore.serializeSessionToJsonl(encrypted, {
			encryptionMeta: meta ?? undefined
		});

		// simulate another machine: parse the file back and unlock with the passphrase
		const [parsed] = conversationsStore.parseSessionsJsonl(jsonl);
		const headerMeta = (parsed.conv as DatabaseConversation & { encryption?: EncryptionMeta })
			.encryption;
		expect(headerMeta).toEqual(meta);

		expect(await EncryptionService.unwrapDekWithPassphrase(headerMeta!, 'wrong')).toBeNull();
		const key = await EncryptionService.unwrapDekWithPassphrase(headerMeta!, 'pw');
		expect(key).not.toBeNull();

		const [decrypted] = await DatabaseService.decryptImportedData([parsed], key!);
		expect(decrypted.conv.name).toBe('secret chat');
		expect((decrypted.conv as { encryption?: unknown }).encryption).toBeUndefined();
		expect(decrypted.messages.map((m) => m.content)).toContain('secret body');
	});
});
