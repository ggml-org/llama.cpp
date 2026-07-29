import { afterEach, describe, expect, it } from 'vitest';
import Dexie from 'dexie';
import { DatabaseService } from '$lib/services/database.service';
import { EncryptionService } from '$lib/services/encryption.service';
import { IDXDB_STORES, IDXDB_TABLES, STORAGE_APP_NAME } from '$lib/constants';
import { AttachmentType, MessageRole, MessageType } from '$lib/enums';

function rawDb(): Dexie {
	const raw = new Dexie(STORAGE_APP_NAME);
	raw.version(1).stores(IDXDB_STORES);
	return raw;
}

async function rawConversation(id: string): Promise<DatabaseConversation> {
	const raw = rawDb();
	try {
		return (await raw.table(IDXDB_TABLES.conversations).get(id)) as DatabaseConversation;
	} finally {
		raw.close();
	}
}

async function rawMessage(id: string): Promise<DatabaseMessage> {
	const raw = rawDb();
	try {
		return (await raw.table(IDXDB_TABLES.messages).get(id)) as DatabaseMessage;
	} finally {
		raw.close();
	}
}

function makeMessage(convId: string, content: string): Omit<DatabaseMessage, 'id'> {
	return {
		convId,
		type: MessageType.TEXT,
		timestamp: Date.now(),
		role: MessageRole.USER,
		content,
		parent: null,
		children: []
	} as Omit<DatabaseMessage, 'id'>;
}

async function seedConversation(
	name: string,
	content: string
): Promise<{ conv: DatabaseConversation; message: DatabaseMessage }> {
	const conv = await DatabaseService.createConversation(name);
	const rootId = await DatabaseService.createRootMessage(conv.id);
	const message = await DatabaseService.createMessageBranch(makeMessage(conv.id, content), rootId);
	return { conv, message };
}

afterEach(async () => {
	EncryptionService.disable();
	const conversations = await DatabaseService.getAllConversations();
	await DatabaseService.bulkDeleteConversations(conversations.map((conv) => conv.id));
});

describe('DatabaseService encryption seam', () => {
	it('encrypts conversation names and message fields at rest when unlocked', async () => {
		await EncryptionService.setupWithPassphrase('pw');

		const extra = [
			{ type: AttachmentType.TEXT, name: 'notes.txt', content: 'file body' }
		] as DatabaseMessageExtra[];
		const conv = await DatabaseService.createConversation('secret chat');
		const rootId = await DatabaseService.createRootMessage(conv.id);
		const message = await DatabaseService.createMessageBranch(
			{ ...makeMessage(conv.id, 'hello secret'), reasoningContent: 'hmm', extra },
			rootId
		);

		// at rest: ciphertext
		const storedConv = await rawConversation(conv.id);
		const storedMessage = await rawMessage(message.id);
		expect(EncryptionService.isEncryptedValue(storedConv.name)).toBe(true);
		expect(EncryptionService.isEncryptedValue(storedMessage.content)).toBe(true);
		expect(EncryptionService.isEncryptedValue(storedMessage.reasoningContent)).toBe(true);
		expect(storedMessage.extra).toHaveLength(1);
		expect(storedMessage.extra?.[0].type).toBe(AttachmentType.ENCRYPTED);

		// through the service: plaintext
		expect((await DatabaseService.getConversation(conv.id))?.name).toBe('secret chat');
		const messages = await DatabaseService.getConversationMessages(conv.id);
		const stored = messages.find((m) => m.id === message.id);
		expect(stored?.content).toBe('hello secret');
		expect(stored?.reasoningContent).toBe('hmm');
		expect(stored?.extra).toEqual(extra);
	});

	it('passes through plaintext records written before encryption was enabled', async () => {
		const { conv, message } = await seedConversation('plain chat', 'plain body');

		await EncryptionService.setupWithPassphrase('pw');

		expect((await DatabaseService.getConversation(conv.id))?.name).toBe('plain chat');
		const messages = await DatabaseService.getConversationMessages(conv.id);
		expect(messages.find((m) => m.id === message.id)?.content).toBe('plain body');

		const storedMessage = await rawMessage(message.id);
		expect(storedMessage.content).toBe('plain body');
	});

	it('encrypts updateConversation and updateMessage partials', async () => {
		await EncryptionService.setupWithPassphrase('pw');
		const { conv, message } = await seedConversation('before', 'before body');

		await DatabaseService.updateConversation(conv.id, { name: 'renamed' });
		await DatabaseService.updateMessage(message.id, {
			content: 'edited',
			reasoningContent: 'thinking',
			toolCalls: '[{"name":"tool"}]'
		});

		const storedConv = await rawConversation(conv.id);
		const storedMessage = await rawMessage(message.id);
		expect(EncryptionService.isEncryptedValue(storedConv.name)).toBe(true);
		expect(EncryptionService.isEncryptedValue(storedMessage.content)).toBe(true);
		expect(EncryptionService.isEncryptedValue(storedMessage.reasoningContent)).toBe(true);
		expect(EncryptionService.isEncryptedValue(storedMessage.toolCalls)).toBe(true);

		expect((await DatabaseService.getConversation(conv.id))?.name).toBe('renamed');
		const messages = await DatabaseService.getConversationMessages(conv.id);
		const updated = messages.find((m) => m.id === message.id);
		expect(updated?.content).toBe('edited');
		expect(updated?.reasoningContent).toBe('thinking');
		expect(updated?.toolCalls).toBe('[{"name":"tool"}]');
	});

	it('does not re-encrypt values that already carry the prefix', async () => {
		await EncryptionService.setupWithPassphrase('pw');
		const { message } = await seedConversation('chat', 'body');

		const storedMessage = await rawMessage(message.id);
		const ciphertext = storedMessage.content;

		await DatabaseService.updateMessage(message.id, { content: ciphertext });

		expect((await rawMessage(message.id)).content).toBe(ciphertext);
	});

	it('throws on writes when encryption is enabled but locked', async () => {
		await EncryptionService.setupWithPassphrase('pw');
		EncryptionService.lock();

		await expect(DatabaseService.createConversation('nope')).rejects.toThrow(
			'Encryption is locked'
		);
		await expect(DatabaseService.updateMessage('missing', { content: 'x' })).rejects.toThrow(
			'Encryption is locked'
		);
	});

	it('encryptAllStoredData encrypts pre-existing plaintext records, idempotently', async () => {
		const { conv, message } = await seedConversation('legacy chat', 'legacy body');

		await EncryptionService.setupWithPassphrase('pw');
		await DatabaseService.encryptAllStoredData();

		expect(EncryptionService.isEncryptedValue((await rawConversation(conv.id)).name)).toBe(true);
		expect(EncryptionService.isEncryptedValue((await rawMessage(message.id)).content)).toBe(true);

		expect((await DatabaseService.getConversation(conv.id))?.name).toBe('legacy chat');
		const messages = await DatabaseService.getConversationMessages(conv.id);
		expect(messages.find((m) => m.id === message.id)?.content).toBe('legacy body');

		// re-running is a no-op and keeps data readable
		await DatabaseService.encryptAllStoredData();
		expect((await DatabaseService.getConversation(conv.id))?.name).toBe('legacy chat');
	});

	it('decryptAllStoredData restores plaintext records, including extras', async () => {
		await EncryptionService.setupWithPassphrase('pw');

		const extra = [
			{ type: AttachmentType.TEXT, name: 'notes.txt', content: 'file body' }
		] as DatabaseMessageExtra[];
		const conv = await DatabaseService.createConversation('secret chat');
		const rootId = await DatabaseService.createRootMessage(conv.id);
		const message = await DatabaseService.createMessageBranch(
			{ ...makeMessage(conv.id, 'secret body'), extra },
			rootId
		);

		await DatabaseService.decryptAllStoredData();

		const storedConv = await rawConversation(conv.id);
		const storedMessage = await rawMessage(message.id);
		expect(storedConv.name).toBe('secret chat');
		expect(storedMessage.content).toBe('secret body');
		expect(storedMessage.extra).toEqual(extra);

		// reads keep working with encryption still enabled
		expect((await DatabaseService.getConversation(conv.id))?.name).toBe('secret chat');
		const messages = await DatabaseService.getConversationMessages(conv.id);
		expect(messages.find((m) => m.id === message.id)?.extra).toEqual(extra);
	});

	it('keeps ciphertext at rest when forking an encrypted conversation', async () => {
		await EncryptionService.setupWithPassphrase('pw');
		const { conv, message } = await seedConversation('source chat', 'source body');
		const sourceCiphertext = (await rawMessage(message.id)).content;

		const fork = await DatabaseService.forkConversation(conv.id, message.id, {
			name: 'forked chat',
			includeAttachments: true
		});

		expect(fork.name).toBe('forked chat');

		const storedFork = await rawConversation(fork.id);
		expect(EncryptionService.isEncryptedValue(storedFork.name)).toBe(true);

		const forkMessages = await DatabaseService.getConversationMessages(fork.id);
		const cloned = forkMessages.find((m) => m.content === 'source body');
		expect(cloned).toBeDefined();
		expect((await rawMessage(cloned!.id)).content).toBe(sourceCiphertext);
	});
});
