// DocumentStoreBackend against an in-memory adapter with the same revision
// rules as llama-server's store (tools/server/server-storage.cpp).

import { MessageRole, MessageType } from '$lib/enums';
import { DatabaseService } from '$lib/services/database.service';
import { DocumentStoreBackend } from '$lib/services/document-store-backend';
import { PreferencesService } from '$lib/services/preferences.service';
import { LlamaServerStorageAdapter, StorageConflictError } from '$lib/storage-adapters';
import type { StorageAdapter, StoredConversation } from '$lib/types';
import type { DatabaseMessage } from '$lib/types/database';
import { ApiError } from '$lib/utils/api-fetch';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const apiFetch = vi.hoisted(() => vi.fn());

vi.mock('$lib/utils/api-fetch', async (importOriginal) => ({
	...(await importOriginal<object>()),
	apiFetch
}));

const docs = new Map<string, StoredConversation>();
const writes: string[] = [];
const memoryAdapter: StorageAdapter = {
	async deleteConversation(id) {
		writes.push(`delete ${id}`);
		docs.delete(id);
	},
	async getConversation(id) {
		const doc = docs.get(id);

		return doc ? structuredClone(doc) : null;
	},
	async getSettings() {
		return {};
	},
	async listConversations() {
		return [...docs.values()].map(({ conv, rev }) => structuredClone({ conv, rev }));
	},
	async putConversation(doc) {
		const current = docs.get(doc.conv.id)?.rev ?? 0;

		if (doc.rev !== current) throw new StorageConflictError(doc.conv.id);

		writes.push(`put ${doc.conv.id}`);
		docs.set(doc.conv.id, structuredClone({ ...doc, rev: current + 1 }));

		return current + 1;
	},
	async updateSettings() {}
};
const newBackend = () => new DocumentStoreBackend(memoryAdapter);

function userMessage(convId: string, content: string): Omit<DatabaseMessage, 'id'> {
	return {
		children: [],
		content,
		convId,
		parent: null,
		role: MessageRole.USER,
		timestamp: Date.now(),
		type: MessageType.TEXT
	};
}

beforeEach(() => {
	docs.clear();
	writes.length = 0;
});

describe('DocumentStoreBackend', () => {
	it('writes one document per conversation with the tree DatabaseService builds', async () => {
		DatabaseService.useBackend(newBackend());

		const conv = await DatabaseService.createConversation('a');
		const rootId = await DatabaseService.createRootMessage(conv.id);
		const user = await DatabaseService.createMessageBranch(userMessage(conv.id, 'hi'), rootId);
		const stored = docs.get(conv.id)!;
		const messages = stored.messages as DatabaseMessage[];

		expect(stored.rev).toBe(3);
		expect(messages.find((m) => m.id === rootId)?.children).toEqual([user.id]);
		expect((stored.conv as { currNode: string }).currNode).toBe(user.id);
	});

	it('replays its edits on a document another client changed meanwhile', async () => {
		DatabaseService.useBackend(newBackend());

		const conv = await DatabaseService.createConversation('a');
		const rootId = await DatabaseService.createRootMessage(conv.id);
		const first = await DatabaseService.createMessageBranch(userMessage(conv.id, 'one'), rootId);
		const second = await DatabaseService.createMessageBranch(userMessage(conv.id, 'two'), rootId);
		// two tabs with the conversation open
		const tabA = newBackend();
		const tabB = newBackend();

		await tabA.getMessages([conv.id]);
		await tabB.getMessages([conv.id]);
		await tabA.updateMessage(first.id, { content: 'one, edited in A' });
		await tabB.updateMessage(second.id, { content: 'two, edited in B' });

		const messages = docs.get(conv.id)!.messages as DatabaseMessage[];

		expect(messages.find((m) => m.id === first.id)?.content).toBe('one, edited in A');
		expect(messages.find((m) => m.id === second.id)?.content).toBe('two, edited in B');
	});

	it('restores the cached document when a transaction fails', async () => {
		const backend = newBackend();

		DatabaseService.useBackend(backend);

		const conv = await DatabaseService.createConversation('a');
		const rootId = await DatabaseService.createRootMessage(conv.id);

		writes.length = 0;

		await expect(
			backend.transaction(async () => {
				await backend.updateMessage(rootId, { content: 'changed' });

				throw new Error('boom');
			})
		).rejects.toThrow('boom');

		expect((await backend.getMessage(rootId))?.content).toBe('');
		expect(writes).toEqual([]);
	});

	it('saves a write that joined a transaction which closed while it loaded', async () => {
		DatabaseService.useBackend(newBackend());

		const a = await DatabaseService.createConversation('a');
		const b = await DatabaseService.createConversation('b');
		const backend = newBackend();

		let joined: Promise<void> = Promise.resolve();

		await backend.transaction(async () => {
			await backend.updateConversation(a.id, { name: 'a2' });
			// not awaited: still loading b when the transaction ends
			joined = backend.updateConversation(b.id, { name: 'b2' });
		});
		await joined;

		expect((docs.get(a.id)!.conv as { name: string }).name).toBe('a2');
		expect((docs.get(b.id)!.conv as { name: string }).name).toBe('b2');
	});

	it('deletes a conversation document and drops it from the list', async () => {
		DatabaseService.useBackend(newBackend());

		const keep = await DatabaseService.createConversation('keep');
		const drop = await DatabaseService.createConversation('drop');

		await DatabaseService.deleteConversation(drop.id);

		expect(docs.has(drop.id)).toBe(false);
		expect((await DatabaseService.getAllConversations()).map((c) => c.id)).toEqual([keep.id]);
	});
});

describe('PreferencesService remote mode', () => {
	it('keeps user data on the remote and device state local, batching writes', async () => {
		const store = new Map<string, string>();
		const push = vi.fn(async () => {});

		(globalThis as unknown as { localStorage: Partial<Storage> }).localStorage = {
			getItem: (key) => store.get(key) ?? null,
			removeItem: (key) => void store.delete(key),
			setItem: (key, value) => void store.set(key, value)
		};

		PreferencesService.useRemote({ 'LlamaUi.favoriteModels': '["a"]' }, push);

		expect(PreferencesService.getItem('LlamaUi.favoriteModels')).toBe('["a"]');

		PreferencesService.setItem('LlamaUi.favoriteModels', '["b"]');
		PreferencesService.setItem('LlamaUi.hiddenModels', '["c"]');
		PreferencesService.removeItem('LlamaUi.recentModels');
		PreferencesService.setItem('LlamaUi.sidebarWidth', '300');

		await Promise.resolve();
		await Promise.resolve();

		expect(push).toHaveBeenCalledTimes(1);
		expect(push).toHaveBeenCalledWith(
			{ 'LlamaUi.favoriteModels': '["b"]', 'LlamaUi.hiddenModels': '["c"]' },
			['LlamaUi.recentModels']
		);
		expect(store.get('LlamaUi.sidebarWidth')).toBe('300');
		expect(store.has('LlamaUi.favoriteModels')).toBe(false);
	});
});

describe('LlamaServerStorageAdapter', () => {
	it('maps the server answers onto the adapter contract', async () => {
		const adapter = new LlamaServerStorageAdapter();
		const conv = { currNode: null, id: 'a', lastModified: 0, name: 'a' };

		apiFetch.mockRejectedValueOnce(new ApiError('not found', 404));
		expect(await adapter.getConversation('a')).toBeNull();

		apiFetch.mockRejectedValueOnce(new ApiError('conflict', 409));
		await expect(adapter.putConversation({ conv, messages: [], rev: 0 })).rejects.toBeInstanceOf(
			StorageConflictError
		);

		apiFetch.mockResolvedValueOnce({ rev: 1 });
		expect(await adapter.putConversation({ conv, messages: [], rev: 0 })).toBe(1);
		expect(apiFetch).toHaveBeenLastCalledWith('./storage/conversations/a', expect.anything());
	});
});
