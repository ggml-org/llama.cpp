/**
 * IdbBackend - IndexedDB (Dexie) implementation of StorageBackend
 *
 * Row-level reads and writes over the conversations/messages tables. Calls
 * made inside `transaction()` join the same Dexie transaction.
 */

import { IDXDB_STORES, IDXDB_TABLES, STORAGE_APP_NAME } from '$lib/constants';
import type { StorageBackend } from '$lib/types/database';
import Dexie, { type EntityTable } from 'dexie';

class LlamaUiDatabase extends Dexie {
	[IDXDB_TABLES.conversations]!: EntityTable<DatabaseConversation, string>;
	[IDXDB_TABLES.messages]!: EntityTable<DatabaseMessage, string>;

	constructor() {
		super(STORAGE_APP_NAME);

		this.version(1).stores(IDXDB_STORES);
	}
}

export class IdbBackend implements StorageBackend {
	private readonly db = new LlamaUiDatabase();

	private get conversations() {
		return this.db[IDXDB_TABLES.conversations];
	}

	private get messages() {
		return this.db[IDXDB_TABLES.messages];
	}

	async addConversation(conv: DatabaseConversation): Promise<void> {
		await this.conversations.add(conv);
	}

	async addMessages(messages: DatabaseMessage[]): Promise<void> {
		await this.messages.bulkAdd(messages);
	}

	async deleteConversations(ids: string[]): Promise<void> {
		await this.conversations.bulkDelete(ids);
		await this.messages.where('convId').anyOf(ids).delete();
	}

	async deleteMessages(ids: string[]): Promise<void> {
		await this.messages.bulkDelete(ids);
	}

	async getConversations(ids: string[]): Promise<(DatabaseConversation | undefined)[]> {
		return await this.conversations.bulkGet(ids);
	}

	async getMessage(id: string): Promise<DatabaseMessage | undefined> {
		return await this.messages.get(id);
	}

	async getMessages(convIds: string[]): Promise<DatabaseMessage[]> {
		return await this.messages.where('convId').anyOf(convIds).toArray();
	}

	async listConversations(): Promise<DatabaseConversation[]> {
		return await this.conversations.orderBy('lastModified').reverse().toArray();
	}

	async putConversations(convs: DatabaseConversation[]): Promise<void> {
		await this.conversations.bulkPut(convs);
	}

	async putMessages(messages: DatabaseMessage[]): Promise<void> {
		await this.messages.bulkPut(messages);
	}

	async transaction<T>(fn: () => Promise<T>): Promise<T> {
		return await this.db.transaction('rw', [this.conversations, this.messages], fn);
	}

	async updateConversation(
		id: string,
		updates: Partial<Omit<DatabaseConversation, 'id'>>
	): Promise<void> {
		await this.conversations.update(id, updates);
	}

	async updateMessage(id: string, updates: Partial<Omit<DatabaseMessage, 'id'>>): Promise<void> {
		await this.messages.update(id, updates);
	}
}
