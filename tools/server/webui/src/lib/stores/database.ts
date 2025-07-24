import Dexie, { type EntityTable } from 'dexie';
import type { DatabaseChat, DatabaseChatMessage, DatabaseAppSettings } from '$lib/types/database';

class ChatDatabase extends Dexie {
	chats!: EntityTable<DatabaseChat, 'id'>;
	messages!: EntityTable<DatabaseChatMessage, 'id'>;
	settings!: EntityTable<DatabaseAppSettings, 'id'>;

	constructor() {
		super('LlamaServerDB');

		this.version(1).stores({
			chats: 'id, name, createdAt, updatedAt, messageCount',
			messages: 'id, chatId, role, timestamp, [chatId+timestamp]',
			settings: 'id'
		});
	}
}

export const db = new ChatDatabase();
