import Dexie, { type EntityTable } from 'dexie';
import type { Conversation, Message, DatabaseAppSettings } from '$lib/types/database';

class LlamacppDatabase extends Dexie {
	conversations!: EntityTable<Conversation, 'id'>;
	messages!: EntityTable<Message, 'id'>;
	settings!: EntityTable<DatabaseAppSettings, 'id'>;

	constructor() {
		super('LlamacppWebui');

		this.version(1).stores({
			conversations: 'id, lastModified, currNode, name',
			messages: 'id, convId, type, role, timestamp, parent, children',
			settings: 'id'
		});
	}
}

export const db = new LlamacppDatabase();
