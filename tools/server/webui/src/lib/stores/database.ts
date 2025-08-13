import Dexie, { type EntityTable } from 'dexie';

class LlamacppDatabase extends Dexie {
	conversations!: EntityTable<DatabaseConversation, string>;
	messages!: EntityTable<DatabaseMessage, string>;
	settings!: EntityTable<DatabaseAppSettings, string>;

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
