import { db } from '$lib/stores/database';

export class DatabaseService {
	static async createConversation(name: string): Promise<DatabaseConversation> {
		const conversation: DatabaseConversation = {
			id: crypto.randomUUID(),
			name,
			lastModified: Date.now(),
			currNode: ''
		};

		await db.conversations.add(conversation);
		return conversation;
	}

	static async getConversation(id: string): Promise<DatabaseConversation | undefined> {
		return await db.conversations.get(id);
	}

	static async getAllConversations(): Promise<DatabaseConversation[]> {
		return await db.conversations.orderBy('lastModified').reverse().toArray();
	}

	static async updateConversation(id: string, updates: Partial<Omit<DatabaseConversation, 'id'>>): Promise<void> {
		await db.conversations.update(id, {
			...updates,
			lastModified: Date.now()
		});
	}

	static async deleteConversation(id: string): Promise<void> {
		await db.transaction('rw', [db.conversations, db.messages], async () => {
			await db.conversations.delete(id);
			await db.messages.where('convId').equals(id).delete();
		});
	}

	static async addMessage(
		message: Omit<DatabaseMessage, 'id'>
	): Promise<DatabaseMessage> {
		const newMessage: DatabaseMessage = {
			...message,
			id: crypto.randomUUID()
		};

		await db.messages.add(newMessage);
		return newMessage;
	}

	static async getConversationMessages(convId: string): Promise<DatabaseMessage[]> {
		return await db.messages.where('convId').equals(convId).sortBy('timestamp');
	}

	static async updateMessage(
		id: string,
		updates: Partial<Omit<DatabaseMessage, 'id'>>
	): Promise<void> {
		await db.messages.update(id, updates);
	}

	static async deleteMessage(id: string): Promise<void> {
		await db.messages.delete(id);
	}

	static async getSettings(): Promise<DatabaseAppSettings> {
		let settings = await db.settings.get('default');

		if (!settings) {
			settings = {
				id: 'default',
				theme: 'system',
				model: 'llama-3.2-3b-instruct',
				temperature: 0.7,
				maxTokens: 2048,
				topP: 0.9,
				topK: 40,
				repeatPenalty: 1.1,
				seed: -1,
				systemPrompt: 'You are a helpful AI assistant.'
			};
			await db.settings.add(settings);
		}

		return settings;
	}

	static async updateSettings(updates: Partial<Omit<DatabaseAppSettings, 'id'>>): Promise<void> {
		await db.settings.update('default', updates);
	}
}
