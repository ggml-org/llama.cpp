import type { DatabaseAppSettings, DatabaseChat, DatabaseChatMessage } from '$lib/app';
import { db } from '$lib/stores/database';

export class DatabaseService {
	static async createChat(name: string, systemPrompt?: string): Promise<DatabaseChat> {
		const chat: DatabaseChat = {
			id: crypto.randomUUID(),
			name,
			createdAt: Date.now(),
			updatedAt: Date.now(),
			messageCount: 0,
			systemPrompt
		};

		await db.chats.add(chat);
		return chat;
	}

	static async getChat(id: string): Promise<DatabaseChat | undefined> {
		return await db.chats.get(id);
	}

	static async getAllChats(): Promise<DatabaseChat[]> {
		return await db.chats.orderBy('updatedAt').reverse().toArray();
	}

	static async updateChat(id: string, updates: Partial<Omit<DatabaseChat, 'id'>>): Promise<void> {
		await db.chats.update(id, {
			...updates,
			updatedAt: Date.now()
		});
	}

	static async deleteChat(id: string): Promise<void> {
		await db.transaction('rw', [db.chats, db.messages], async () => {
			await db.chats.delete(id);
			await db.messages.where('chatId').equals(id).delete();
		});
	}

	static async addMessage(
		message: Omit<DatabaseChatMessage, 'id'>
	): Promise<DatabaseChatMessage> {
		const newMessage: DatabaseChatMessage = {
			...message,
			id: crypto.randomUUID()
		};

		await db.transaction('rw', [db.messages, db.chats], async () => {
			await db.messages.add(newMessage);

			const messageCount = await db.messages.where('chatId').equals(message.chatId).count();

			await db.chats.update(message.chatId, {
				messageCount,
				updatedAt: Date.now()
			});
		});

		return newMessage;
	}

	static async getChatMessages(chatId: string): Promise<DatabaseChatMessage[]> {
		return await db.messages.where('chatId').equals(chatId).sortBy('timestamp');
	}

	static async updateMessage(
		id: string,
		updates: Partial<Omit<DatabaseChatMessage, 'id'>>
	): Promise<void> {
		await db.messages.update(id, updates);
	}

	static async deleteMessage(id: string): Promise<void> {
		const message = await db.messages.get(id);
		if (!message) return;

		await db.transaction('rw', [db.messages, db.chats], async () => {
			await db.messages.delete(id);

			// Update chat message count
			const messageCount = await db.messages.where('chatId').equals(message.chatId).count();
			await db.chats.update(message.chatId, {
				messageCount,
				updatedAt: Date.now()
			});
		});
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
