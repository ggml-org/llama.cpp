import Dexie, { type EntityTable } from 'dexie';

// Database schema interfaces
export interface ChatMessage {
	id: string;
	chatId: string;
	role: 'user' | 'assistant' | 'system';
	content: string;
	timestamp: number;
	tokens?: number;
	model?: string;
}

export interface Chat {
	id: string; // UUID v4
	name: string;
	createdAt: number;
	updatedAt: number;
	messageCount: number;
	model?: string;
	systemPrompt?: string;
}

export interface AppSettings {
	id: string;
	theme: 'light' | 'dark' | 'system';
	model: string;
	temperature: number;
	maxTokens: number;
	topP: number;
	topK: number;
	repeatPenalty: number;
	seed: number;
	systemPrompt: string;
	// Add other llama.cpp parameters as needed
}

// Dexie database class
class ChatDatabase extends Dexie {
	chats!: EntityTable<Chat, 'id'>;
	messages!: EntityTable<ChatMessage, 'id'>;
	settings!: EntityTable<AppSettings, 'id'>;

	constructor() {
		super('LlamaCppChatDB');

		this.version(1).stores({
			chats: 'id, name, createdAt, updatedAt, messageCount',
			messages: 'id, chatId, role, timestamp, [chatId+timestamp]',
			settings: 'id'
		});
	}
}

// Create database instance
export const db = new ChatDatabase();

// Database utility functions
export class DatabaseService {
	// Chat operations
	static async createChat(name: string, systemPrompt?: string): Promise<Chat> {
		const chat: Chat = {
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

	static async getChat(id: string): Promise<Chat | undefined> {
		return await db.chats.get(id);
	}

	static async getAllChats(): Promise<Chat[]> {
		return await db.chats.orderBy('updatedAt').reverse().toArray();
	}

	static async updateChat(id: string, updates: Partial<Omit<Chat, 'id'>>): Promise<void> {
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

	// Message operations
	static async addMessage(message: Omit<ChatMessage, 'id'>): Promise<ChatMessage> {
		const newMessage: ChatMessage = {
			...message,
			id: crypto.randomUUID()
		};

		await db.transaction('rw', [db.messages, db.chats], async () => {
			await db.messages.add(newMessage);
			
			// Update chat message count and timestamp
			const messageCount = await db.messages.where('chatId').equals(message.chatId).count();
			await db.chats.update(message.chatId, {
				messageCount,
				updatedAt: Date.now()
			});
		});

		return newMessage;
	}

	static async getChatMessages(chatId: string): Promise<ChatMessage[]> {
		return await db.messages
			.where('chatId')
			.equals(chatId)
			.sortBy('timestamp');
	}

	static async updateMessage(id: string, updates: Partial<Omit<ChatMessage, 'id'>>): Promise<void> {
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

	// Settings operations
	static async getSettings(): Promise<AppSettings> {
		let settings = await db.settings.get('default');
		
		if (!settings) {
			// Create default settings
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

	static async updateSettings(updates: Partial<Omit<AppSettings, 'id'>>): Promise<void> {
		await db.settings.update('default', updates);
	}
}
