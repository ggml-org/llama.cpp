import { ChatService } from '$lib/services/chat';
import { DatabaseService } from '$lib/services';
import type { DatabaseChat, DatabaseChatMessage } from '$lib/app';
import { goto } from '$app/navigation';
import { browser } from '$app/environment';

class ChatStore {
	activeChat = $state<DatabaseChat | null>(null);
	activeChatMessages = $state<DatabaseChatMessage[]>([]);
	chats = $state<DatabaseChat[]>([]);
	currentResponse = $state('');
	isInitialized = $state(false);
	isLoading = $state(false);

	constructor() {
		if (browser) {
			this.initialize();
		}
	}

	async initialize() {
		try {
			await this.loadChats();
			this.isInitialized = true;
		} catch (error) {
			console.error('Failed to initialize chat store:', error);
		}
	}

	async loadChats() {
		this.chats = await DatabaseService.getAllChats();
	}

	async createChat(name?: string, systemPrompt?: string): Promise<string> {
		const chatName = name || `Chat ${new Date().toLocaleString()}`;

		const chat = await DatabaseService.createChat(chatName, systemPrompt);

		this.chats.unshift(chat);

		this.activeChat = chat;
		this.activeChatMessages = [];

		await goto(`/chat/${chat.id}`);

		return chat.id;
	}

	async loadChat(chatId: string): Promise<boolean> {
		try {
			const chat = await DatabaseService.getChat(chatId);
			if (!chat) {
				return false;
			}

			this.activeChat = chat;
			this.activeChatMessages = await DatabaseService.getChatMessages(chatId);
			return true;
		} catch (error) {
			console.error('Failed to load chat:', error);
			return false;
		}
	}

	async addMessage(
		role: 'user' | 'assistant' | 'system',
		content: string
	): Promise<DatabaseChatMessage | null> {
		if (!this.activeChat) {
			console.error('No active chat when trying to add message');
			return null;
		}

		try {
			const message = await DatabaseService.addMessage({
				chatId: this.activeChat.id,
				role,
				content,
				timestamp: Date.now()
			});

			this.activeChatMessages.push(message);

			const chatIndex = this.chats.findIndex((c) => c.id === this.activeChat!.id);
			if (chatIndex !== -1) {
				this.chats[chatIndex].messageCount += 1;
				this.chats[chatIndex].updatedAt = Date.now();

				const updatedChat = this.chats.splice(chatIndex, 1)[0];
				this.chats.unshift(updatedChat);
			}

			return message;
		} catch (error) {
			console.error('Failed to add message:', error);
			return null;
		}
	}

	async sendMessage(content: string): Promise<void> {
		if (!content.trim() || this.isLoading) return;

		if (!this.activeChat) {
			await this.createChat();
		}

		if (!this.activeChat) {
			console.error('No active chat available for sending message');
			return;
		}
		this.isLoading = true;
		this.currentResponse = '';

		try {
			const userMessage = await this.addMessage('user', content);
			if (!userMessage) {
				throw new Error('Failed to add user message');
			}

			const allMessages = await DatabaseService.getChatMessages(this.activeChat.id);

			const assistantMessage = await this.addMessage('assistant', '');
			if (!assistantMessage) {
				throw new Error('Failed to create assistant message');
			}

			let streamedContent = '';
			const chatService = new ChatService();
			await chatService.sendChatCompletion(
				allMessages,
				{
					stream: true,
					temperature: 0.7,
					max_tokens: 2048,
					onChunk: (chunk: string) => {
						streamedContent += chunk;
						this.currentResponse = streamedContent;

						const messageIndex = this.activeChatMessages.findIndex(
							(m) => m.id === assistantMessage.id
						);
						if (messageIndex !== -1) {
							this.activeChatMessages[messageIndex].content = streamedContent;
						}
					},
					onComplete: async () => {
						await DatabaseService.updateMessage(assistantMessage.id, {
							content: streamedContent
						});

						this.isLoading = false;
						this.currentResponse = '';
					},
					onError: (error: Error) => {
						console.error('Streaming error:', error);
						this.isLoading = false;
						this.currentResponse = '';

						const messageIndex = this.activeChatMessages.findIndex(
							(m) => m.id === assistantMessage.id
						);
						if (messageIndex !== -1) {
							this.activeChatMessages[messageIndex].content = `Error: ${error.message}`;
						}
					}
				}
			);
		} catch (error) {
			console.error('Failed to send message:', error);
			this.isLoading = false;
		}
	}

	stopGeneration() {
		this.isLoading = false;
		this.currentResponse = '';
	}

	async updateChatName(chatId: string, name: string): Promise<void> {
		try {
			await DatabaseService.updateChat(chatId, { name });

			const chatIndex = this.chats.findIndex((c) => c.id === chatId);
			if (chatIndex !== -1) {
				this.chats[chatIndex].name = name;
			}

			if (this.activeChat?.id === chatId) {
				this.activeChat.name = name;
			}
		} catch (error) {
			console.error('Failed to update chat name:', error);
		}
	}

	async deleteChat(chatId: string): Promise<void> {
		try {
			await DatabaseService.deleteChat(chatId);

			this.chats = this.chats.filter((c) => c.id !== chatId);

			if (this.activeChat?.id === chatId) {
				this.activeChat = null;
				this.activeChatMessages = [];
				await goto('/?new_chat=true');
			}
		} catch (error) {
			console.error('Failed to delete chat:', error);
		}
	}

	clearActiveChat() {
		this.activeChat = null;
		this.activeChatMessages = [];
		this.currentResponse = '';
		this.isLoading = false;
	}
}

export const chatStore = new ChatStore();

export const chats = () => chatStore.chats;
export const activeChat = () => chatStore.activeChat;
export const activeChatMessages = () => chatStore.activeChatMessages;
export const isLoading = () => chatStore.isLoading;
export const currentResponse = () => chatStore.currentResponse;
export const isInitialized = () => chatStore.isInitialized;

export const createChat = chatStore.createChat.bind(chatStore);
export const loadChat = chatStore.loadChat.bind(chatStore);
export const sendMessage = chatStore.sendMessage.bind(chatStore);
export const updateChatName = chatStore.updateChatName.bind(chatStore);
export const deleteChat = chatStore.deleteChat.bind(chatStore);
export const clearActiveChat = chatStore.clearActiveChat.bind(chatStore);

export function stopGeneration() {
	chatStore.stopGeneration();
}
export const chatMessages = () => chatStore.activeChatMessages;
