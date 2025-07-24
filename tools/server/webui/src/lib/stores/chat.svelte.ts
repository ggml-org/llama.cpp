import { DatabaseService, type Chat, type ChatMessage } from './database';
import { ApiService } from '$lib/services/api';
import { goto } from '$app/navigation';
import { browser } from '$app/environment';

// Global reactive chat state
class ChatStore {
	// All chats loaded from IndexedDB
	chats = $state<Chat[]>([]);
	
	// Currently active chat
	activeChat = $state<Chat | null>(null);
	activeChatMessages = $state<ChatMessage[]>([]);
	
	// UI state
	isLoading = $state(false);
	currentResponse = $state('');
	isInitialized = $state(false);

	constructor() {
		if (browser) {
			this.initialize();
		}
	}

	// Initialize store by loading data from IndexedDB
	async initialize() {
		try {
			await this.loadChats();
			this.isInitialized = true;
		} catch (error) {
			console.error('Failed to initialize chat store:', error);
		}
	}

	// Load all chats from IndexedDB
	async loadChats() {
		this.chats = await DatabaseService.getAllChats();
	}

	// Create a new chat
	async createChat(name?: string, systemPrompt?: string): Promise<string> {
		const chatName = name || `Chat ${new Date().toLocaleString()}`;
		console.log('Creating new chat:', chatName);
		
		const chat = await DatabaseService.createChat(chatName, systemPrompt);
		console.log('Chat created in database:', chat);
		
		// Add to local state
		this.chats.unshift(chat);
		
		// Set as active chat immediately
		this.activeChat = chat;
		this.activeChatMessages = [];
		console.log('Set as active chat, messages cleared');
		
		// Navigate to new chat
		await goto(`/chat/${chat.id}`);
		console.log('Navigated to chat:', chat.id);
		
		return chat.id;
	}

	// Load a specific chat and its messages
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

	// Add a message to the active chat
	async addMessage(role: 'user' | 'assistant' | 'system', content: string): Promise<ChatMessage | null> {
		if (!this.activeChat) {
			console.error('No active chat when trying to add message');
			return null;
		}

		console.log('Adding message:', { role, content, chatId: this.activeChat.id });

		try {
			const message = await DatabaseService.addMessage({
				chatId: this.activeChat.id,
				role,
				content,
				timestamp: Date.now()
			});

			console.log('Message added to database:', message);

			// Add to local state
			this.activeChatMessages.push(message);
			console.log('Current activeChatMessages count:', this.activeChatMessages.length);

			// Update chat in local state
			const chatIndex = this.chats.findIndex(c => c.id === this.activeChat!.id);
			if (chatIndex !== -1) {
				this.chats[chatIndex].messageCount += 1;
				this.chats[chatIndex].updatedAt = Date.now();
				
				// Move to top of list
				const updatedChat = this.chats.splice(chatIndex, 1)[0];
				this.chats.unshift(updatedChat);
				console.log('Updated chat in local state');
			}

			return message;
		} catch (error) {
			console.error('Failed to add message:', error);
			return null;
		}
	}

	// Send a user message and get AI response
	async sendMessage(content: string): Promise<void> {
		if (!content.trim() || this.isLoading) return;

		console.log('Sending message:', content);
		console.log('Current active chat:', this.activeChat?.id);

		// If no active chat, create one first
		if (!this.activeChat) {
			console.log('No active chat, creating new one...');
			await this.createChat();
		}

		// Double-check we have an active chat
		if (!this.activeChat) {
			console.error('No active chat available for sending message');
			return;
		}

		console.log('Active chat confirmed:', this.activeChat.id);
		this.isLoading = true;
		this.currentResponse = '';

		try {
			// Add user message
			console.log('Adding user message...');
			const userMessage = await this.addMessage('user', content);
			if (!userMessage) {
				throw new Error('Failed to add user message');
			}
			console.log('User message added successfully');

			// Get all messages for context (including the new user message)
			const allMessages = await DatabaseService.getChatMessages(this.activeChat.id);
			console.log('Sending', allMessages.length, 'messages to API');

			// Create assistant message placeholder for streaming
			const assistantMessage = await this.addMessage('assistant', '');
			if (!assistantMessage) {
				throw new Error('Failed to create assistant message');
			}

			// Send to API with streaming
			let streamedContent = '';
			await ApiService.sendChatCompletion(
				allMessages,
				// onChunk - update the message content as we receive chunks
				(chunk: string) => {
					streamedContent += chunk;
					this.currentResponse = streamedContent;
					
					// Update the assistant message in local state
					const messageIndex = this.activeChatMessages.findIndex(m => m.id === assistantMessage.id);
					if (messageIndex !== -1) {
						this.activeChatMessages[messageIndex].content = streamedContent;
					}
				},
				// onComplete - finalize the message
				async () => {
					console.log('Streaming complete, final content length:', streamedContent.length);
					
					// Update the message in the database with final content
					await DatabaseService.updateMessage(assistantMessage.id, {
						content: streamedContent
					});
					
					this.isLoading = false;
					this.currentResponse = '';
					console.log('AI response completed and saved');
				},
				// onError - handle streaming errors
				(error: Error) => {
					console.error('Streaming error:', error);
					this.isLoading = false;
					this.currentResponse = '';
					
					// Update message with error
					const messageIndex = this.activeChatMessages.findIndex(m => m.id === assistantMessage.id);
					if (messageIndex !== -1) {
						this.activeChatMessages[messageIndex].content = `Error: ${error.message}`;
					}
				}
			);

		} catch (error) {
			console.error('Failed to send message:', error);
			this.isLoading = false;
		}
	}

	// Stop generation
	stopGeneration() {
		this.isLoading = false;
		this.currentResponse = '';
		console.log('Generation stopped');
		// TODO: Add actual abort controller logic when we implement it
	}

	// Update chat name
	async updateChatName(chatId: string, name: string): Promise<void> {
		try {
			await DatabaseService.updateChat(chatId, { name });
			
			// Update local state
			const chatIndex = this.chats.findIndex(c => c.id === chatId);
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

	// Delete a chat
	async deleteChat(chatId: string): Promise<void> {
		try {
			await DatabaseService.deleteChat(chatId);
			
			// Remove from local state
			this.chats = this.chats.filter(c => c.id !== chatId);
			
			// If this was the active chat, clear it
			if (this.activeChat?.id === chatId) {
				this.activeChat = null;
				this.activeChatMessages = [];
				// Navigate to home
				await goto('/');
			}
		} catch (error) {
			console.error('Failed to delete chat:', error);
		}
	}

	// Clear active chat (for navigation)
	clearActiveChat() {
		this.activeChat = null;
		this.activeChatMessages = [];
		this.currentResponse = '';
		this.isLoading = false;
	}
}

// Create global store instance
export const chatStore = new ChatStore();

// Export reactive getters for easy access
export const chats = () => chatStore.chats;
export const activeChat = () => chatStore.activeChat;
export const activeChatMessages = () => chatStore.activeChatMessages;
export const isLoading = () => chatStore.isLoading;
export const currentResponse = () => chatStore.currentResponse;
export const isInitialized = () => chatStore.isInitialized;

// Export store methods
export const createChat = chatStore.createChat.bind(chatStore);
export const loadChat = chatStore.loadChat.bind(chatStore);
export const sendMessage = chatStore.sendMessage.bind(chatStore);
export const updateChatName = chatStore.updateChatName.bind(chatStore);
export const deleteChat = chatStore.deleteChat.bind(chatStore);
export const clearActiveChat = chatStore.clearActiveChat.bind(chatStore);

// Stop generation function
export function stopGeneration() {
	chatStore.stopGeneration();
}

// Legacy exports for compatibility
export const chatMessages = () => chatStore.activeChatMessages;
