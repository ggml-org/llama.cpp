import { ChatService } from '$lib/services/chat';
import { DatabaseService } from '$lib/services';
import type { DatabaseChat, DatabaseChatMessage } from '$lib/app';
import { goto } from '$app/navigation';
import { browser } from '$app/environment';
import { extractPartialThinking } from '$lib/utils/thinking';

class ChatStore {
	activeChat = $state<DatabaseChat | null>(null);
	activeChatMessages = $state<DatabaseChatMessage[]>([]);
	chats = $state<DatabaseChat[]>([]);
	currentResponse = $state('');
	isInitialized = $state(false);
	isLoading = $state(false);
	private chatService = new ChatService();

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
			await this.chatService.sendChatCompletion(
				allMessages,
				{
					stream: true,
					temperature: 0.7,
					max_tokens: 2048,
					onChunk: (chunk: string) => {
						streamedContent += chunk;
						this.currentResponse = streamedContent;

						// Parse thinking content during streaming
						const partialThinking = extractPartialThinking(streamedContent);
						
						const messageIndex = this.activeChatMessages.findIndex(
							(m) => m.id === assistantMessage.id
						);
						if (messageIndex !== -1) {
							// Update message with parsed content
							this.activeChatMessages[messageIndex].content = partialThinking.remainingContent || streamedContent;
							// Update thinking content if present
							if (partialThinking.thinking) {
								this.activeChatMessages[messageIndex].thinking = partialThinking.thinking;
							}
						}
					},
					onComplete: async () => {
						// Update assistant message in database
						await DatabaseService.updateMessage(assistantMessage.id, {
							content: streamedContent
						});

						this.isLoading = false;
						this.currentResponse = '';
					},
					onError: (error: Error) => {
						// Don't log or show error if it's an AbortError (user stopped generation)
						if (error.name === 'AbortError' || error instanceof DOMException) {
							this.isLoading = false;
							this.currentResponse = '';
							return;
						}

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
			// Don't log or show error if it's an AbortError (user stopped generation)
			if (error instanceof Error && (error.name === 'AbortError' || error instanceof DOMException)) {
				this.isLoading = false;
				return;
			}

			console.error('Failed to send message:', error);
			this.isLoading = false;
		}
	}

	stopGeneration() {
		this.chatService.abort();
		this.isLoading = false;
		this.currentResponse = '';
	}

	async updateMessage(messageId: string, newContent: string): Promise<void> {
		if (!this.activeChat || this.isLoading) return;

		try {
			// Find the message to update
			const messageIndex = this.activeChatMessages.findIndex((m) => m.id === messageId);
			if (messageIndex === -1) {
				console.error('Message not found for update');
				return;
			}

			const messageToUpdate = this.activeChatMessages[messageIndex];
			const originalContent = messageToUpdate.content; // Store original content for rollback
			
			// Only allow updating user messages
			if (messageToUpdate.role !== 'user') {
				console.error('Only user messages can be edited');
				return;
			}

			// Update the message in local state immediately for UI responsiveness
			this.activeChatMessages[messageIndex].content = newContent;

			// Remove all messages after the updated message (including assistant responses)
			const messagesToRemove = this.activeChatMessages.slice(messageIndex + 1);
			for (const message of messagesToRemove) {
				await DatabaseService.deleteMessage(message.id);
			}

			// Update local state to remove subsequent messages
			this.activeChatMessages = this.activeChatMessages.slice(0, messageIndex + 1);

			// Update chat message count
			const chatIndex = this.chats.findIndex((c) => c.id === this.activeChat!.id);
			if (chatIndex !== -1) {
				this.chats[chatIndex].messageCount -= messagesToRemove.length;
				this.chats[chatIndex].updatedAt = Date.now();
			}

			// Regenerate response to the updated message
			// We need to generate a new assistant response based on the conversation history
			this.isLoading = true;
			this.currentResponse = '';

			try {
				const allMessages = await DatabaseService.getChatMessages(this.activeChat.id);

				const assistantMessage = await this.addMessage('assistant', '');
				if (!assistantMessage) {
					throw new Error('Failed to create assistant message');
				}

				let streamedContent = '';
				await this.chatService.sendChatCompletion(
					allMessages,
					{
						stream: true,
						temperature: 0.7,
						max_tokens: 2048,
						onChunk: (chunk: string) => {
							streamedContent += chunk;
							this.currentResponse = streamedContent;

							// Parse thinking content during streaming
							const partialThinking = extractPartialThinking(streamedContent);
							
							const messageIndex = this.activeChatMessages.findIndex(
								(m) => m.id === assistantMessage.id
							);
							if (messageIndex !== -1) {
								// Update message with parsed content
								this.activeChatMessages[messageIndex].content = partialThinking.remainingContent || streamedContent;
								// Update thinking content if present
								if (partialThinking.thinking) {
									this.activeChatMessages[messageIndex].thinking = partialThinking.thinking;
								}
							}
						},
						onComplete: async () => {
							// Update assistant message in database
							await DatabaseService.updateMessage(assistantMessage.id, {
								content: streamedContent
							});

							// Only now update the edited user message in database after successful completion
							await DatabaseService.updateMessage(messageId, { content: newContent });

							this.isLoading = false;
							this.currentResponse = '';
						},
						onError: (error: Error) => {
							// Don't log or show error if it's an AbortError (user stopped generation)
							if (error.name === 'AbortError' || error instanceof DOMException) {
								this.isLoading = false;
								this.currentResponse = '';
								return;
							}

							console.error('Streaming error:', error);
							this.isLoading = false;
							this.currentResponse = '';

							// Rollback the edited message to original content on error
							const editedMessageIndex = this.activeChatMessages.findIndex(
								(m) => m.id === messageId
							);
							if (editedMessageIndex !== -1) {
								this.activeChatMessages[editedMessageIndex].content = originalContent;
							}

							const assistantMessageIndex = this.activeChatMessages.findIndex(
								(m) => m.id === assistantMessage.id
							);
							if (assistantMessageIndex !== -1) {
								this.activeChatMessages[assistantMessageIndex].content = `Error: ${error.message}`;
							}
						}
					}
				);
			} catch (regenerateError) {
				console.error('Failed to regenerate response:', regenerateError);
				this.isLoading = false;
				
				// Rollback the edited message to original content on regeneration error
				const editedMessageIndex = this.activeChatMessages.findIndex(
					(m) => m.id === messageId
				);
				if (editedMessageIndex !== -1) {
					this.activeChatMessages[editedMessageIndex].content = originalContent;
				}
			}
		} catch (error) {
			// Don't log or show error if it's an AbortError (user stopped generation)
			if (error instanceof Error && (error.name === 'AbortError' || error instanceof DOMException)) {
				return;
			}

			console.error('Failed to update message:', error);
		}
	}

	async regenerateMessage(messageId: string): Promise<void> {
		if (!this.activeChat || this.isLoading) return;

		try {
			// Find the assistant message to regenerate
			const messageIndex = this.activeChatMessages.findIndex((m) => m.id === messageId);
			if (messageIndex === -1) {
				console.error('Message not found for regeneration');
				return;
			}

			const messageToRegenerate = this.activeChatMessages[messageIndex];
			
			// Only allow regenerating assistant messages
			if (messageToRegenerate.role !== 'assistant') {
				console.error('Only assistant messages can be regenerated');
				return;
			}

			// Remove the assistant message and all subsequent messages
			const messagesToRemove = this.activeChatMessages.slice(messageIndex);
			for (const message of messagesToRemove) {
				await DatabaseService.deleteMessage(message.id);
			}

			// Update local state to remove the messages
			this.activeChatMessages = this.activeChatMessages.slice(0, messageIndex);

			// Update chat message count
			const chatIndex = this.chats.findIndex((c) => c.id === this.activeChat!.id);
			if (chatIndex !== -1) {
				this.chats[chatIndex].messageCount -= messagesToRemove.length;
				this.chats[chatIndex].updatedAt = Date.now();
			}

			// Generate a new response based on the conversation history
			this.isLoading = true;
			this.currentResponse = '';

			try {
				const allMessages = await DatabaseService.getChatMessages(this.activeChat.id);

				const assistantMessage = await this.addMessage('assistant', '');
				if (!assistantMessage) {
					throw new Error('Failed to create assistant message');
				}

				let streamedContent = '';
				await this.chatService.sendChatCompletion(
					allMessages,
					{
						stream: true,
						temperature: 0.7,
						max_tokens: 2048,
						onChunk: (chunk: string) => {
							streamedContent += chunk;
							this.currentResponse = streamedContent;

							// Parse thinking content during streaming
							const partialThinking = extractPartialThinking(streamedContent);
							
							const messageIndex = this.activeChatMessages.findIndex(
								(m) => m.id === assistantMessage.id
							);
							if (messageIndex !== -1) {
								// Update message with parsed content
								this.activeChatMessages[messageIndex].content = partialThinking.remainingContent || streamedContent;
								// Update thinking content if present
								if (partialThinking.thinking) {
									this.activeChatMessages[messageIndex].thinking = partialThinking.thinking;
								}
							}
						},
						onComplete: async () => {
							// Update assistant message in database
							await DatabaseService.updateMessage(assistantMessage.id, {
								content: streamedContent
							});

							this.isLoading = false;
							this.currentResponse = '';
						},
						onError: (error: Error) => {
							// Don't log or show error if it's an AbortError (user stopped generation)
							if (error.name === 'AbortError' || error instanceof DOMException) {
								this.isLoading = false;
								this.currentResponse = '';
								return;
							}

							console.error('Streaming error:', error);
							this.isLoading = false;
							this.currentResponse = '';

							const assistantMessageIndex = this.activeChatMessages.findIndex(
								(m) => m.id === assistantMessage.id
							);
							if (assistantMessageIndex !== -1) {
								this.activeChatMessages[assistantMessageIndex].content = `Error: ${error.message}`;
							}
						}
					}
				);
			} catch (regenerateError) {
				console.error('Failed to regenerate response:', regenerateError);
				this.isLoading = false;
			}
		} catch (error) {
			// Don't log or show error if it's an AbortError (user stopped generation)
			if (error instanceof Error && (error.name === 'AbortError' || error instanceof DOMException)) {
				return;
			}

			console.error('Failed to regenerate message:', error);
		}
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
export const updateMessage = chatStore.updateMessage.bind(chatStore);
export const regenerateMessage = chatStore.regenerateMessage.bind(chatStore);
export const updateChatName = chatStore.updateChatName.bind(chatStore);
export const deleteChat = chatStore.deleteChat.bind(chatStore);
export const clearActiveChat = chatStore.clearActiveChat.bind(chatStore);

export function stopGeneration() {
	chatStore.stopGeneration();
}
export const chatMessages = () => chatStore.activeChatMessages;
