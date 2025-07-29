import { ChatService } from '$lib/services/chat';
import { DatabaseService } from '$lib/services';
import type { Conversation, Message } from '$lib/types/database';
import { goto } from '$app/navigation';
import { browser } from '$app/environment';
import { extractPartialThinking } from '$lib/utils/thinking';

class ChatStore {
	activeConversation = $state<Conversation | null>(null);
	activeMessages = $state<Message[]>([]);
	conversations = $state<Conversation[]>([]);
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
			await this.loadConversations();
			this.isInitialized = true;
		} catch (error) {
			console.error('Failed to initialize chat store:', error);
		}
	}

	async loadConversations() {
		this.conversations = await DatabaseService.getAllConversations();
	}

	async createConversation(name?: string): Promise<string> {
		const conversationName = name || `Chat ${new Date().toLocaleString()}`;

		const conversation = await DatabaseService.createConversation(conversationName);

		this.conversations.unshift(conversation);

		this.activeConversation = conversation;
		this.activeMessages = [];

		await goto(`/chat/${conversation.id}`);

		return conversation.id;
	}

	async loadConversation(convId: string): Promise<boolean> {
		try {
			const conversation = await DatabaseService.getConversation(convId);
			if (!conversation) {
				return false;
			}

			this.activeConversation = conversation;
			this.activeMessages = await DatabaseService.getConversationMessages(convId);
			return true;
		} catch (error) {
			console.error('Failed to load conversation:', error);
			return false;
		}
	}

	async addMessage(
		role: 'user' | 'assistant' | 'system',
		content: string,
		type: 'root' | 'text' | 'think' = 'text',
		parent: string = '-1'
	): Promise<Message | null> {
		if (!this.activeConversation) {
			console.error('No active conversation when trying to add message');
			return null;
		}

		try {
			const message = await DatabaseService.addMessage({
				convId: this.activeConversation.id,
				role,
				content,
				type,
				timestamp: Date.now(),
				parent,
				children: []
			});

			this.activeMessages.push(message);

			// Update lastModified
			const convIndex = this.conversations.findIndex((c) => c.id === this.activeConversation!.id);
			if (convIndex !== -1) {
				this.conversations[convIndex].lastModified = Date.now();

				const updatedConv = this.conversations.splice(convIndex, 1)[0];
				this.conversations.unshift(updatedConv);
			}

			return message;
		} catch (error) {
			console.error('Failed to add message:', error);
			return null;
		}
	}

	async sendMessage(content: string): Promise<void> {
		if (!content.trim() || this.isLoading) return;

		if (!this.activeConversation) {
			await this.createConversation();
		}

		if (!this.activeConversation) {
			console.error('No active conversation available for sending message');
			return;
		}
		this.isLoading = true;
		this.currentResponse = '';

		try {
			const userMessage = await this.addMessage('user', content);
			if (!userMessage) {
				throw new Error('Failed to add user message');
			}

			const allMessages = await DatabaseService.getConversationMessages(this.activeConversation.id);

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
						
						const messageIndex = this.activeMessages.findIndex(
							(m) => m.id === assistantMessage.id
						);
						if (messageIndex !== -1) {
							// Update message with parsed content
							this.activeMessages[messageIndex].content = partialThinking.remainingContent || streamedContent;
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

						const messageIndex = this.activeMessages.findIndex(
							(m: Message) => m.id === assistantMessage.id
						);
						if (messageIndex !== -1) {
							this.activeMessages[messageIndex].content = `Error: ${error.message}`;
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
		this.savePartialResponseIfNeeded();
		this.isLoading = false;
		this.currentResponse = '';
	}

	/**
	 * Gracefully stop generation and save partial response
	 * This method handles both async and sync scenarios
	 */
	async gracefulStop(): Promise<void> {
		if (!this.isLoading) {
			return;
		}

		this.chatService.abort();
		await this.savePartialResponseIfNeeded();
		this.isLoading = false;
		this.currentResponse = '';
	}

	/**
	 * Save partial response to database if there's content to save
	 */
	private async savePartialResponseIfNeeded() {
		console.log('savePartialResponseIfNeeded called:', {
			currentResponse: this.currentResponse,
			activeMessagesLength: this.activeMessages.length
		});
		
		if (!this.currentResponse.trim() || !this.activeMessages.length) {
			console.log('Early return: no current response or no messages');
			return;
		}

		// Find the last assistant message that might be incomplete
		const lastMessage = this.activeMessages[this.activeMessages.length - 1];
		console.log('Last message:', {
			role: lastMessage?.role,
			content: lastMessage?.content,
			contentLength: lastMessage?.content?.length
		});
		
		// Check if the last message is an assistant message (regardless of current content)
		// During streaming, the message content is updated in real-time, so we need to save it
		if (lastMessage && lastMessage.role === 'assistant') {
			try {
				// Parse thinking content before saving
				const partialThinking = extractPartialThinking(this.currentResponse);
				
				console.log('Saving partial response:', {
					messageId: lastMessage.id,
					currentResponse: this.currentResponse,
					remainingContent: partialThinking.remainingContent,
					thinking: partialThinking.thinking
				});
				
				// Update the assistant message with the partial response
				const updateData: { content: string; thinking?: string } = {
					content: partialThinking.remainingContent || this.currentResponse
				};
				if (partialThinking.thinking) {
					updateData.thinking = partialThinking.thinking;
				}
				
				await DatabaseService.updateMessage(lastMessage.id, updateData);
				console.log('Successfully saved partial response to database');

				// Update the local state to reflect the saved content
				lastMessage.content = partialThinking.remainingContent || this.currentResponse;
				if (partialThinking.thinking) {
					// Note: thinking content is part of the message content now
				}
			} catch (error) {
				console.error('Failed to save partial response:', error);
				// In case of error, at least update the local state
				lastMessage.content = this.currentResponse;
			}
		} else {
			console.log('No assistant message found to save partial response to');
		}
	}

	async updateMessage(messageId: string, newContent: string): Promise<void> {
		if (!this.activeConversation || this.isLoading) return;

		try {
			// Find the message to update
			const messageIndex = this.activeMessages.findIndex((m: Message) => m.id === messageId);
			if (messageIndex === -1) {
				console.error('Message not found for update');
				return;
			}

			const messageToUpdate = this.activeMessages[messageIndex];
			const originalContent = messageToUpdate.content; // Store original content for rollback
			
			// Only allow updating user messages
			if (messageToUpdate.role !== 'user') {
				console.error('Only user messages can be edited');
				return;
			}

			// Update the message in local state immediately for UI responsiveness
			this.activeMessages[messageIndex].content = newContent;

			// Remove all messages after the updated message (including assistant responses)
			const messagesToRemove = this.activeMessages.slice(messageIndex + 1);
			for (const message of messagesToRemove) {
				await DatabaseService.deleteMessage(message.id);
			}

			// Update local state to remove subsequent messages
			this.activeMessages = this.activeMessages.slice(0, messageIndex + 1);

			// Update chat message count
			const chatIndex = this.conversations.findIndex((c) => c.id === this.activeConversation!.id);
			if (chatIndex !== -1) {
				this.conversations[chatIndex].messageCount -= messagesToRemove.length;
				this.conversations[chatIndex].updatedAt = Date.now();
			}

			// Regenerate response to the updated message
			// We need to generate a new assistant response based on the conversation history
			this.isLoading = true;
			this.currentResponse = '';

			try {
				const allMessages = await DatabaseService.getConversationMessages(this.activeConversation.id);

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
							
							const messageIndex = this.activeMessages.findIndex(
								(m: Message) => m.id === assistantMessage.id
							);
							if (messageIndex !== -1) {
								// Update message with parsed content
								this.activeMessages[messageIndex].content = partialThinking.remainingContent || streamedContent;
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
							const editedMessageIndex = this.activeMessages.findIndex(
								(m: Message) => m.id === messageId
							);
							if (editedMessageIndex !== -1) {
								this.activeMessages[editedMessageIndex].content = originalContent;
							}

							const assistantMessageIndex = this.activeMessages.findIndex(
								(m: Message) => m.id === assistantMessage.id
							);
							if (assistantMessageIndex !== -1) {
								this.activeMessages[assistantMessageIndex].content = `Error: ${error.message}`;
							}
						}
					}
				);
			} catch (regenerateError) {
				console.error('Failed to regenerate response:', regenerateError);
				this.isLoading = false;
				
				// Rollback the edited message to original content on regeneration error
				const messageIndex = this.activeMessages.findIndex(
					(m: Message) => m.id === messageId
				);
				if (messageIndex !== -1) {
					this.activeMessages[messageIndex].content = originalContent;
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
		if (!this.activeConversation || this.isLoading) return;

		try {
			// Find the assistant message to regenerate
			const messageIndex = this.activeMessages.findIndex((m: Message) => m.id === messageId);
			if (messageIndex === -1) {
				console.error('Message not found for regeneration');
				return;
			}

			const messageToRegenerate = this.activeMessages[messageIndex];
			
			// Only allow regenerating assistant messages
			if (messageToRegenerate.role !== 'assistant') {
				console.error('Only assistant messages can be regenerated');
				return;
			}

			// Remove the assistant message and all subsequent messages
			const messagesToRemove = this.activeMessages.slice(messageIndex);
			for (const message of messagesToRemove) {
				await DatabaseService.deleteMessage(message.id);
			}

			// Update local state to remove the messages
			this.activeMessages = this.activeMessages.slice(0, messageIndex);

			// Update chat message count
			const chatIndex = this.conversations.findIndex((c) => c.id === this.activeConversation!.id);
			if (chatIndex !== -1) {
				this.conversations[chatIndex].messageCount -= messagesToRemove.length;
				this.conversations[chatIndex].updatedAt = Date.now();
			}

			// Generate a new response based on the conversation history
			this.isLoading = true;
			this.currentResponse = '';

			try {
				const allMessages = await DatabaseService.getConversationMessages(this.activeConversation.id);

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
							
							const messageIndex = this.activeMessages.findIndex(
								(m: Message) => m.id === assistantMessage.id
							);
							if (messageIndex !== -1) {
								// Update message with parsed content
								this.activeMessages[messageIndex].content = partialThinking.remainingContent || streamedContent;
								// Update thinking content if present
								// Note: thinking content is now part of the message content
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

	async updateConversationName(convId: string, name: string): Promise<void> {
		try {
			await DatabaseService.updateConversation(convId, { name });

			const convIndex = this.conversations.findIndex((c) => c.id === convId);
			if (convIndex !== -1) {
				this.conversations[convIndex].name = name;
			}

			if (this.activeConversation?.id === convId) {
				this.activeConversation.name = name;
			}
		} catch (error) {
			console.error('Failed to update conversation name:', error);
		}
	}

	async deleteConversation(convId: string): Promise<void> {
		try {
			await DatabaseService.deleteConversation(convId);

			this.conversations = this.conversations.filter((c) => c.id !== convId);

			if (this.activeConversation?.id === convId) {
				this.activeConversation = null;
				this.activeMessages = [];
				await goto('/?new_chat=true');
			}
		} catch (error) {
			console.error('Failed to delete conversation:', error);
		}
	}

	clearActiveConversation() {
		this.activeConversation = null;
		this.activeMessages = [];
		this.currentResponse = '';
		this.isLoading = false;
	}
}

export const chatStore = new ChatStore();

export const conversations = () => chatStore.conversations;
export const activeConversation = () => chatStore.activeConversation;
export const activeMessages = () => chatStore.activeMessages;
export const isLoading = () => chatStore.isLoading;
export const currentResponse = () => chatStore.currentResponse;
export const isInitialized = () => chatStore.isInitialized;

export const createConversation = chatStore.createConversation.bind(chatStore);
export const loadConversation = chatStore.loadConversation.bind(chatStore);
export const sendMessage = chatStore.sendMessage.bind(chatStore);
export const updateMessage = chatStore.updateMessage.bind(chatStore);
export const regenerateMessage = chatStore.regenerateMessage.bind(chatStore);
export const updateConversationName = chatStore.updateConversationName.bind(chatStore);
export const deleteConversation = chatStore.deleteConversation.bind(chatStore);
export const clearActiveConversation = chatStore.clearActiveConversation.bind(chatStore);
export const gracefulStop = chatStore.gracefulStop.bind(chatStore);

export function stopGeneration() {
	chatStore.stopGeneration();
}
export const messages = () => chatStore.activeMessages;
