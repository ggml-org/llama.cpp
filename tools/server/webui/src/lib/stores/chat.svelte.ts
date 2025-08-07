import { ChatService } from '$lib/services/chat';
import { DatabaseService } from '$lib/services';
import { goto } from '$app/navigation';
import { browser } from '$app/environment';
import { extractPartialThinking } from '$lib/utils/thinking';
import { config } from '$lib/stores/settings.svelte';

class ChatStore {
	activeConversation = $state<DatabaseConversation | null>(null);
	activeMessages = $state<DatabaseMessage[]>([]);
	conversations = $state<DatabaseConversation[]>([]);
	currentResponse = $state('');
	isInitialized = $state(false);
	isLoading = $state(false);
	maxContextError = $state<{ message: string; estimatedTokens: number; maxAllowed: number; maxContext: number } | null>(null);
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
		role: ChatRole,
		content: string,
		type: ChatMessageType = 'text',
		parent: string = '-1',
		extras?: DatabaseMessageExtra[]
	): Promise<DatabaseMessage | null> {
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
				thinking: '',
				children: [],
				extra: extras
			});

			this.activeMessages.push(message);

			// Update conversation timestamp
			this.updateConversationTimestamp();

			return message;
		} catch (error) {
			console.error('Failed to add message:', error);

			return null;
		}
	}

	/**
	 * Private helper method to handle streaming chat completion
	 * Reduces code duplication across sendMessage, updateMessage, and regenerateMessage
	 */
	private async streamChatCompletion(
		allMessages: DatabaseMessage[],
		assistantMessage: DatabaseMessage,
		onComplete?: (content: string) => Promise<void>,
		onError?: (error: Error) => void
	): Promise<void> {
		let streamedContent = '';

		// Get current settings
		const currentConfig = config();
		
		// Build complete options from settings
		const apiOptions = {
			stream: true,
			// Generation parameters
			temperature: Number(currentConfig.temperature) || 0.8,
			max_tokens: Number(currentConfig.max_tokens) || 2048,
			// Sampling parameters
			dynatemp_range: Number(currentConfig.dynatemp_range) || 0.0,
			dynatemp_exponent: Number(currentConfig.dynatemp_exponent) || 1.0,
			top_k: Number(currentConfig.top_k) || 40,
			top_p: Number(currentConfig.top_p) || 0.95,
			min_p: Number(currentConfig.min_p) || 0.05,
			xtc_probability: Number(currentConfig.xtc_probability) || 0.0,
			xtc_threshold: Number(currentConfig.xtc_threshold) || 0.1,
			typical_p: Number(currentConfig.typical_p) || 1.0,
			// Penalty parameters
			repeat_last_n: Number(currentConfig.repeat_last_n) || 64,
			repeat_penalty: Number(currentConfig.repeat_penalty) || 1.0,
			presence_penalty: Number(currentConfig.presence_penalty) || 0.0,
			frequency_penalty: Number(currentConfig.frequency_penalty) || 0.0,
			dry_multiplier: Number(currentConfig.dry_multiplier) || 0.0,
			dry_base: Number(currentConfig.dry_base) || 1.75,
			dry_allowed_length: Number(currentConfig.dry_allowed_length) || 2,
			dry_penalty_last_n: Number(currentConfig.dry_penalty_last_n) || -1,
			// Sampler configuration
			samplers: currentConfig.samplers || 'top_k;tfs_z;typical_p;top_p;min_p;temperature',
			// Custom parameters
			custom: currentConfig.custom || '',
		};
	
		await this.chatService.sendChatCompletion(
			allMessages,
			{
				...apiOptions,
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

					// Call custom completion handler if provided
					if (onComplete) {
						await onComplete(streamedContent);
					}

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
						(m: DatabaseMessage) => m.id === assistantMessage.id
					);

					if (messageIndex !== -1) {
						this.activeMessages[messageIndex].content = `Error: ${error.message}`;
					}

					// Call custom error handler if provided
					if (onError) {
						onError(error);
					}
				}
			}
		);
	}

	/**
	 * Private helper to handle abort errors consistently
	 */
	private isAbortError(error: unknown): boolean {
		return error instanceof Error && (error.name === 'AbortError' || error instanceof DOMException);
	}

	/**
	 * Private helper to update conversation lastModified timestamp and move to top
	 */
	private updateConversationTimestamp(): void {
		if (!this.activeConversation) return;

		const chatIndex = this.conversations.findIndex((c) => c.id === this.activeConversation!.id);

		if (chatIndex !== -1) {
			this.conversations[chatIndex].lastModified = Date.now();
			const updatedConv = this.conversations.splice(chatIndex, 1)[0];
			this.conversations.unshift(updatedConv);
		}
	}

	async sendMessage(content: string, extras?: DatabaseMessageExtra[]): Promise<void> {
		if (!content.trim() || this.isLoading) return;

		let isNewConversation = false;

		if (!this.activeConversation) {
			await this.createConversation();
			isNewConversation = true;
		}

		if (!this.activeConversation) {
			console.error('No active conversation available for sending message');
			return;
		}

		this.isLoading = true;
		this.currentResponse = '';

		try {
			const userMessage = await this.addMessage('user', content, 'text', '-1', extras);

			if (!userMessage) {
				throw new Error('Failed to add user message');
			}

			// If this is a new conversation, update the title with the first user prompt
			if (isNewConversation && content) {
				const title = content.trim();
				await this.updateConversationName(this.activeConversation.id, title);
			}

			const allMessages = await DatabaseService.getConversationMessages(this.activeConversation.id);
			const assistantMessage = await this.addMessage('assistant', '');

			if (!assistantMessage) {
				throw new Error('Failed to create assistant message');
			}

			await this.streamChatCompletion(allMessages, assistantMessage);
		} catch (error) {
			if (this.isAbortError(error)) {
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
	 * Clear context error state
	 */
	clearMaxContextError(): void {
		this.maxContextError = null;
	}

	// Allow external modules to set context error without importing heavy utils here
	setMaxContextError(error: { message: string; estimatedTokens: number; maxAllowed: number; maxContext: number } | null): void {
		this.maxContextError = error;
	}

	private async savePartialResponseIfNeeded() {		
		if (!this.currentResponse.trim() || !this.activeMessages.length) {
			return;
		}

		const lastMessage = this.activeMessages[this.activeMessages.length - 1];
		
		if (lastMessage && lastMessage.role === 'assistant') {
			try {
				const partialThinking = extractPartialThinking(this.currentResponse);
				
				console.log('Saving partial response:', {
					messageId: lastMessage.id,
					currentResponse: this.currentResponse,
					remainingContent: partialThinking.remainingContent,
					thinking: partialThinking.thinking
				});
				
				const updateData: { content: string; thinking?: string } = {
					content: partialThinking.remainingContent || this.currentResponse
				};

				if (partialThinking.thinking) {
					updateData.thinking = partialThinking.thinking;
				}
				
				await DatabaseService.updateMessage(lastMessage.id, updateData);
				console.log('Successfully saved partial response to database');

				lastMessage.content = partialThinking.remainingContent || this.currentResponse;
			} catch (error) {
				console.error('Failed to save partial response:', error);

				lastMessage.content = this.currentResponse;
			}
		} else {
			console.log('No assistant message found to save partial response to');
		}
	}

	async updateMessage(messageId: string, newContent: string): Promise<void> {
		if (!this.activeConversation) return;
		
		// If currently loading, gracefully abort the ongoing generation
		if (this.isLoading) {
			this.stopGeneration();
		}

		try {
			const messageIndex = this.activeMessages.findIndex((m: DatabaseMessage) => m.id === messageId);

			if (messageIndex === -1) {
				console.error('Message not found for update');
				return;
			}

			const messageToUpdate = this.activeMessages[messageIndex];
			const originalContent = messageToUpdate.content;
			
			if (messageToUpdate.role !== 'user') {
				console.error('Only user messages can be edited');
				return;
			}

			this.activeMessages[messageIndex].content = newContent;

			// Update the message in database immediately to ensure consistency
			// This prevents issues with rapid consecutive edits during regeneration
			await DatabaseService.updateMessage(messageId, { content: newContent });

			const messagesToRemove = this.activeMessages.slice(messageIndex + 1);
			for (const message of messagesToRemove) {
				await DatabaseService.deleteMessage(message.id);
			}

			this.activeMessages = this.activeMessages.slice(0, messageIndex + 1);

			// Update conversation timestamp
			this.updateConversationTimestamp();

			this.isLoading = true;
			this.currentResponse = '';

			try {
				// Use current in-memory messages which contain the updated content
				// instead of fetching from database which still has the old content
				const assistantMessage = await this.addMessage('assistant', '');

				if (!assistantMessage) {
					throw new Error('Failed to create assistant message');
				}

				await this.streamChatCompletion(
					this.activeMessages.slice(0, -1), // Exclude the just-added empty assistant message
					assistantMessage,
					undefined,
					(error: Error) => {
						// Restore original content on error
						const editedMessageIndex = this.activeMessages.findIndex(
							(m: DatabaseMessage) => m.id === messageId
						);
						if (editedMessageIndex !== -1) {
							this.activeMessages[editedMessageIndex].content = originalContent;
						}
					}
				);
			} catch (regenerateError) {
				console.error('Failed to regenerate response:', regenerateError);
				this.isLoading = false;
				
				const messageIndex = this.activeMessages.findIndex(
					(m: DatabaseMessage) => m.id === messageId
				);
				if (messageIndex !== -1) {
					this.activeMessages[messageIndex].content = originalContent;
				}
			}
		} catch (error) {
			if (this.isAbortError(error)) {
				return;
			}

			console.error('Failed to update message:', error);
		}
	}

	async regenerateMessage(messageId: string): Promise<void> {
		if (!this.activeConversation || this.isLoading) return;

		try {
			const messageIndex = this.activeMessages.findIndex((m: DatabaseMessage) => m.id === messageId);
			if (messageIndex === -1) {
				console.error('Message not found for regeneration');
				return;
			}

			const messageToRegenerate = this.activeMessages[messageIndex];
			
			if (messageToRegenerate.role !== 'assistant') {
				console.error('Only assistant messages can be regenerated');
				return;
			}

			const messagesToRemove = this.activeMessages.slice(messageIndex);

			for (const message of messagesToRemove) {
				await DatabaseService.deleteMessage(message.id);
			}

			this.activeMessages = this.activeMessages.slice(0, messageIndex);

			// Update conversation timestamp
			this.updateConversationTimestamp();

			this.isLoading = true;
			this.currentResponse = '';

			try {
				const allMessages = await DatabaseService.getConversationMessages(this.activeConversation.id);
				const assistantMessage = await this.addMessage('assistant', '');
				
				if (!assistantMessage) {
					throw new Error('Failed to create assistant message');
				}

				await this.streamChatCompletion(allMessages, assistantMessage);
			} catch (regenerateError) {
				console.error('Failed to regenerate response:', regenerateError);
				this.isLoading = false;
			}
		} catch (error) {
			if (this.isAbortError(error)) {
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
export const maxContextError = () => chatStore.maxContextError;

export const createConversation = chatStore.createConversation.bind(chatStore);
export const loadConversation = chatStore.loadConversation.bind(chatStore);
export const sendMessage = chatStore.sendMessage.bind(chatStore);
export const updateMessage = chatStore.updateMessage.bind(chatStore);
export const regenerateMessage = chatStore.regenerateMessage.bind(chatStore);
export const updateConversationName = chatStore.updateConversationName.bind(chatStore);
export const deleteConversation = chatStore.deleteConversation.bind(chatStore);
export const clearActiveConversation = chatStore.clearActiveConversation.bind(chatStore);
export const gracefulStop = chatStore.gracefulStop.bind(chatStore);
export const clearMaxContextError = chatStore.clearMaxContextError.bind(chatStore);
export const setMaxContextError = chatStore.setMaxContextError.bind(chatStore);

export function stopGeneration() {
	chatStore.stopGeneration();
}
export const messages = () => chatStore.activeMessages;
