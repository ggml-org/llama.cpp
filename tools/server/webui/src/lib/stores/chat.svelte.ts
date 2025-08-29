import { DatabaseStore } from '$lib/stores/database';
import { chatService, slotsService } from '$lib/services';
import { serverStore } from '$lib/stores/server.svelte';
import type {
	DatabaseConversation,
	DatabaseMessage,
	DatabaseMessageExtra
} from '$lib/types/database';
import { config } from '$lib/stores/settings.svelte';
import { filterByLeafNodeId, findLeafNode, findDescendantMessages } from '$lib/utils/branching';
import { browser } from '$app/environment';
import { goto } from '$app/navigation';
import { extractPartialThinking } from '$lib/utils/thinking';

/**
 * ChatStore - Central state management for chat conversations and AI interactions
 *
 * This store manages the complete chat experience including:
 * - Conversation lifecycle (create, load, delete, update)
 * - Message management with branching support for conversation trees
 * - Real-time AI response streaming with reasoning content support
 * - File attachment handling and processing
 * - Context error management and recovery
 * - Database persistence through DatabaseStore integration
 *
 * **Architecture & Relationships:**
 * - **ChatService**: Handles low-level API communication with AI models
 *   - ChatStore orchestrates ChatService for streaming responses
 *   - ChatService provides abort capabilities and error handling
 *   - ChatStore manages the UI state while ChatService handles network layer
 *
 * - **DatabaseStore**: Provides persistent storage for conversations and messages
 *   - ChatStore uses DatabaseStore for all CRUD operations
 *   - Maintains referential integrity for conversation trees
 *   - Handles message branching and parent-child relationships
 *
 * - **SlotsService**: Monitors server resource usage during AI generation
 *   - ChatStore coordinates slots polling during streaming
 *   - Provides real-time feedback on server capacity
 *
 * **Key Features:**
 * - Reactive state management using Svelte 5 runes ($state)
 * - Conversation branching for exploring different response paths
 * - Streaming AI responses with real-time content updates
 * - File attachment support (images, PDFs, text files, audio)
 * - Context window management with error recovery
 * - Partial response saving when generation is interrupted
 * - Message editing with automatic response regeneration
 */
class ChatStore {
	activeConversation = $state<DatabaseConversation | null>(null);
	activeMessages = $state<DatabaseMessage[]>([]);
	conversations = $state<DatabaseConversation[]>([]);
	currentResponse = $state('');
	isInitialized = $state(false);
	isLoading = $state(false);
	maxContextError = $state<{ message: string; estimatedTokens: number; maxContext: number } | null>(
		null
	);

	constructor() {
		if (browser) {
			this.initialize();
		}
	}

	/**
	 * Initializes the chat store by loading conversations from the database
	 */
	async initialize() {
		try {
			await this.loadConversations();

			this.maxContextError = null;

			this.isInitialized = true;
		} catch (error) {
			console.error('Failed to initialize chat store:', error);
		}
	}

	/**
	 * Loads all conversations from the database
	 */
	async loadConversations() {
		this.conversations = await DatabaseStore.getAllConversations();
	}

	/**
	 * Creates a new conversation and navigates to it
	 * @param name - Optional name for the conversation, defaults to timestamped name
	 * @returns The ID of the created conversation
	 */
	async createConversation(name?: string): Promise<string> {
		const conversationName = name || `Chat ${new Date().toLocaleString()}`;
		const conversation = await DatabaseStore.createConversation(conversationName);

		this.conversations.unshift(conversation);

		this.activeConversation = conversation;
		this.activeMessages = [];

		this.maxContextError = null;

		await goto(`/chat/${conversation.id}`);

		return conversation.id;
	}

	/**
	 * Loads a specific conversation and its messages
	 * @param convId - The conversation ID to load
	 * @returns True if conversation was loaded successfully, false otherwise
	 */
	async loadConversation(convId: string): Promise<boolean> {
		try {
			const conversation = await DatabaseStore.getConversation(convId);

			if (!conversation) {
				return false;
			}

			this.activeConversation = conversation;

			if (conversation.currNode) {
				const allMessages = await DatabaseStore.getConversationMessages(convId);
				this.activeMessages = filterByLeafNodeId(
					allMessages,
					conversation.currNode,
					false
				) as DatabaseMessage[];
			} else {
				// Load all messages for conversations without currNode (backward compatibility)
				this.activeMessages = await DatabaseStore.getConversationMessages(convId);
			}

			this.maxContextError = null;

			return true;
		} catch (error) {
			console.error('Failed to load conversation:', error);

			return false;
		}
	}

	/**
	 * Adds a new message to the active conversation
	 * @param role - The role of the message sender (user/assistant)
	 * @param content - The message content
	 * @param type - The message type, defaults to 'text'
	 * @param parent - Parent message ID, defaults to '-1' for auto-detection
	 * @param extras - Optional extra data (files, attachments, etc.)
	 * @returns The created message or null if failed
	 */
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
			let parentId: string | null = null;

			if (parent === '-1') {
				if (this.activeMessages.length > 0) {
					parentId = this.activeMessages[this.activeMessages.length - 1].id;
				} else {
					const allMessages = await DatabaseStore.getConversationMessages(
						this.activeConversation.id
					);
					const rootMessage = allMessages.find((m) => m.parent === null && m.type === 'root');

					if (!rootMessage) {
						const rootId = await DatabaseStore.createRootMessage(this.activeConversation.id);
						parentId = rootId;
					} else {
						parentId = rootMessage.id;
					}
				}
			} else {
				parentId = parent;
			}

			const message = await DatabaseStore.createMessageBranch(
				{
					convId: this.activeConversation.id,
					role,
					content,
					type,
					timestamp: Date.now(),
					thinking: '',
					children: [],
					extra: extras
				},
				parentId
			);

			this.activeMessages.push(message);

			await DatabaseStore.updateCurrentNode(this.activeConversation.id, message.id);
			this.activeConversation.currNode = message.id;

			this.updateConversationTimestamp();

			return message;
		} catch (error) {
			console.error('Failed to add message:', error);
			return null;
		}
	}

	/**
	 * Gets API options from current configuration settings
	 * @returns API options object for chat completion requests
	 */
	private getApiOptions() {
		const currentConfig = config();
		return {
			stream: true,
			temperature: Number(currentConfig.temperature) || 0.8,
			max_tokens: Number(currentConfig.max_tokens) || 2048,
			timings_per_token: currentConfig.showTokensPerSecond || false,
			dynatemp_range: Number(currentConfig.dynatemp_range) || 0.0,
			dynatemp_exponent: Number(currentConfig.dynatemp_exponent) || 1.0,
			top_k: Number(currentConfig.top_k) || 40,
			top_p: Number(currentConfig.top_p) || 0.95,
			min_p: Number(currentConfig.min_p) || 0.05,
			xtc_probability: Number(currentConfig.xtc_probability) || 0.0,
			xtc_threshold: Number(currentConfig.xtc_threshold) || 0.1,
			typical_p: Number(currentConfig.typical_p) || 1.0,
			repeat_last_n: Number(currentConfig.repeat_last_n) || 64,
			repeat_penalty: Number(currentConfig.repeat_penalty) || 1.0,
			presence_penalty: Number(currentConfig.presence_penalty) || 0.0,
			frequency_penalty: Number(currentConfig.frequency_penalty) || 0.0,
			dry_multiplier: Number(currentConfig.dry_multiplier) || 0.0,
			dry_base: Number(currentConfig.dry_base) || 1.75,
			dry_allowed_length: Number(currentConfig.dry_allowed_length) || 2,
			dry_penalty_last_n: Number(currentConfig.dry_penalty_last_n) || -1,
			samplers: currentConfig.samplers || 'top_k;tfs_z;typical_p;top_p;min_p;temperature',
			custom: currentConfig.custom || ''
		};
	}

	/**
	 * Handles streaming chat completion with the AI model
	 * @param allMessages - All messages in the conversation
	 * @param assistantMessage - The assistant message to stream content into
	 * @param onComplete - Optional callback when streaming completes
	 * @param onError - Optional callback when an error occurs
	 */
	private async streamChatCompletion(
		allMessages: DatabaseMessage[],
		assistantMessage: DatabaseMessage,
		onComplete?: (content: string) => Promise<void>,
		onError?: (error: Error) => void
	): Promise<void> {
		let streamedContent = '';

		let streamedReasoningContent = '';

		// Start slots polling when streaming begins
		slotsService.startStreamingPolling();

		await chatService.sendMessage(allMessages, {
			...this.getApiOptions(),

			onChunk: (chunk: string) => {
				streamedContent += chunk;
				this.currentResponse = streamedContent;

				const partialThinking = extractPartialThinking(streamedContent);
				const messageIndex = this.findMessageIndex(assistantMessage.id);
				this.updateMessageAtIndex(messageIndex, {
					content: partialThinking.remainingContent || streamedContent
				});

				slotsService.updateSlotsState();
			},

			onReasoningChunk: (reasoningChunk: string) => {
				streamedReasoningContent += reasoningChunk;
				const messageIndex = this.findMessageIndex(assistantMessage.id);
				this.updateMessageAtIndex(messageIndex, { thinking: streamedReasoningContent });

				slotsService.updateSlotsState();
			},

			onComplete: async (finalContent?: string, reasoningContent?: string) => {
				// Stop slots polling when streaming completes
				slotsService.stopStreamingPolling();

				// Update assistant message in database
				await DatabaseStore.updateMessage(assistantMessage.id, {
					content: finalContent || streamedContent,
					thinking: reasoningContent || streamedReasoningContent
				});

				// Update currNode to assistant message after streaming completes
				await DatabaseStore.updateCurrentNode(this.activeConversation!.id, assistantMessage.id);
				this.activeConversation!.currNode = assistantMessage.id;

				// Refresh active messages to ensure UI state matches database state
				await this.refreshActiveMessages();

				// Call custom completion handler if provided
				if (onComplete) {
					await onComplete(streamedContent);
				}

				this.isLoading = false;
				this.currentResponse = '';
			},

			onError: (error: Error) => {
				// Stop slots polling on any error
				slotsService.stopStreamingPolling();

				// Don't log or show error if it's an AbortError (user stopped generation)
				if (error.name === 'AbortError' || error instanceof DOMException) {
					this.isLoading = false;
					this.currentResponse = '';
					return;
				}

				// Handle context errors specially
				if (error.name === 'ContextError') {
					console.warn('Context error detected:', error.message);
					this.isLoading = false;
					this.currentResponse = '';

					// Remove the assistant message that was created but failed
					const messageIndex = this.activeMessages.findIndex(
						(m: DatabaseMessage) => m.id === assistantMessage.id
					);

					if (messageIndex !== -1) {
						// Remove from UI
						this.activeMessages.splice(messageIndex, 1);
						// Remove from database
						DatabaseStore.deleteMessage(assistantMessage.id).catch(console.error);
					}

					this.maxContextError = {
						message: error.message,
						estimatedTokens: 0, // Server-side error, we don't have client estimates
						maxContext: serverStore.serverProps?.default_generation_settings.n_ctx ?? 4096 // Use server's actual n_ctx, fallback to 4096
					};

					if (onError) {
						onError(error);
					}
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

				if (onError) {
					onError(error);
				}
			}
		});
	}

	/**
	 * Checks if an error is an abort error (user cancelled operation)
	 * @param error - The error to check
	 * @returns True if the error is an abort error
	 */
	private isAbortError(error: unknown): boolean {
		return error instanceof Error && (error.name === 'AbortError' || error instanceof DOMException);
	}

	/**
	 * Finds the index of a message in the active messages array
	 * @param messageId - The message ID to find
	 * @returns The index of the message, or -1 if not found
	 */
	private findMessageIndex(messageId: string): number {
		return this.activeMessages.findIndex((m) => m.id === messageId);
	}

	/**
	 * Updates a message at a specific index with partial data
	 * @param index - The index of the message to update
	 * @param updates - Partial message data to update
	 */
	private updateMessageAtIndex(index: number, updates: Partial<DatabaseMessage>): void {
		if (index !== -1) {
			Object.assign(this.activeMessages[index], updates);
		}
	}

	/**
	 * Creates a new assistant message in the database
	 * @param parentId - Optional parent message ID, defaults to '-1'
	 * @returns The created assistant message or null if failed
	 */
	private async createAssistantMessage(parentId?: string): Promise<DatabaseMessage | null> {
		if (!this.activeConversation) return null;

		return await DatabaseStore.createMessageBranch(
			{
				convId: this.activeConversation.id,
				type: 'text',
				role: 'assistant',
				content: '',
				timestamp: Date.now(),
				thinking: '',
				children: []
			},
			parentId || null
		);
	}

	/**
	 * Updates conversation lastModified timestamp and moves it to top of list
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

	/**
	 * Sends a new message and generates AI response
	 * @param content - The message content to send
	 * @param extras - Optional extra data (files, attachments, etc.)
	 */
	async sendMessage(content: string, extras?: DatabaseMessageExtra[]): Promise<void> {
		if ((!content.trim() && (!extras || extras.length === 0)) || this.isLoading) return;

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

		let userMessage: DatabaseMessage | null = null;

		try {
			userMessage = await this.addMessage('user', content, 'text', '-1', extras);

			if (!userMessage) {
				throw new Error('Failed to add user message');
			}

			// If this is a new conversation, update the title with the first user prompt
			if (isNewConversation && content) {
				const title = content.trim();
				await this.updateConversationName(this.activeConversation.id, title);
			}

			const allMessages = await DatabaseStore.getConversationMessages(this.activeConversation.id);
			const assistantMessage = await this.createAssistantMessage(userMessage.id);

			if (!assistantMessage) {
				throw new Error('Failed to create assistant message');
			}

			this.activeMessages.push(assistantMessage);
			// Don't update currNode until after streaming completes to maintain proper conversation path

			await this.streamChatCompletion(allMessages, assistantMessage, undefined, (error: Error) => {
				if (error.name === 'ContextError' && userMessage) {
					const userMessageIndex = this.findMessageIndex(userMessage.id);
					if (userMessageIndex !== -1) {
						this.activeMessages.splice(userMessageIndex, 1);
						DatabaseStore.deleteMessage(userMessage.id).catch(console.error);
					}
				}
			});
		} catch (error) {
			if (this.isAbortError(error)) {
				this.isLoading = false;
				return;
			}

			if (error instanceof Error && error.name === 'ContextError' && userMessage) {
				const userMessageIndex = this.findMessageIndex(userMessage.id);
				if (userMessageIndex !== -1) {
					this.activeMessages.splice(userMessageIndex, 1);
					DatabaseStore.deleteMessage(userMessage.id).catch(console.error);
				}
			}

			console.error('Failed to send message:', error);
			this.isLoading = false;
		}
	}

	/**
	 * Stops the current message generation
	 */
	stopGeneration() {
		slotsService.stopStreamingPolling();
		chatService.abort();
		this.savePartialResponseIfNeeded();
		this.isLoading = false;
		this.currentResponse = '';
	}

	/**
	 * Gracefully stops generation and saves partial response
	 */
	async gracefulStop(): Promise<void> {
		if (!this.isLoading) return;

		slotsService.stopStreamingPolling();
		chatService.abort();
		await this.savePartialResponseIfNeeded();
		this.isLoading = false;
		this.currentResponse = '';
	}

	/**
	 * Clears the max context error state
	 */
	clearMaxContextError(): void {
		this.maxContextError = null;
	}

	/**
	 * Sets the max context error state
	 * @param error - The context error details or null to clear
	 */
	setMaxContextError(
		error: { message: string; estimatedTokens: number; maxContext: number } | null
	): void {
		this.maxContextError = error;
	}

	/**
	 * Saves partial response if generation was interrupted
	 */
	private async savePartialResponseIfNeeded() {
		if (!this.currentResponse.trim() || !this.activeMessages.length) {
			return;
		}

		const lastMessage = this.activeMessages[this.activeMessages.length - 1];

		if (lastMessage && lastMessage.role === 'assistant') {
			try {
				const partialThinking = extractPartialThinking(this.currentResponse);

				const updateData: { content: string; thinking?: string } = {
					content: partialThinking.remainingContent || this.currentResponse
				};

				if (partialThinking.thinking) {
					updateData.thinking = partialThinking.thinking;
				}

				await DatabaseStore.updateMessage(lastMessage.id, updateData);

				lastMessage.content = partialThinking.remainingContent || this.currentResponse;
			} catch (error) {
				lastMessage.content = this.currentResponse;
				console.error('Failed to save partial response:', error);
			}
		} else {
			console.error('Last message is not an assistant message');
		}
	}

	/**
	 * Updates a user message and regenerates the assistant response
	 * @param messageId - The ID of the message to update
	 * @param newContent - The new content for the message
	 */
	async updateMessage(messageId: string, newContent: string): Promise<void> {
		if (!this.activeConversation) return;

		if (this.isLoading) {
			this.stopGeneration();
		}

		try {
			const messageIndex = this.findMessageIndex(messageId);
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

			this.updateMessageAtIndex(messageIndex, { content: newContent });
			await DatabaseStore.updateMessage(messageId, { content: newContent });

			const messagesToRemove = this.activeMessages.slice(messageIndex + 1);
			for (const message of messagesToRemove) {
				await DatabaseStore.deleteMessage(message.id);
			}

			this.activeMessages = this.activeMessages.slice(0, messageIndex + 1);
			this.updateConversationTimestamp();

			this.isLoading = true;
			this.currentResponse = '';

			try {
				const assistantMessage = await this.createAssistantMessage();
				if (!assistantMessage) {
					throw new Error('Failed to create assistant message');
				}

				this.activeMessages.push(assistantMessage);
				await DatabaseStore.updateCurrentNode(this.activeConversation.id, assistantMessage.id);
				this.activeConversation.currNode = assistantMessage.id;

				await this.streamChatCompletion(
					this.activeMessages.slice(0, -1),
					assistantMessage,
					undefined,
					() => {
						const editedMessageIndex = this.findMessageIndex(messageId);
						this.updateMessageAtIndex(editedMessageIndex, { content: originalContent });
					}
				);
			} catch (regenerateError) {
				console.error('Failed to regenerate response:', regenerateError);
				this.isLoading = false;

				const messageIndex = this.findMessageIndex(messageId);
				this.updateMessageAtIndex(messageIndex, { content: originalContent });
			}
		} catch (error) {
			if (this.isAbortError(error)) {
				return;
			}

			console.error('Failed to update message:', error);
		}
	}

	/**
	 * Regenerates an assistant message with a new response
	 * @param messageId - The ID of the assistant message to regenerate
	 */
	async regenerateMessage(messageId: string): Promise<void> {
		if (!this.activeConversation || this.isLoading) return;

		try {
			const messageIndex = this.findMessageIndex(messageId);
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
				await DatabaseStore.deleteMessage(message.id);
			}

			this.activeMessages = this.activeMessages.slice(0, messageIndex);
			this.updateConversationTimestamp();

			this.isLoading = true;
			this.currentResponse = '';

			try {
				const allMessages = await DatabaseStore.getConversationMessages(this.activeConversation.id);
				const assistantMessage = await this.createAssistantMessage();

				if (!assistantMessage) {
					throw new Error('Failed to create assistant message');
				}

				this.activeMessages.push(assistantMessage);
				await DatabaseStore.updateCurrentNode(this.activeConversation.id, assistantMessage.id);
				this.activeConversation.currNode = assistantMessage.id;

				await this.streamChatCompletion(allMessages, assistantMessage);
			} catch (regenerateError) {
				console.error('Failed to regenerate response:', regenerateError);
				this.isLoading = false;
			}
		} catch (error) {
			if (this.isAbortError(error)) return;
			console.error('Failed to regenerate message:', error);
		}
	}

	/**
	 * Updates the name of a conversation
	 * @param convId - The conversation ID to update
	 * @param name - The new name for the conversation
	 */
	async updateConversationName(convId: string, name: string): Promise<void> {
		try {
			await DatabaseStore.updateConversation(convId, { name });

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

	/**
	 * Deletes a conversation and all its messages
	 * @param convId - The conversation ID to delete
	 */
	async deleteConversation(convId: string): Promise<void> {
		try {
			await DatabaseStore.deleteConversation(convId);

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

	/**
	 * Gets information about what messages will be deleted when deleting a specific message
	 * @param messageId - The ID of the message to be deleted
	 * @returns Object with deletion info including count and types of messages
	 */
	async getDeletionInfo(messageId: string): Promise<{
		totalCount: number;
		userMessages: number;
		assistantMessages: number;
		messageTypes: string[];
	}> {
		if (!this.activeConversation) {
			return { totalCount: 0, userMessages: 0, assistantMessages: 0, messageTypes: [] };
		}

		const allMessages = await DatabaseStore.getConversationMessages(this.activeConversation.id);
		const descendants = findDescendantMessages(allMessages, messageId);
		const allToDelete = [messageId, ...descendants];

		const messagesToDelete = allMessages.filter((m) => allToDelete.includes(m.id));

		let userMessages = 0;
		let assistantMessages = 0;
		const messageTypes: string[] = [];

		for (const msg of messagesToDelete) {
			if (msg.role === 'user') {
				userMessages++;
				if (!messageTypes.includes('user message')) messageTypes.push('user message');
			} else if (msg.role === 'assistant') {
				assistantMessages++;
				if (!messageTypes.includes('assistant response')) messageTypes.push('assistant response');
			}
		}

		return {
			totalCount: allToDelete.length,
			userMessages,
			assistantMessages,
			messageTypes
		};
	}

	/**
	 * Deletes a message and all its descendants, updating conversation path if needed
	 * @param messageId - The ID of the message to delete
	 */
	async deleteMessage(messageId: string): Promise<void> {
		try {
			if (!this.activeConversation) return;

			// Get all messages to find siblings before deletion
			const allMessages = await DatabaseStore.getConversationMessages(this.activeConversation.id);
			const messageToDelete = allMessages.find((m) => m.id === messageId);

			if (!messageToDelete) {
				console.error('Message to delete not found');
				return;
			}

			// Check if the deleted message is in the current conversation path
			const currentPath = filterByLeafNodeId(
				allMessages,
				this.activeConversation.currNode || '',
				false
			);
			const isInCurrentPath = currentPath.some((m) => m.id === messageId);

			// If the deleted message is in the current path, we need to update currNode
			if (isInCurrentPath && messageToDelete.parent) {
				// Find all siblings (messages with same parent)
				const siblings = allMessages.filter(
					(m) => m.parent === messageToDelete.parent && m.id !== messageId
				);

				if (siblings.length > 0) {
					// Find the latest sibling (highest timestamp)
					const latestSibling = siblings.reduce((latest, sibling) =>
						sibling.timestamp > latest.timestamp ? sibling : latest
					);

					// Find the leaf node for this sibling branch to get the complete conversation path
					const leafNodeId = findLeafNode(allMessages, latestSibling.id);

					// Update conversation to use the leaf node of the latest remaining sibling
					await DatabaseStore.updateCurrentNode(this.activeConversation.id, leafNodeId);
					this.activeConversation.currNode = leafNodeId;
				} else {
					// No siblings left, navigate to parent if it exists
					if (messageToDelete.parent) {
						const parentLeafId = findLeafNode(allMessages, messageToDelete.parent);
						await DatabaseStore.updateCurrentNode(this.activeConversation.id, parentLeafId);
						this.activeConversation.currNode = parentLeafId;
					}
				}
			}

			// Use cascading deletion to remove the message and all its descendants
			await DatabaseStore.deleteMessageCascading(this.activeConversation.id, messageId);

			// Refresh active messages to show the updated branch
			await this.refreshActiveMessages();

			// Update conversation timestamp
			this.updateConversationTimestamp();
		} catch (error) {
			console.error('Failed to delete message:', error);
		}
	}

	/**
	 * Clears the active conversation and resets state
	 */
	clearActiveConversation() {
		this.activeConversation = null;
		this.activeMessages = [];
		this.currentResponse = '';
		this.isLoading = false;
		this.maxContextError = null;
	}

	/** Refreshes active messages based on currNode after branch navigation */
	async refreshActiveMessages(): Promise<void> {
		if (!this.activeConversation) return;

		const allMessages = await DatabaseStore.getConversationMessages(this.activeConversation.id);
		if (allMessages.length === 0) {
			this.activeMessages = [];
			return;
		}

		const leafNodeId =
			this.activeConversation.currNode ||
			allMessages.reduce((latest, msg) => (msg.timestamp > latest.timestamp ? msg : latest)).id;

		const currentPath = filterByLeafNodeId(allMessages, leafNodeId, false) as DatabaseMessage[];

		this.activeMessages.length = 0;
		this.activeMessages.push(...currentPath);
	}
	/**
	 * Navigates to a specific sibling branch by updating currNode and refreshing messages
	 * @param siblingId - The sibling message ID to navigate to
	 */
	async navigateToSibling(siblingId: string): Promise<void> {
		if (!this.activeConversation) return;

		await DatabaseStore.updateCurrentNode(this.activeConversation.id, siblingId);
		this.activeConversation.currNode = siblingId;
		await this.refreshActiveMessages();
	}
	/**
	 * Edits a message by creating a new branch with the edited content
	 * @param messageId - The ID of the message to edit
	 * @param newContent - The new content for the message
	 */
	async editMessageWithBranching(messageId: string, newContent: string): Promise<void> {
		if (!this.activeConversation || this.isLoading) return;

		try {
			const messageIndex = this.findMessageIndex(messageId);
			if (messageIndex === -1) {
				console.error('Message not found for editing');
				return;
			}

			const messageToEdit = this.activeMessages[messageIndex];
			if (messageToEdit.role !== 'user') {
				console.error('Only user messages can be edited');
				return;
			}

			let parentId = messageToEdit.parent;

			if (parentId === undefined || parentId === null) {
				const allMessages = await DatabaseStore.getConversationMessages(this.activeConversation.id);
				const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);
				if (rootMessage) {
					parentId = rootMessage.id;
				} else {
					console.error('No root message found for editing');
					return;
				}
			}

			const newMessage = await DatabaseStore.createMessageBranch(
				{
					convId: messageToEdit.convId,
					type: messageToEdit.type,
					timestamp: Date.now(),
					role: messageToEdit.role,
					content: newContent,
					thinking: messageToEdit.thinking || '',
					children: [],
					extra: messageToEdit.extra ? JSON.parse(JSON.stringify(messageToEdit.extra)) : undefined
				},
				parentId
			);

			await DatabaseStore.updateCurrentNode(this.activeConversation.id, newMessage.id);
			this.activeConversation.currNode = newMessage.id;
			this.updateConversationTimestamp();
			await this.refreshActiveMessages();

			if (messageToEdit.role === 'user') {
				await this.generateResponseForMessage(newMessage.id);
			}
		} catch (error) {
			console.error('Failed to edit message with branching:', error);
		}
	}

	/**
	 * Regenerates an assistant message by creating a new branch with a new response
	 * @param messageId - The ID of the assistant message to regenerate
	 */
	async regenerateMessageWithBranching(messageId: string): Promise<void> {
		if (!this.activeConversation || this.isLoading) return;

		try {
			const messageIndex = this.findMessageIndex(messageId);
			if (messageIndex === -1) {
				console.error('Message not found for regeneration');
				return;
			}

			const messageToRegenerate = this.activeMessages[messageIndex];
			if (messageToRegenerate.role !== 'assistant') {
				console.error('Only assistant messages can be regenerated');
				return;
			}

			// Find parent message in all conversation messages, not just active path
			const conversationMessages = await DatabaseStore.getConversationMessages(
				this.activeConversation.id
			);
			const parentMessage = conversationMessages.find((m) => m.id === messageToRegenerate.parent);
			if (!parentMessage) {
				console.error('Parent message not found for regeneration');
				return;
			}

			this.isLoading = true;
			this.currentResponse = '';

			const newAssistantMessage = await DatabaseStore.createMessageBranch(
				{
					convId: this.activeConversation.id,
					type: 'text',
					timestamp: Date.now(),
					role: 'assistant',
					content: '',
					thinking: '',
					children: []
				},
				parentMessage.id
			);

			await DatabaseStore.updateCurrentNode(this.activeConversation.id, newAssistantMessage.id);
			this.activeConversation.currNode = newAssistantMessage.id;
			this.updateConversationTimestamp();
			await this.refreshActiveMessages();

			const allConversationMessages = await DatabaseStore.getConversationMessages(
				this.activeConversation.id
			);
			const conversationPath = filterByLeafNodeId(
				allConversationMessages,
				parentMessage.id,
				false
			) as DatabaseMessage[];

			await this.streamChatCompletion(conversationPath, newAssistantMessage);
		} catch (error) {
			console.error('Failed to regenerate message with branching:', error);
			this.isLoading = false;
		}
	}

	/**
	 * Generates a new assistant response for a given user message
	 * @param userMessageId - ID of user message to respond to
	 */
	private async generateResponseForMessage(userMessageId: string): Promise<void> {
		if (!this.activeConversation) return;

		this.isLoading = true;
		this.currentResponse = '';

		try {
			// Get conversation path up to the user message
			const allMessages = await DatabaseStore.getConversationMessages(this.activeConversation.id);
			const conversationPath = filterByLeafNodeId(
				allMessages,
				userMessageId,
				false
			) as DatabaseMessage[];

			// Create new assistant message branch
			const assistantMessage = await DatabaseStore.createMessageBranch(
				{
					convId: this.activeConversation.id,
					type: 'text',
					timestamp: Date.now(),
					role: 'assistant',
					content: '',
					thinking: '',
					children: []
				},
				userMessageId
			);

			// Add assistant message to active messages immediately for UI reactivity
			this.activeMessages.push(assistantMessage);

			// Stream response to new assistant message
			await this.streamChatCompletion(conversationPath, assistantMessage);
		} catch (error) {
			console.error('Failed to generate response:', error);
			this.isLoading = false;
		}
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
export const deleteConversation = chatStore.deleteConversation.bind(chatStore);
export const sendMessage = chatStore.sendMessage.bind(chatStore);
export const gracefulStop = chatStore.gracefulStop.bind(chatStore);
export const clearMaxContextError = chatStore.clearMaxContextError.bind(chatStore);
export const setMaxContextError = chatStore.setMaxContextError.bind(chatStore);

// Branching operations
export const refreshActiveMessages = chatStore.refreshActiveMessages.bind(chatStore);
export const navigateToSibling = chatStore.navigateToSibling.bind(chatStore);
export const editMessageWithBranching = chatStore.editMessageWithBranching.bind(chatStore);
export const regenerateMessageWithBranching =
	chatStore.regenerateMessageWithBranching.bind(chatStore);
export const deleteMessage = chatStore.deleteMessage.bind(chatStore);
export const getDeletionInfo = chatStore.getDeletionInfo.bind(chatStore);
export const updateConversationName = chatStore.updateConversationName.bind(chatStore);

export function stopGeneration() {
	chatStore.stopGeneration();
}
export const messages = () => chatStore.activeMessages;
