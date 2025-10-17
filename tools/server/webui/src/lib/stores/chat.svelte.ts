import { DatabaseStore } from '$lib/stores/database';
import { chatService, slotsService } from '$lib/services';
import { serverStore } from '$lib/stores/server.svelte';
import { config } from '$lib/stores/settings.svelte';
import { filterByLeafNodeId, findLeafNode, findDescendantMessages } from '$lib/utils/branching';
import { browser } from '$app/environment';
import { goto } from '$app/navigation';
import { toast } from 'svelte-sonner';
import type { ExportedConversations } from '$lib/types/database';

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
 * - Partial response saving when generation is interrupted
 * - Message editing with automatic response regeneration
 */
class ChatStore {
	activeConversation = $state<DatabaseConversation | null>(null);
	activeMessages = $state<DatabaseMessage[]>([]);
	conversations = $state<DatabaseConversation[]>([]);
	currentResponse = $state('');
	errorDialogState = $state<{ type: 'timeout' | 'server'; message: string } | null>(null);
	isInitialized = $state(false);
	isLoading = $state(false);
	titleUpdateConfirmationCallback?: (currentTitle: string, newTitle: string) => Promise<boolean>;

	constructor() {
		if (browser) {
			this.initialize();
		}
	}

	/**
	 * Initializes the chat store by loading conversations from the database
	 * Sets up the initial state and loads existing conversations
	 */
	async initialize(): Promise<void> {
		try {
			await this.loadConversations();

			this.isInitialized = true;
		} catch (error) {
			console.error('Failed to initialize chat store:', error);
		}
	}

	/**
	 * Loads all conversations from the database
	 * Refreshes the conversations list from persistent storage
	 */
	async loadConversations(): Promise<void> {
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

		await goto(`#/chat/${conversation.id}`);

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
	 * Converts settings store values to API-compatible format
	 * @returns API options object for chat completion requests
	 */
	private getApiOptions(): Record<string, unknown> {
		const currentConfig = config();
		const hasValue = (value: unknown): boolean =>
			value !== undefined && value !== null && value !== '';

		const apiOptions: Record<string, unknown> = {
			stream: true,
			timings_per_token: true
		};

		if (hasValue(currentConfig.temperature)) {
			apiOptions.temperature = Number(currentConfig.temperature);
		}
		if (hasValue(currentConfig.max_tokens)) {
			apiOptions.max_tokens = Number(currentConfig.max_tokens);
		}
		if (hasValue(currentConfig.dynatemp_range)) {
			apiOptions.dynatemp_range = Number(currentConfig.dynatemp_range);
		}
		if (hasValue(currentConfig.dynatemp_exponent)) {
			apiOptions.dynatemp_exponent = Number(currentConfig.dynatemp_exponent);
		}
		if (hasValue(currentConfig.top_k)) {
			apiOptions.top_k = Number(currentConfig.top_k);
		}
		if (hasValue(currentConfig.top_p)) {
			apiOptions.top_p = Number(currentConfig.top_p);
		}
		if (hasValue(currentConfig.min_p)) {
			apiOptions.min_p = Number(currentConfig.min_p);
		}
		if (hasValue(currentConfig.xtc_probability)) {
			apiOptions.xtc_probability = Number(currentConfig.xtc_probability);
		}
		if (hasValue(currentConfig.xtc_threshold)) {
			apiOptions.xtc_threshold = Number(currentConfig.xtc_threshold);
		}
		if (hasValue(currentConfig.typ_p)) {
			apiOptions.typ_p = Number(currentConfig.typ_p);
		}
		if (hasValue(currentConfig.repeat_last_n)) {
			apiOptions.repeat_last_n = Number(currentConfig.repeat_last_n);
		}
		if (hasValue(currentConfig.repeat_penalty)) {
			apiOptions.repeat_penalty = Number(currentConfig.repeat_penalty);
		}
		if (hasValue(currentConfig.presence_penalty)) {
			apiOptions.presence_penalty = Number(currentConfig.presence_penalty);
		}
		if (hasValue(currentConfig.frequency_penalty)) {
			apiOptions.frequency_penalty = Number(currentConfig.frequency_penalty);
		}
		if (hasValue(currentConfig.dry_multiplier)) {
			apiOptions.dry_multiplier = Number(currentConfig.dry_multiplier);
		}
		if (hasValue(currentConfig.dry_base)) {
			apiOptions.dry_base = Number(currentConfig.dry_base);
		}
		if (hasValue(currentConfig.dry_allowed_length)) {
			apiOptions.dry_allowed_length = Number(currentConfig.dry_allowed_length);
		}
		if (hasValue(currentConfig.dry_penalty_last_n)) {
			apiOptions.dry_penalty_last_n = Number(currentConfig.dry_penalty_last_n);
		}
		if (currentConfig.samplers) {
			apiOptions.samplers = currentConfig.samplers;
		}
		if (currentConfig.custom) {
			apiOptions.custom = currentConfig.custom;
		}

		return apiOptions;
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
		let modelCaptured = false;

		const captureModelIfNeeded = (updateDbImmediately = true): string | undefined => {
			if (!modelCaptured) {
				const currentModelName = serverStore.modelName;

				if (currentModelName) {
					if (updateDbImmediately) {
						DatabaseStore.updateMessage(assistantMessage.id, { model: currentModelName }).catch(
							console.error
						);
					}

					const messageIndex = this.findMessageIndex(assistantMessage.id);

					this.updateMessageAtIndex(messageIndex, { model: currentModelName });
					modelCaptured = true;

					return currentModelName;
				}
			}
			return undefined;
		};

		slotsService.startStreaming();

		await chatService.sendMessage(allMessages, {
			...this.getApiOptions(),

			onChunk: (chunk: string) => {
				streamedContent += chunk;
				this.currentResponse = streamedContent;

				captureModelIfNeeded();
				const messageIndex = this.findMessageIndex(assistantMessage.id);
				this.updateMessageAtIndex(messageIndex, {
					content: streamedContent
				});
			},

			onReasoningChunk: (reasoningChunk: string) => {
				streamedReasoningContent += reasoningChunk;

				captureModelIfNeeded();

				const messageIndex = this.findMessageIndex(assistantMessage.id);

				this.updateMessageAtIndex(messageIndex, { thinking: streamedReasoningContent });
			},

			onComplete: async (
				finalContent?: string,
				reasoningContent?: string,
				timings?: ChatMessageTimings
			) => {
				slotsService.stopStreaming();

				const updateData: {
					content: string;
					thinking: string;
					timings?: ChatMessageTimings;
					model?: string;
				} = {
					content: finalContent || streamedContent,
					thinking: reasoningContent || streamedReasoningContent,
					timings: timings
				};

				const capturedModel = captureModelIfNeeded(false);

				if (capturedModel) {
					updateData.model = capturedModel;
				}

				await DatabaseStore.updateMessage(assistantMessage.id, updateData);

				const messageIndex = this.findMessageIndex(assistantMessage.id);

				const localUpdateData: { timings?: ChatMessageTimings; model?: string } = {
					timings: timings
				};

				if (updateData.model) {
					localUpdateData.model = updateData.model;
				}

				this.updateMessageAtIndex(messageIndex, localUpdateData);

				await DatabaseStore.updateCurrentNode(this.activeConversation!.id, assistantMessage.id);
				this.activeConversation!.currNode = assistantMessage.id;
				await this.refreshActiveMessages();

				if (onComplete) {
					await onComplete(streamedContent);
				}

				this.isLoading = false;
				this.currentResponse = '';
			},

			onError: (error: Error) => {
				slotsService.stopStreaming();

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
					const [failedMessage] = this.activeMessages.splice(messageIndex, 1);

					if (failedMessage) {
						DatabaseStore.deleteMessage(failedMessage.id).catch((cleanupError) => {
							console.error('Failed to remove assistant message after error:', cleanupError);
						});
					}
				}

				const dialogType = error.name === 'TimeoutError' ? 'timeout' : 'server';

				this.showErrorDialog(dialogType, error.message);

				if (onError) {
					onError(error);
				}
			}
		});
	}

	private showErrorDialog(type: 'timeout' | 'server', message: string): void {
		this.errorDialogState = { type, message };
	}

	dismissErrorDialog(): void {
		this.errorDialogState = null;
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
	 * Ensures recently active conversations appear first in the sidebar
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

		this.errorDialogState = null;
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

			const assistantMessage = await this.createAssistantMessage(userMessage.id);

			if (!assistantMessage) {
				throw new Error('Failed to create assistant message');
			}

			this.activeMessages.push(assistantMessage);
			// Don't update currNode until after streaming completes to maintain proper conversation path

			const conversationContext = this.activeMessages.slice(0, -1);

			await this.streamChatCompletion(conversationContext, assistantMessage);
		} catch (error) {
			if (this.isAbortError(error)) {
				this.isLoading = false;
				return;
			}

			console.error('Failed to send message:', error);
			this.isLoading = false;
			if (!this.errorDialogState) {
				if (error instanceof Error) {
					const dialogType = error.name === 'TimeoutError' ? 'timeout' : 'server';
					this.showErrorDialog(dialogType, error.message);
				} else {
					this.showErrorDialog('server', 'Unknown error occurred while sending message');
				}
			}
		}
	}

	/**
	 * Stops the current message generation
	 * Aborts ongoing requests and saves partial response if available
	 */
	stopGeneration(): void {
		slotsService.stopStreaming();
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

		slotsService.stopStreaming();
		chatService.abort();
		await this.savePartialResponseIfNeeded();
		this.isLoading = false;
		this.currentResponse = '';
	}

	/**
	 * Saves partial response if generation was interrupted
	 * Preserves user's partial content and timing data when generation is stopped early
	 */
	private async savePartialResponseIfNeeded(): Promise<void> {
		if (!this.currentResponse.trim() || !this.activeMessages.length) {
			return;
		}

		const lastMessage = this.activeMessages[this.activeMessages.length - 1];

		if (lastMessage && lastMessage.role === 'assistant') {
			try {
				const updateData: {
					content: string;
					thinking?: string;
					timings?: ChatMessageTimings;
				} = {
					content: this.currentResponse
				};

				if (lastMessage.thinking?.trim()) {
					updateData.thinking = lastMessage.thinking;
				}

				const lastKnownState = await slotsService.getCurrentState();

				if (lastKnownState) {
					updateData.timings = {
						prompt_n: lastKnownState.promptTokens || 0,
						predicted_n: lastKnownState.tokensDecoded || 0,
						cache_n: lastKnownState.cacheTokens || 0,
						// We don't have ms data from the state, but we can estimate
						predicted_ms:
							lastKnownState.tokensPerSecond && lastKnownState.tokensDecoded
								? (lastKnownState.tokensDecoded / lastKnownState.tokensPerSecond) * 1000
								: undefined
					};
				}

				await DatabaseStore.updateMessage(lastMessage.id, updateData);

				lastMessage.content = this.currentResponse;
				if (updateData.thinking !== undefined) {
					lastMessage.thinking = updateData.thinking;
				}
				if (updateData.timings) {
					lastMessage.timings = updateData.timings;
				}
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

			const allMessages = await DatabaseStore.getConversationMessages(this.activeConversation.id);
			const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);
			const isFirstUserMessage =
				rootMessage && messageToUpdate.parent === rootMessage.id && messageToUpdate.role === 'user';

			this.updateMessageAtIndex(messageIndex, { content: newContent });
			await DatabaseStore.updateMessage(messageId, { content: newContent });

			// If this is the first user message, update the conversation title with confirmation if needed
			if (isFirstUserMessage && newContent.trim()) {
				await this.updateConversationTitleWithConfirmation(
					this.activeConversation.id,
					newContent.trim(),
					this.titleUpdateConfirmationCallback
				);
			}

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
				const parentMessageId =
					this.activeMessages.length > 0
						? this.activeMessages[this.activeMessages.length - 1].id
						: null;

				const assistantMessage = await this.createAssistantMessage(parentMessageId);

				if (!assistantMessage) {
					throw new Error('Failed to create assistant message');
				}

				this.activeMessages.push(assistantMessage);

				const conversationContext = this.activeMessages.slice(0, -1);

				await this.streamChatCompletion(conversationContext, assistantMessage);
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
	 * Sets the callback function for title update confirmations
	 * @param callback - Function to call when confirmation is needed
	 */
	setTitleUpdateConfirmationCallback(
		callback: (currentTitle: string, newTitle: string) => Promise<boolean>
	): void {
		this.titleUpdateConfirmationCallback = callback;
	}

	/**
	 * Updates conversation title with optional confirmation dialog based on settings
	 * @param convId - The conversation ID to update
	 * @param newTitle - The new title content
	 * @param onConfirmationNeeded - Callback when user confirmation is needed
	 * @returns Promise<boolean> - True if title was updated, false if cancelled
	 */
	async updateConversationTitleWithConfirmation(
		convId: string,
		newTitle: string,
		onConfirmationNeeded?: (currentTitle: string, newTitle: string) => Promise<boolean>
	): Promise<boolean> {
		try {
			const currentConfig = config();

			// Only ask for confirmation if the setting is enabled and callback is provided
			if (currentConfig.askForTitleConfirmation && onConfirmationNeeded) {
				const conversation = await DatabaseStore.getConversation(convId);
				if (!conversation) return false;

				const shouldUpdate = await onConfirmationNeeded(conversation.name, newTitle);
				if (!shouldUpdate) return false;
			}

			await this.updateConversationName(convId, newTitle);
			return true;
		} catch (error) {
			console.error('Failed to update conversation title with confirmation:', error);
			return false;
		}
	}

	/**
	 * Downloads a conversation as JSON file
	 * @param convId - The conversation ID to download
	 */
	async downloadConversation(convId: string): Promise<void> {
		if (!this.activeConversation || this.activeConversation.id !== convId) {
			// Load the conversation if not currently active
			const conversation = await DatabaseStore.getConversation(convId);
			if (!conversation) return;

			const messages = await DatabaseStore.getConversationMessages(convId);
			const conversationData = {
				conv: conversation,
				messages
			};

			this.triggerDownload(conversationData);
		} else {
			// Use current active conversation data
			const conversationData: ExportedConversations = {
				conv: this.activeConversation!,
				messages: this.activeMessages
			};

			this.triggerDownload(conversationData);
		}
	}

	/**
	 * Triggers file download in browser
	 * @param data - Data to download (expected: { conv: DatabaseConversation, messages: DatabaseMessage[] })
	 * @param filename - Optional filename
	 */
	private triggerDownload(data: ExportedConversations, filename?: string): void {
		const conversation =
			'conv' in data ? data.conv : Array.isArray(data) ? data[0]?.conv : undefined;
		if (!conversation) {
			console.error('Invalid data: missing conversation');
			return;
		}
		const conversationName = conversation.name ? conversation.name.trim() : '';
		const convId = conversation.id || 'unknown';
		const truncatedSuffix = conversationName
			.toLowerCase()
			.replace(/[^a-z0-9]/gi, '_')
			.replace(/_+/g, '_')
			.substring(0, 20);
		const downloadFilename = filename || `conversation_${convId}_${truncatedSuffix}.json`;

		const conversationJson = JSON.stringify(data, null, 2);
		const blob = new Blob([conversationJson], {
			type: 'application/json'
		});
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = downloadFilename;
		document.body.appendChild(a);
		a.click();
		document.body.removeChild(a);
		URL.revokeObjectURL(url);
	}

	/**
	 * Exports all conversations with their messages as a JSON file
	 */
	async exportAllConversations(): Promise<void> {
		try {
			const allConversations = await DatabaseStore.getAllConversations();
			if (allConversations.length === 0) {
				throw new Error('No conversations to export');
			}

			const allData: ExportedConversations = await Promise.all(
				allConversations.map(async (conv) => {
					const messages = await DatabaseStore.getConversationMessages(conv.id);
					return { conv, messages };
				})
			);

			const blob = new Blob([JSON.stringify(allData, null, 2)], {
				type: 'application/json'
			});
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = `all_conversations_${new Date().toISOString().split('T')[0]}.json`;
			document.body.appendChild(a);
			a.click();
			document.body.removeChild(a);
			URL.revokeObjectURL(url);

			toast.success(`All conversations (${allConversations.length}) prepared for download`);
		} catch (err) {
			console.error('Failed to export conversations:', err);
			throw err;
		}
	}

	/**
	 * Imports conversations from a JSON file.
	 * Supports both single conversation (object) and multiple conversations (array).
	 * Uses DatabaseStore for safe, encapsulated data access
	 */
	async importConversations(): Promise<void> {
		return new Promise((resolve, reject) => {
			const input = document.createElement('input');
			input.type = 'file';
			input.accept = '.json';

			input.onchange = async (e) => {
				const file = (e.target as HTMLInputElement)?.files?.[0];
				if (!file) {
					reject(new Error('No file selected'));
					return;
				}

				try {
					const text = await file.text();
					const parsedData = JSON.parse(text);
					let importedData: ExportedConversations;

					if (Array.isArray(parsedData)) {
						importedData = parsedData;
					} else if (
						parsedData &&
						typeof parsedData === 'object' &&
						'conv' in parsedData &&
						'messages' in parsedData
					) {
						// Single conversation object
						importedData = [parsedData];
					} else {
						throw new Error(
							'Invalid file format: expected array of conversations or single conversation object'
						);
					}

					const result = await DatabaseStore.importConversations(importedData);

					// Refresh UI
					await this.loadConversations();

					toast.success(`Imported ${result.imported} conversation(s), skipped ${result.skipped}`);

					resolve(undefined);
				} catch (err: unknown) {
					const message = err instanceof Error ? err.message : 'Unknown error';
					console.error('Failed to import conversations:', err);
					toast.error('Import failed', {
						description: message
					});
					reject(new Error(`Import failed: ${message}`));
				}
			};

			input.click();
		});
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
				await goto(`?new_chat=true#/`);
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
	 * Used when navigating away from chat or starting fresh
	 */
	clearActiveConversation(): void {
		this.activeConversation = null;
		this.activeMessages = [];
		this.currentResponse = '';
		this.isLoading = false;
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

		// Get the current first user message before navigation
		const allMessages = await DatabaseStore.getConversationMessages(this.activeConversation.id);
		const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);
		const currentFirstUserMessage = this.activeMessages.find(
			(m) => m.role === 'user' && m.parent === rootMessage?.id
		);

		const currentLeafNodeId = findLeafNode(allMessages, siblingId);

		await DatabaseStore.updateCurrentNode(this.activeConversation.id, currentLeafNodeId);
		this.activeConversation.currNode = currentLeafNodeId;
		await this.refreshActiveMessages();

		// Only show title dialog if we're navigating between different first user message siblings
		if (rootMessage && this.activeMessages.length > 0) {
			// Find the first user message in the new active path
			const newFirstUserMessage = this.activeMessages.find(
				(m) => m.role === 'user' && m.parent === rootMessage.id
			);

			// Only show dialog if:
			// 1. We have a new first user message
			// 2. It's different from the previous one (different ID or content)
			// 3. The new message has content
			if (
				newFirstUserMessage &&
				newFirstUserMessage.content.trim() &&
				(!currentFirstUserMessage ||
					newFirstUserMessage.id !== currentFirstUserMessage.id ||
					newFirstUserMessage.content.trim() !== currentFirstUserMessage.content.trim())
			) {
				await this.updateConversationTitleWithConfirmation(
					this.activeConversation.id,
					newFirstUserMessage.content.trim(),
					this.titleUpdateConfirmationCallback
				);
			}
		}
	}

	/**
	 * Edits an assistant message with optional branching
	 * @param messageId - The ID of the assistant message to edit
	 * @param newContent - The new content for the message
	 * @param shouldBranch - Whether to create a branch or replace in-place
	 */
	async editAssistantMessage(
		messageId: string,
		newContent: string,
		shouldBranch: boolean
	): Promise<void> {
		if (!this.activeConversation || this.isLoading) return;

		try {
			const messageIndex = this.findMessageIndex(messageId);

			if (messageIndex === -1) {
				console.error('Message not found for editing');
				return;
			}

			const messageToEdit = this.activeMessages[messageIndex];

			if (messageToEdit.role !== 'assistant') {
				console.error('Only assistant messages can be edited with this method');
				return;
			}

			if (shouldBranch) {
				const newMessage = await DatabaseStore.createMessageBranch(
					{
						convId: messageToEdit.convId,
						type: messageToEdit.type,
						timestamp: Date.now(),
						role: messageToEdit.role,
						content: newContent,
						thinking: messageToEdit.thinking || '',
						children: [],
						model: messageToEdit.model // Preserve original model info when branching
					},
					messageToEdit.parent!
				);

				await DatabaseStore.updateCurrentNode(this.activeConversation.id, newMessage.id);
				this.activeConversation.currNode = newMessage.id;
			} else {
				await DatabaseStore.updateMessage(messageToEdit.id, {
					content: newContent,
					timestamp: Date.now()
				});

				this.updateMessageAtIndex(messageIndex, {
					content: newContent,
					timestamp: Date.now()
				});
			}

			this.updateConversationTimestamp();
			await this.refreshActiveMessages();
		} catch (error) {
			console.error('Failed to edit assistant message:', error);
		}
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

			// Check if this is the first user message in the conversation
			// First user message is one that has the root message as its parent
			const allMessages = await DatabaseStore.getConversationMessages(this.activeConversation.id);
			const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);
			const isFirstUserMessage =
				rootMessage && messageToEdit.parent === rootMessage.id && messageToEdit.role === 'user';

			let parentId = messageToEdit.parent;

			if (parentId === undefined || parentId === null) {
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
					extra: messageToEdit.extra ? JSON.parse(JSON.stringify(messageToEdit.extra)) : undefined,
					model: messageToEdit.model // Preserve original model info when branching
				},
				parentId
			);

			await DatabaseStore.updateCurrentNode(this.activeConversation.id, newMessage.id);
			this.activeConversation.currNode = newMessage.id;
			this.updateConversationTimestamp();

			// If this is the first user message, update the conversation title with confirmation if needed
			if (isFirstUserMessage && newContent.trim()) {
				await this.updateConversationTitleWithConfirmation(
					this.activeConversation.id,
					newContent.trim(),
					this.titleUpdateConfirmationCallback
				);
			}

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
			if (this.isAbortError(error)) return;

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

		this.errorDialogState = null;
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
export const errorDialog = () => chatStore.errorDialogState;

export const createConversation = chatStore.createConversation.bind(chatStore);
export const downloadConversation = chatStore.downloadConversation.bind(chatStore);
export const exportAllConversations = chatStore.exportAllConversations.bind(chatStore);
export const importConversations = chatStore.importConversations.bind(chatStore);
export const deleteConversation = chatStore.deleteConversation.bind(chatStore);
export const sendMessage = chatStore.sendMessage.bind(chatStore);
export const dismissErrorDialog = chatStore.dismissErrorDialog.bind(chatStore);

export const gracefulStop = chatStore.gracefulStop.bind(chatStore);

// Branching operations
export const refreshActiveMessages = chatStore.refreshActiveMessages.bind(chatStore);
export const navigateToSibling = chatStore.navigateToSibling.bind(chatStore);
export const editAssistantMessage = chatStore.editAssistantMessage.bind(chatStore);
export const editMessageWithBranching = chatStore.editMessageWithBranching.bind(chatStore);
export const regenerateMessageWithBranching =
	chatStore.regenerateMessageWithBranching.bind(chatStore);
export const deleteMessage = chatStore.deleteMessage.bind(chatStore);
export const getDeletionInfo = chatStore.getDeletionInfo.bind(chatStore);
export const updateConversationName = chatStore.updateConversationName.bind(chatStore);
export const setTitleUpdateConfirmationCallback =
	chatStore.setTitleUpdateConfirmationCallback.bind(chatStore);

export function stopGeneration() {
	chatStore.stopGeneration();
}
export const messages = () => chatStore.activeMessages;
