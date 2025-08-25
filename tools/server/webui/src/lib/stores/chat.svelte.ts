import { ChatService } from '$lib/services/chat';
import { DatabaseService } from '$lib/services';
import { goto } from '$app/navigation';
import { browser } from '$app/environment';
import { extractPartialThinking } from '$lib/utils/thinking';
import { config } from '$lib/stores/settings.svelte';
import { slotsService } from '$lib/services/slots';
import { filterByLeafNodeId, findLeafNode, findDescendantMessages } from '$lib/utils/branching';

class ChatStore {
	activeConversation = $state<DatabaseConversation | null>(null);
	activeMessages = $state<DatabaseMessage[]>([]);
	conversations = $state<DatabaseConversation[]>([]);
	currentResponse = $state('');
	isInitialized = $state(false);
	isLoading = $state(false);
	maxContextError = $state<{ message: string; estimatedTokens: number; maxContext: number } | null>(null);
	private chatService = new ChatService();

	constructor() {
		if (browser) {
			this.initialize();
		}
	}

	async initialize() {
		try {
			await this.loadConversations();

			this.maxContextError = null;

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

		this.maxContextError = null;

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
			
			if (conversation.currNode) {
				const allMessages = await DatabaseService.getConversationMessages(convId);
				this.activeMessages = filterByLeafNodeId(allMessages, conversation.currNode, false) as DatabaseMessage[];
			} else {
				// Load all messages for conversations without currNode (backward compatibility)
				this.activeMessages = await DatabaseService.getConversationMessages(convId);
			}

			this.maxContextError = null;

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
			let parentId: string | null = null;
			
			if (parent === '-1') {
				if (this.activeMessages.length > 0) {
					parentId = this.activeMessages[this.activeMessages.length - 1].id;
				} else {
					const allMessages = await DatabaseService.getConversationMessages(this.activeConversation.id);
					const rootMessage = allMessages.find(m => m.parent === null && m.type === 'root');
					
					if (!rootMessage) {
						const rootId = await DatabaseService.createRootMessage(this.activeConversation.id);
						parentId = rootId;
					} else {
						parentId = rootMessage.id;
					}
				}
			} else {
				parentId = parent;
			}

			const message = await DatabaseService.createMessageBranch({
				convId: this.activeConversation.id,
				role,
				content,
				type,
				timestamp: Date.now(),
				thinking: '',
				children: [],
				extra: extras
			}, parentId);

			this.activeMessages.push(message);

			await DatabaseService.updateCurrentNode(this.activeConversation.id, message.id);
			this.activeConversation.currNode = message.id;

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

		const currentConfig = config();
		
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
	
		let streamedReasoningContent = '';

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

				onReasoningChunk: (reasoningChunk: string) => {
					streamedReasoningContent += reasoningChunk;

					const messageIndex = this.activeMessages.findIndex(
						(m) => m.id === assistantMessage.id
					);

					if (messageIndex !== -1) {
						// Update message with reasoning content
						this.activeMessages[messageIndex].thinking = streamedReasoningContent;
					}
				},

				onComplete: async (finalContent?: string, reasoningContent?: string) => {
					// Update assistant message in database
					await DatabaseService.updateMessage(assistantMessage.id, {
						content: finalContent || streamedContent,
						thinking: reasoningContent || streamedReasoningContent
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

					// Handle context errors specially
					if (error.name === 'ContextError') {
						console.warn('Context error detected:', error.message);
						this.isLoading = false;
						this.currentResponse = '';

						slotsService.stopPolling();

						// Remove the assistant message that was created but failed
						const messageIndex = this.activeMessages.findIndex(
							(m: DatabaseMessage) => m.id === assistantMessage.id
						);

						if (messageIndex !== -1) {
							// Remove from UI
							this.activeMessages.splice(messageIndex, 1);
							// Remove from database
							DatabaseService.deleteMessage(assistantMessage.id).catch(console.error);
						}

						this.maxContextError = {
							message: error.message,
							estimatedTokens: 0, // Server-side error, we don't have client estimates
							maxContext: 4096 // Default fallback, will be updated by context service if available
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

			const allMessages = await DatabaseService.getConversationMessages(this.activeConversation.id);
			const assistantMessage = await this.addMessage('assistant', '');

			if (!assistantMessage) {
				throw new Error('Failed to create assistant message');
			}

			await this.streamChatCompletion(allMessages, assistantMessage, undefined, (error: Error) => {
				// Handle context errors by also removing the user message
				if (error.name === 'ContextError' && userMessage) {
					slotsService.stopPolling();
					
					// Remove user message from UI
					const userMessageIndex = this.activeMessages.findIndex(
						(m: DatabaseMessage) => m.id === userMessage!.id
					);
					if (userMessageIndex !== -1) {
						this.activeMessages.splice(userMessageIndex, 1);
						// Remove from database
						DatabaseService.deleteMessage(userMessage.id).catch(console.error);
					}
				}
			});
		} catch (error) {
			if (this.isAbortError(error)) {
				this.isLoading = false;
				return;
			}

			// Handle context errors by removing the user message if it was added
			if (error instanceof Error && error.name === 'ContextError' && userMessage) {
				slotsService.stopPolling();
				
				const userMessageIndex = this.activeMessages.findIndex(
					(m: DatabaseMessage) => m.id === userMessage.id
				);
				if (userMessageIndex !== -1) {
					this.activeMessages.splice(userMessageIndex, 1);
					DatabaseService.deleteMessage(userMessage.id).catch(console.error);
				}
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
	setMaxContextError(error: { message: string; estimatedTokens: number; maxContext: number } | null): void {
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
				
				
				const updateData: { content: string; thinking?: string } = {
					content: partialThinking.remainingContent || this.currentResponse
				};

				if (partialThinking.thinking) {
					updateData.thinking = partialThinking.thinking;
				}
				
				await DatabaseService.updateMessage(lastMessage.id, updateData);

				lastMessage.content = partialThinking.remainingContent || this.currentResponse;
			} catch (error) {
				lastMessage.content = this.currentResponse;
				console.error('Failed to save partial response:', error);
			}
		} else {
			console.error('Last message is not an assistant message');
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
					() => {
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

	/**
	 * Gets information about what messages will be deleted when deleting a specific message.
	 * This is used to inform the user about cascading deletion.
	 * 
	 * @param messageId - ID of the message to be deleted
	 * @returns Object with deletion info including count and types of messages
	 */
	async getDeletionInfo(messageId: string): Promise<{ 
		totalCount: number; 
		userMessages: number; 
		assistantMessages: number; 
		messageTypes: string[] 
	}> {
		if (!this.activeConversation) {
			return { totalCount: 0, userMessages: 0, assistantMessages: 0, messageTypes: [] };
		}

		const allMessages = await DatabaseService.getConversationMessages(this.activeConversation.id);
		const descendants = findDescendantMessages(allMessages, messageId);
		const allToDelete = [messageId, ...descendants];
		
		const messagesToDelete = allMessages.filter(m => allToDelete.includes(m.id));
		
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

	async deleteMessage(messageId: string): Promise<void> {
		try {
			if (!this.activeConversation) return;

			// Get all messages to find siblings before deletion
			const allMessages = await DatabaseService.getConversationMessages(this.activeConversation.id);
			const messageToDelete = allMessages.find(m => m.id === messageId);
			
			if (!messageToDelete) {
				console.error('Message to delete not found');
				return;
			}

			// Check if the deleted message is in the current conversation path
			const currentPath = filterByLeafNodeId(allMessages, this.activeConversation.currNode || '', false);
			const isInCurrentPath = currentPath.some(m => m.id === messageId);
			
			// If the deleted message is in the current path, we need to update currNode
			if (isInCurrentPath && messageToDelete.parent) {
				// Find all siblings (messages with same parent)
				const siblings = allMessages.filter(m => m.parent === messageToDelete.parent && m.id !== messageId);
				
				if (siblings.length > 0) {
					// Find the latest sibling (highest timestamp)
					const latestSibling = siblings.reduce((latest, sibling) => 
						sibling.timestamp > latest.timestamp ? sibling : latest
					);
					
					// Find the leaf node for this sibling branch to get the complete conversation path
					const leafNodeId = findLeafNode(allMessages, latestSibling.id);
					
					// Update conversation to use the leaf node of the latest remaining sibling
					await DatabaseService.updateCurrentNode(this.activeConversation.id, leafNodeId);
					this.activeConversation.currNode = leafNodeId;
				} else {
					// No siblings left, navigate to parent if it exists
					if (messageToDelete.parent) {
						const parentLeafId = findLeafNode(allMessages, messageToDelete.parent);
						await DatabaseService.updateCurrentNode(this.activeConversation.id, parentLeafId);
						this.activeConversation.currNode = parentLeafId;
					}
				}
			}

			// Use cascading deletion to remove the message and all its descendants
			await DatabaseService.deleteMessageCascading(this.activeConversation.id, messageId);

			// Refresh active messages to show the updated branch
			await this.refreshActiveMessages();

			// Update conversation timestamp
			this.updateConversationTimestamp();
		} catch (error) {
			console.error('Failed to delete message:', error);
		}
	}

	clearActiveConversation() {
		this.activeConversation = null;
		this.activeMessages = [];
		this.currentResponse = '';
		this.isLoading = false;
		this.maxContextError = null;
	}

	// === BRANCHING OPERATIONS ===

	/**
	 * Refreshes active messages to show current conversation path based on currNode.
	 * This is called after branch navigation to update the displayed messages.
	 */
	async refreshActiveMessages(): Promise<void> {
		if (!this.activeConversation) return;

		const allMessages = await DatabaseService.getConversationMessages(this.activeConversation.id);
		if (allMessages.length === 0) {
			this.activeMessages = [];
			return;
		}

		// Get current leaf node or use latest message
		const leafNodeId = this.activeConversation.currNode ||
			allMessages.reduce((latest, msg) => 
				msg.timestamp > latest.timestamp ? msg : latest
			).id;

		// Get conversation path for current branch
		const currentPath = filterByLeafNodeId(allMessages, leafNodeId, false) as DatabaseMessage[];
		
		// Force reactive update by clearing and reassigning
		this.activeMessages.length = 0;
		this.activeMessages.push(...currentPath);
	}

	/**
	 * Navigates to a specific sibling branch by updating currNode and refreshing messages.
	 * 
	 * @param siblingId - The sibling message ID to navigate to
	 */
	async navigateToSibling(siblingId: string): Promise<void> {
		if (!this.activeConversation) return;

		// Update conversation's current node
		await DatabaseService.updateCurrentNode(this.activeConversation.id, siblingId);
		
		// Update local state
		this.activeConversation.currNode = siblingId;
		
		// Refresh active messages to show new branch
		await this.refreshActiveMessages();
	}

	/**
	 * Edits a message by creating a new branch with the edited content.
	 * This preserves the original message and creates a sibling with new content.
	 * 
	 * @param messageId - ID of message to edit
	 * @param newContent - New content for the message
	 */
	async editMessageWithBranching(messageId: string, newContent: string): Promise<void> {
		if (!this.activeConversation || this.isLoading) return;

		try {
			const messageIndex = this.activeMessages.findIndex((m: DatabaseMessage) => m.id === messageId);
			if (messageIndex === -1) {
				console.error('Message not found for editing');
				return;
			}

			const messageToEdit = this.activeMessages[messageIndex];
			if (messageToEdit.role !== 'user') {
				console.error('Only user messages can be edited');
				return;
			}

			// Use the same parent as the original message, or find appropriate parent if undefined
			let parentId = messageToEdit.parent;
			
			// If parent is undefined/null, find the appropriate parent
			if (parentId === undefined || parentId === null) {
				// Get all messages to find root or previous message
				const allMessages = await DatabaseService.getConversationMessages(this.activeConversation.id);
				
				// Find root message
				const rootMessage = allMessages.find(m => m.type === 'root' && m.parent === null);
				if (rootMessage) {
					parentId = rootMessage.id;
				} else {
					console.error('No root message found for editing');
					return;
				}
			}

			// Create new message branch with edited content (use snapshot to avoid proxy cloning issues)
			const newMessage = await DatabaseService.createMessageBranch({
				convId: messageToEdit.convId,
				type: messageToEdit.type,
				timestamp: Date.now(),
				role: messageToEdit.role,
				content: newContent,
				thinking: messageToEdit.thinking || '',
				children: [],
				extra: messageToEdit.extra ? JSON.parse(JSON.stringify(messageToEdit.extra)) : undefined
			}, parentId);

			// Update conversation's current node to the new message
			await DatabaseService.updateCurrentNode(this.activeConversation.id, newMessage.id);
			this.activeConversation.currNode = newMessage.id;

			// Update conversation timestamp
			this.updateConversationTimestamp();

			// Navigate to the new branch (this will show only the new path)
			await this.navigateToSibling(newMessage.id);

			// If this was a user message, regenerate assistant response
			if (messageToEdit.role === 'user') {
				await this.generateResponseForMessage(newMessage.id);
			}
		} catch (error) {
			console.error('Failed to edit message with branching:', error);
		}
	}

	/**
	 * Regenerates an assistant message by creating a new branch with a new response.
	 * This preserves the original response and creates a sibling with new content.
	 * 
	 * @param messageId - ID of assistant message to regenerate
	 */
	async regenerateMessageWithBranching(messageId: string): Promise<void> {
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

			// Find parent message (should be user message)
			const parentMessage = this.activeMessages.find(m => m.id === messageToRegenerate.parent);
			if (!parentMessage) {
				console.error('Parent message not found for regeneration');
				return;
			}

			// Create new assistant message branch with empty content
			const newAssistantMessage = await DatabaseService.createMessageBranch({
				convId: this.activeConversation.id,
				type: 'text',
				timestamp: Date.now(),
				role: 'assistant',
				content: '',
				thinking: '',
				children: []
			}, parentMessage.id);

			// Update conversation's current node to the new message
			await DatabaseService.updateCurrentNode(this.activeConversation.id, newAssistantMessage.id);
			this.activeConversation.currNode = newAssistantMessage.id;

			// Update conversation timestamp
			this.updateConversationTimestamp();

			// Navigate to the new branch (this will show only the new path)
			await this.navigateToSibling(newAssistantMessage.id);

			// Stream new response to the new assistant message
			const allMessages = await DatabaseService.getConversationMessages(this.activeConversation.id);
			const conversationPath = filterByLeafNodeId(allMessages, parentMessage.id, false) as DatabaseMessage[];
			
			await this.streamChatCompletion(conversationPath, newAssistantMessage);
		} catch (error) {
			console.error('Failed to regenerate message with branching:', error);
		}
	}

	/**
	 * Generates a new assistant response for a given user message.
	 * Creates a new assistant message branch and streams the response.
	 * 
	 * @param userMessageId - ID of user message to respond to
	 */
	private async generateResponseForMessage(userMessageId: string): Promise<void> {
		if (!this.activeConversation) return;

		this.isLoading = true;
		this.currentResponse = '';

		try {
			// Get conversation path up to the user message
			const allMessages = await DatabaseService.getConversationMessages(this.activeConversation.id);
			const conversationPath = filterByLeafNodeId(allMessages, userMessageId, false) as DatabaseMessage[];

			// Create new assistant message branch
			const assistantMessage = await DatabaseService.createMessageBranch({
				convId: this.activeConversation.id,
				type: 'text',
				timestamp: Date.now(),
				role: 'assistant',
				content: '',
				thinking: '',
				children: []
			}, userMessageId);

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
export const regenerateMessageWithBranching = chatStore.regenerateMessageWithBranching.bind(chatStore);
export const deleteMessage = chatStore.deleteMessage.bind(chatStore);
export const getDeletionInfo = chatStore.getDeletionInfo.bind(chatStore);
export const updateConversationName = chatStore.updateConversationName.bind(chatStore);

export function stopGeneration() {
	chatStore.stopGeneration();
}
export const messages = () => chatStore.activeMessages;
