import { DatabaseService, ChatService } from '$lib/services';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import { config } from '$lib/stores/settings.svelte';
import { agenticClient } from '$lib/clients';
import { contextSize, isRouterMode } from '$lib/stores/server.svelte';
import {
	selectedModelName,
	modelsStore,
	selectedModelContextSize
} from '$lib/stores/models.svelte';
import {
	normalizeModelName,
	filterByLeafNodeId,
	findDescendantMessages,
	findLeafNode
} from '$lib/utils';
import { DEFAULT_CONTEXT } from '$lib/constants/default-context';
import { getAgenticConfig } from '$lib/utils/agentic';
import { SYSTEM_MESSAGE_PLACEHOLDER } from '$lib/constants/ui';
import type { ChatMessageTimings, ChatMessagePromptProgress } from '$lib/types/chat';
import type { DatabaseMessage, DatabaseMessageExtra } from '$lib/types/database';

export interface ApiProcessingState {
	status: 'idle' | 'preparing' | 'generating';
	tokensDecoded: number;
	tokensRemaining: number;
	contextUsed: number;
	contextTotal: number;
	outputTokensUsed: number;
	outputTokensMax: number;
	hasNextToken: boolean;
	tokensPerSecond: number;
	temperature: number;
	topP: number;
	speculative: boolean;
	progressPercent?: number;
	promptProgress?: {
		total: number;
		cache: number;
		processed: number;
		time_ms: number;
	};
	promptTokens: number;
	promptMs?: number;
	cacheTokens: number;
}

export interface ErrorDialogState {
	type: 'timeout' | 'server';
	message: string;
	contextInfo?: { n_prompt_tokens: number; n_ctx: number };
}

export interface ChatStreamCallbacks {
	onChunk?: (chunk: string) => void;
	onReasoningChunk?: (chunk: string) => void;
	onToolCallChunk?: (chunk: string) => void;
	onAttachments?: (extras: DatabaseMessageExtra[]) => void;
	onModel?: (model: string) => void;
	onTimings?: (timings?: ChatMessageTimings, promptProgress?: ChatMessagePromptProgress) => void;
	onComplete?: (
		content?: string,
		reasoningContent?: string,
		timings?: ChatMessageTimings,
		toolCallContent?: string
	) => void;
	onError?: (error: Error) => void;
}

type ChatRole = 'user' | 'assistant' | 'system' | 'tool';
type ChatMessageType = 'text' | 'root';

interface ChatStoreStateCallbacks {
	setChatLoading: (convId: string, loading: boolean) => void;
	setChatStreaming: (convId: string, response: string, messageId: string) => void;
	clearChatStreaming: (convId: string) => void;
	getChatStreaming: (convId: string) => { response: string; messageId: string } | undefined;
	setProcessingState: (convId: string, state: ApiProcessingState | null) => void;
	getProcessingState: (convId: string) => ApiProcessingState | null;
	setActiveProcessingConversation: (convId: string | null) => void;
	setStreamingActive: (active: boolean) => void;
	showErrorDialog: (state: ErrorDialogState | null) => void;
	getAbortController: (convId: string) => AbortController;
	abortRequest: (convId?: string) => void;
	setPendingEditMessageId: (messageId: string | null) => void;
	getActiveConversationId: () => string | null;
	getCurrentResponse: () => string;
}

/**
 * ChatClient - Business Logic Facade for Chat Operations
 *
 * Coordinates AI interactions, message operations, and streaming orchestration.
 *
 * **Architecture & Relationships:**
 * - **ChatClient** (this class): Business logic facade
 *   - Uses ChatService for low-level API operations
 *   - Updates chatStore with reactive state
 *   - Coordinates with conversationsStore for persistence
 *   - Handles streaming, branching, and error recovery
 *
 * - **ChatService**: Stateless API layer (sendMessage, streaming)
 * - **chatStore**: Reactive state only ($state, getters, setters)
 *
 * **Key Responsibilities:**
 * - Message lifecycle (send, edit, delete, branch)
 * - AI streaming orchestration
 * - Processing state management (timing, context info)
 * - Error handling (timeout, server errors)
 * - Graceful stop with partial response saving
 */
export class ChatClient {
	private storeCallbacks: ChatStoreStateCallbacks | null = null;

	/**
	 *
	 *
	 * Store Integration
	 *
	 *
	 */

	/**
	 * Sets callbacks for store state updates.
	 * Called by chatStore during initialization to establish bidirectional communication.
	 */
	setStoreCallbacks(callbacks: ChatStoreStateCallbacks): void {
		this.storeCallbacks = callbacks;
	}

	private get store(): ChatStoreStateCallbacks {
		if (!this.storeCallbacks) {
			throw new Error('ChatClient: Store callbacks not initialized');
		}
		return this.storeCallbacks;
	}

	/**
	 *
	 *
	 * Message Operations
	 *
	 *
	 */

	private getMessageByIdWithRole(
		messageId: string,
		expectedRole?: ChatRole
	): { message: DatabaseMessage; index: number } | null {
		const index = conversationsStore.findMessageIndex(messageId);
		if (index === -1) return null;

		const message = conversationsStore.activeMessages[index];
		if (expectedRole && message.role !== expectedRole) return null;

		return { message, index };
	}

	/**
	 * Adds a new message to the active conversation.
	 * @param role - Message role (user, assistant, system, tool)
	 * @param content - Message text content
	 * @param type - Message type (text or root)
	 * @param parent - Parent message ID, or '-1' to append to conversation end
	 * @param extras - Optional attachments (images, files, etc.)
	 * @returns The created message or null if failed
	 */
	async addMessage(
		role: ChatRole,
		content: string,
		type: ChatMessageType = 'text',
		parent: string = '-1',
		extras?: DatabaseMessageExtra[]
	): Promise<DatabaseMessage | null> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv) {
			console.error('No active conversation when trying to add message');
			return null;
		}

		try {
			let parentId: string | null = null;

			if (parent === '-1') {
				const activeMessages = conversationsStore.activeMessages;
				if (activeMessages.length > 0) {
					parentId = activeMessages[activeMessages.length - 1].id;
				} else {
					const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
					const rootMessage = allMessages.find((m) => m.parent === null && m.type === 'root');
					if (!rootMessage) {
						parentId = await DatabaseService.createRootMessage(activeConv.id);
					} else {
						parentId = rootMessage.id;
					}
				}
			} else {
				parentId = parent;
			}

			const message = await DatabaseService.createMessageBranch(
				{
					convId: activeConv.id,
					role,
					content,
					type,
					timestamp: Date.now(),
					thinking: '',
					toolCalls: '',
					children: [],
					extra: extras
				},
				parentId
			);

			conversationsStore.addMessageToActive(message);
			await conversationsStore.updateCurrentNode(message.id);
			conversationsStore.updateConversationTimestamp();

			return message;
		} catch (error) {
			console.error('Failed to add message:', error);
			return null;
		}
	}

	/**
	 * Adds a system message placeholder at the conversation start.
	 * Triggers edit mode to allow user to customize the system prompt.
	 * If conversation doesn't exist, creates one first.
	 */
	async addSystemPrompt(): Promise<void> {
		let activeConv = conversationsStore.activeConversation;

		if (!activeConv) {
			await conversationsStore.createConversation();
			activeConv = conversationsStore.activeConversation;
		}
		if (!activeConv) return;

		try {
			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);
			let rootId: string;

			if (!rootMessage) {
				rootId = await DatabaseService.createRootMessage(activeConv.id);
			} else {
				rootId = rootMessage.id;
			}

			const existingSystemMessage = allMessages.find(
				(m) => m.role === 'system' && m.parent === rootId
			);

			if (existingSystemMessage) {
				this.store.setPendingEditMessageId(existingSystemMessage.id);

				if (!conversationsStore.activeMessages.some((m) => m.id === existingSystemMessage.id)) {
					conversationsStore.activeMessages.unshift(existingSystemMessage);
				}
				return;
			}

			const activeMessages = conversationsStore.activeMessages;
			const firstActiveMessage = activeMessages.find((m) => m.parent === rootId);

			const systemMessage = await DatabaseService.createSystemMessage(
				activeConv.id,
				SYSTEM_MESSAGE_PLACEHOLDER,
				rootId
			);

			if (firstActiveMessage) {
				await DatabaseService.updateMessage(firstActiveMessage.id, {
					parent: systemMessage.id
				});

				await DatabaseService.updateMessage(systemMessage.id, {
					children: [firstActiveMessage.id]
				});

				const updatedRootChildren = rootMessage
					? rootMessage.children.filter((id: string) => id !== firstActiveMessage.id)
					: [];
				await DatabaseService.updateMessage(rootId, {
					children: [
						...updatedRootChildren.filter((id: string) => id !== systemMessage.id),
						systemMessage.id
					]
				});

				const firstMsgIndex = conversationsStore.findMessageIndex(firstActiveMessage.id);
				if (firstMsgIndex !== -1) {
					conversationsStore.updateMessageAtIndex(firstMsgIndex, { parent: systemMessage.id });
				}
			}

			conversationsStore.activeMessages.unshift(systemMessage);
			this.store.setPendingEditMessageId(systemMessage.id);
			conversationsStore.updateConversationTimestamp();
		} catch (error) {
			console.error('Failed to add system prompt:', error);
		}
	}

	/**
	 * Removes a system message placeholder without deleting its children.
	 * @returns true if the entire conversation was deleted, false otherwise
	 */
	async removeSystemPromptPlaceholder(messageId: string): Promise<boolean> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv) return false;

		try {
			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const systemMessage = allMessages.find((m) => m.id === messageId);
			if (!systemMessage || systemMessage.role !== 'system') return false;

			const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);
			if (!rootMessage) return false;

			const isEmptyConversation = allMessages.length === 2 && systemMessage.children.length === 0;

			if (isEmptyConversation) {
				await conversationsStore.deleteConversation(activeConv.id);
				return true;
			}

			for (const childId of systemMessage.children) {
				await DatabaseService.updateMessage(childId, { parent: rootMessage.id });

				const childIndex = conversationsStore.findMessageIndex(childId);
				if (childIndex !== -1) {
					conversationsStore.updateMessageAtIndex(childIndex, { parent: rootMessage.id });
				}
			}

			const newRootChildren = [
				...rootMessage.children.filter((id: string) => id !== messageId),
				...systemMessage.children
			];

			await DatabaseService.updateMessage(rootMessage.id, { children: newRootChildren });
			await DatabaseService.deleteMessage(messageId);

			const systemIndex = conversationsStore.findMessageIndex(messageId);
			if (systemIndex !== -1) {
				conversationsStore.activeMessages.splice(systemIndex, 1);
			}

			conversationsStore.updateConversationTimestamp();

			return false;
		} catch (error) {
			console.error('Failed to remove system prompt placeholder:', error);
			return false;
		}
	}

	/**
	 *
	 *
	 * Message Sending & Streaming
	 *
	 *
	 */

	private async createAssistantMessage(parentId?: string): Promise<DatabaseMessage | null> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv) return null;

		return await DatabaseService.createMessageBranch(
			{
				convId: activeConv.id,
				type: 'text',
				role: 'assistant',
				content: '',
				timestamp: Date.now(),
				thinking: '',
				toolCalls: '',
				children: [],
				model: null
			},
			parentId || null
		);
	}

	/**
	 * Sends a user message and triggers AI response generation.
	 * Creates conversation if none exists, handles system prompt injection,
	 * and orchestrates the streaming response flow.
	 * @param content - User message text
	 * @param extras - Optional attachments
	 */
	async sendMessage(content: string, extras?: DatabaseMessageExtra[]): Promise<void> {
		if (!content.trim() && (!extras || extras.length === 0)) return;
		const activeConv = conversationsStore.activeConversation;
		if (activeConv && this.isChatLoading(activeConv.id)) return;

		let isNewConversation = false;
		if (!activeConv) {
			await conversationsStore.createConversation();
			isNewConversation = true;
		}
		const currentConv = conversationsStore.activeConversation;
		if (!currentConv) return;

		this.store.showErrorDialog(null);
		this.store.setChatLoading(currentConv.id, true);
		this.store.clearChatStreaming(currentConv.id);

		try {
			if (isNewConversation) {
				const rootId = await DatabaseService.createRootMessage(currentConv.id);
				const currentConfig = config();
				const systemPrompt = currentConfig.systemMessage?.toString().trim();

				if (systemPrompt) {
					const systemMessage = await DatabaseService.createSystemMessage(
						currentConv.id,
						systemPrompt,
						rootId
					);
					conversationsStore.addMessageToActive(systemMessage);
				}
			}

			const userMessage = await this.addMessage('user', content, 'text', '-1', extras);
			if (!userMessage) throw new Error('Failed to add user message');
			if (isNewConversation && content)
				await conversationsStore.updateConversationName(currentConv.id, content.trim());

			const assistantMessage = await this.createAssistantMessage(userMessage.id);
			if (!assistantMessage) throw new Error('Failed to create assistant message');

			conversationsStore.addMessageToActive(assistantMessage);
			await this.streamChatCompletion(
				conversationsStore.activeMessages.slice(0, -1),
				assistantMessage
			);
		} catch (error) {
			if (this.isAbortError(error)) {
				this.store.setChatLoading(currentConv.id, false);
				return;
			}
			console.error('Failed to send message:', error);
			this.store.setChatLoading(currentConv.id, false);

			const dialogType =
				error instanceof Error && error.name === 'TimeoutError' ? 'timeout' : 'server';
			const contextInfo = (
				error as Error & { contextInfo?: { n_prompt_tokens: number; n_ctx: number } }
			).contextInfo;

			this.store.showErrorDialog({
				type: dialogType,
				message: error instanceof Error ? error.message : 'Unknown error',
				contextInfo
			});
		}
	}

	private async streamChatCompletion(
		allMessages: DatabaseMessage[],
		assistantMessage: DatabaseMessage,
		onComplete?: (content: string) => Promise<void>,
		onError?: (error: Error) => void,
		modelOverride?: string | null
	): Promise<void> {
		if (isRouterMode()) {
			const modelName = modelOverride || selectedModelName();
			if (modelName && !modelsStore.getModelProps(modelName)) {
				await modelsStore.fetchModelProps(modelName);
			}
		}

		let streamedContent = '';
		let streamedReasoningContent = '';
		let streamedToolCallContent = '';
		let resolvedModel: string | null = null;
		let modelPersisted = false;
		let streamedExtras: DatabaseMessageExtra[] = assistantMessage.extra
			? JSON.parse(JSON.stringify(assistantMessage.extra))
			: [];

		const recordModel = (modelName: string | null | undefined, persistImmediately = true): void => {
			if (!modelName) return;
			const normalizedModel = normalizeModelName(modelName);
			if (!normalizedModel || normalizedModel === resolvedModel) return;
			resolvedModel = normalizedModel;
			const messageIndex = conversationsStore.findMessageIndex(assistantMessage.id);
			conversationsStore.updateMessageAtIndex(messageIndex, { model: normalizedModel });
			if (persistImmediately && !modelPersisted) {
				modelPersisted = true;
				DatabaseService.updateMessage(assistantMessage.id, { model: normalizedModel }).catch(() => {
					modelPersisted = false;
					resolvedModel = null;
				});
			}
		};

		this.store.setStreamingActive(true);
		this.store.setActiveProcessingConversation(assistantMessage.convId);

		const abortController = this.store.getAbortController(assistantMessage.convId);

		const streamCallbacks: ChatStreamCallbacks = {
			onChunk: (chunk: string) => {
				streamedContent += chunk;
				this.store.setChatStreaming(assistantMessage.convId, streamedContent, assistantMessage.id);
				const idx = conversationsStore.findMessageIndex(assistantMessage.id);
				conversationsStore.updateMessageAtIndex(idx, { content: streamedContent });
			},
			onReasoningChunk: (reasoningChunk: string) => {
				streamedReasoningContent += reasoningChunk;
				const idx = conversationsStore.findMessageIndex(assistantMessage.id);
				conversationsStore.updateMessageAtIndex(idx, { thinking: streamedReasoningContent });
			},
			onToolCallChunk: (toolCallChunk: string) => {
				const chunk = toolCallChunk.trim();
				if (!chunk) return;
				streamedToolCallContent = chunk;
				const idx = conversationsStore.findMessageIndex(assistantMessage.id);
				conversationsStore.updateMessageAtIndex(idx, { toolCalls: streamedToolCallContent });
			},
			onAttachments: (extras: DatabaseMessageExtra[]) => {
				if (!extras.length) return;
				streamedExtras = [...streamedExtras, ...extras];
				const idx = conversationsStore.findMessageIndex(assistantMessage.id);
				conversationsStore.updateMessageAtIndex(idx, { extra: streamedExtras });
				DatabaseService.updateMessage(assistantMessage.id, { extra: streamedExtras }).catch(
					console.error
				);
			},
			onModel: (modelName: string) => recordModel(modelName),
			onTimings: (timings?: ChatMessageTimings, promptProgress?: ChatMessagePromptProgress) => {
				const tokensPerSecond =
					timings?.predicted_ms && timings?.predicted_n
						? (timings.predicted_n / timings.predicted_ms) * 1000
						: 0;
				this.updateProcessingStateFromTimings(
					{
						prompt_n: timings?.prompt_n || 0,
						prompt_ms: timings?.prompt_ms,
						predicted_n: timings?.predicted_n || 0,
						predicted_per_second: tokensPerSecond,
						cache_n: timings?.cache_n || 0,
						prompt_progress: promptProgress
					},
					assistantMessage.convId
				);
			},
			onComplete: async (
				finalContent?: string,
				reasoningContent?: string,
				timings?: ChatMessageTimings,
				toolCallContent?: string
			) => {
				this.store.setStreamingActive(false);

				const updateData: Record<string, unknown> = {
					content: finalContent || streamedContent,
					thinking: reasoningContent || streamedReasoningContent,
					toolCalls: toolCallContent || streamedToolCallContent,
					timings
				};
				if (streamedExtras.length > 0) {
					updateData.extra = streamedExtras;
				}
				if (resolvedModel && !modelPersisted) {
					updateData.model = resolvedModel;
				}
				await DatabaseService.updateMessage(assistantMessage.id, updateData);

				const idx = conversationsStore.findMessageIndex(assistantMessage.id);
				const uiUpdate: Partial<DatabaseMessage> = {
					content: updateData.content as string,
					toolCalls: updateData.toolCalls as string
				};
				if (streamedExtras.length > 0) {
					uiUpdate.extra = streamedExtras;
				}
				if (timings) uiUpdate.timings = timings;
				if (resolvedModel) uiUpdate.model = resolvedModel;

				conversationsStore.updateMessageAtIndex(idx, uiUpdate);
				await conversationsStore.updateCurrentNode(assistantMessage.id);

				if (onComplete) await onComplete(streamedContent);
				this.store.setChatLoading(assistantMessage.convId, false);
				this.store.clearChatStreaming(assistantMessage.convId);
				this.store.setProcessingState(assistantMessage.convId, null);

				if (isRouterMode()) {
					modelsStore.fetchRouterModels().catch(console.error);
				}
			},
			onError: (error: Error) => {
				this.store.setStreamingActive(false);

				if (this.isAbortError(error)) {
					this.store.setChatLoading(assistantMessage.convId, false);
					this.store.clearChatStreaming(assistantMessage.convId);
					this.store.setProcessingState(assistantMessage.convId, null);
					return;
				}

				console.error('Streaming error:', error);

				this.store.setChatLoading(assistantMessage.convId, false);
				this.store.clearChatStreaming(assistantMessage.convId);
				this.store.setProcessingState(assistantMessage.convId, null);

				const idx = conversationsStore.findMessageIndex(assistantMessage.id);

				if (idx !== -1) {
					const failedMessage = conversationsStore.removeMessageAtIndex(idx);
					if (failedMessage) DatabaseService.deleteMessage(failedMessage.id).catch(console.error);
				}

				const contextInfo = (
					error as Error & { contextInfo?: { n_prompt_tokens: number; n_ctx: number } }
				).contextInfo;

				this.store.showErrorDialog({
					type: error.name === 'TimeoutError' ? 'timeout' : 'server',
					message: error.message,
					contextInfo
				});

				if (onError) onError(error);
			}
		};

		const perChatOverrides = conversationsStore.activeConversation?.mcpServerOverrides;
		const agenticConfig = getAgenticConfig(config(), perChatOverrides);
		if (agenticConfig.enabled) {
			const agenticResult = await agenticClient.runAgenticFlow({
				messages: allMessages,
				options: {
					...this.getApiOptions(),
					...(modelOverride ? { model: modelOverride } : {})
				},
				callbacks: streamCallbacks,
				signal: abortController.signal,
				perChatOverrides
			});

			if (agenticResult.handled) {
				return;
			}
		}

		await ChatService.sendMessage(
			allMessages,
			{
				...this.getApiOptions(),
				...(modelOverride ? { model: modelOverride } : {}),
				...streamCallbacks
			},
			assistantMessage.convId,
			abortController.signal
		);
	}

	/**
	 *
	 *
	 * Generation Control
	 *
	 *
	 */

	/**
	 * Stops generation for the active conversation.
	 * Saves any partial response before aborting.
	 */
	async stopGeneration(): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv) return;
		await this.stopGenerationForChat(activeConv.id);
	}

	/**
	 * Stops generation for a specific conversation.
	 * @param convId - Conversation ID to stop
	 */
	async stopGenerationForChat(convId: string): Promise<void> {
		await this.savePartialResponseIfNeeded(convId);

		this.store.setStreamingActive(false);
		this.store.abortRequest(convId);
		this.store.setChatLoading(convId, false);
		this.store.clearChatStreaming(convId);
		this.store.setProcessingState(convId, null);
	}

	private async savePartialResponseIfNeeded(convId?: string): Promise<void> {
		const conversationId = convId || conversationsStore.activeConversation?.id;
		if (!conversationId) return;

		const streamingState = this.store.getChatStreaming(conversationId);
		if (!streamingState || !streamingState.response.trim()) return;

		const messages =
			conversationId === conversationsStore.activeConversation?.id
				? conversationsStore.activeMessages
				: await conversationsStore.getConversationMessages(conversationId);

		if (!messages.length) return;

		const lastMessage = messages[messages.length - 1];

		if (lastMessage?.role === 'assistant') {
			try {
				const updateData: { content: string; thinking?: string; timings?: ChatMessageTimings } = {
					content: streamingState.response
				};
				if (lastMessage.thinking?.trim()) updateData.thinking = lastMessage.thinking;
				const lastKnownState = this.store.getProcessingState(conversationId);
				if (lastKnownState) {
					updateData.timings = {
						prompt_n: lastKnownState.promptTokens || 0,
						prompt_ms: lastKnownState.promptMs,
						predicted_n: lastKnownState.tokensDecoded || 0,
						cache_n: lastKnownState.cacheTokens || 0,
						predicted_ms:
							lastKnownState.tokensPerSecond && lastKnownState.tokensDecoded
								? (lastKnownState.tokensDecoded / lastKnownState.tokensPerSecond) * 1000
								: undefined
					};
				}

				await DatabaseService.updateMessage(lastMessage.id, updateData);

				lastMessage.content = this.store.getCurrentResponse();

				if (updateData.thinking) lastMessage.thinking = updateData.thinking;
				if (updateData.timings) lastMessage.timings = updateData.timings;
			} catch (error) {
				lastMessage.content = this.store.getCurrentResponse();
				console.error('Failed to save partial response:', error);
			}
		}
	}

	/**
	 *
	 *
	 * Message Editing
	 *
	 *
	 */

	/**
	 * Updates a user message content and regenerates the AI response.
	 * Deletes all messages after the edited one before regenerating.
	 * @param messageId - ID of the user message to update
	 * @param newContent - New message content
	 */
	async updateMessage(messageId: string, newContent: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv) return;
		if (this.isChatLoading(activeConv.id)) await this.stopGeneration();

		const result = this.getMessageByIdWithRole(messageId, 'user');
		if (!result) return;
		const { message: messageToUpdate, index: messageIndex } = result;
		const originalContent = messageToUpdate.content;

		try {
			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);
			const isFirstUserMessage = rootMessage && messageToUpdate.parent === rootMessage.id;

			conversationsStore.updateMessageAtIndex(messageIndex, { content: newContent });
			await DatabaseService.updateMessage(messageId, { content: newContent });

			if (isFirstUserMessage && newContent.trim()) {
				await conversationsStore.updateConversationTitleWithConfirmation(
					activeConv.id,
					newContent.trim()
				);
			}

			const messagesToRemove = conversationsStore.activeMessages.slice(messageIndex + 1);

			for (const message of messagesToRemove) await DatabaseService.deleteMessage(message.id);

			conversationsStore.sliceActiveMessages(messageIndex + 1);
			conversationsStore.updateConversationTimestamp();

			this.store.setChatLoading(activeConv.id, true);
			this.store.clearChatStreaming(activeConv.id);

			const assistantMessage = await this.createAssistantMessage();
			if (!assistantMessage) throw new Error('Failed to create assistant message');

			conversationsStore.addMessageToActive(assistantMessage);

			await conversationsStore.updateCurrentNode(assistantMessage.id);
			await this.streamChatCompletion(
				conversationsStore.activeMessages.slice(0, -1),
				assistantMessage,
				undefined,
				() => {
					conversationsStore.updateMessageAtIndex(conversationsStore.findMessageIndex(messageId), {
						content: originalContent
					});
				}
			);
		} catch (error) {
			if (!this.isAbortError(error)) console.error('Failed to update message:', error);
		}
	}

	/**
	 *
	 *
	 * Message Regeneration
	 *
	 *
	 */

	/**
	 * Regenerates an assistant message by deleting it and all following messages,
	 * then generating a new response.
	 * @param messageId - ID of the assistant message to regenerate
	 */
	async regenerateMessage(messageId: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv || this.isChatLoading(activeConv.id)) return;

		const result = this.getMessageByIdWithRole(messageId, 'assistant');
		if (!result) return;
		const { index: messageIndex } = result;

		try {
			const messagesToRemove = conversationsStore.activeMessages.slice(messageIndex);
			for (const message of messagesToRemove) await DatabaseService.deleteMessage(message.id);
			conversationsStore.sliceActiveMessages(messageIndex);
			conversationsStore.updateConversationTimestamp();

			this.store.setChatLoading(activeConv.id, true);
			this.store.clearChatStreaming(activeConv.id);

			const parentMessageId =
				conversationsStore.activeMessages.length > 0
					? conversationsStore.activeMessages[conversationsStore.activeMessages.length - 1].id
					: undefined;
			const assistantMessage = await this.createAssistantMessage(parentMessageId);
			if (!assistantMessage) throw new Error('Failed to create assistant message');
			conversationsStore.addMessageToActive(assistantMessage);
			await this.streamChatCompletion(
				conversationsStore.activeMessages.slice(0, -1),
				assistantMessage
			);
		} catch (error) {
			if (!this.isAbortError(error)) console.error('Failed to regenerate message:', error);
			this.store.setChatLoading(activeConv?.id || '', false);
		}
	}

	/**
	 * Regenerates an assistant message as a new branch.
	 * Creates a sibling message instead of replacing the original.
	 * @param messageId - ID of the assistant message to regenerate
	 * @param modelOverride - Optional model to use instead of default
	 */
	async regenerateMessageWithBranching(messageId: string, modelOverride?: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv || this.isChatLoading(activeConv.id)) return;
		try {
			const idx = conversationsStore.findMessageIndex(messageId);
			if (idx === -1) return;
			const msg = conversationsStore.activeMessages[idx];
			if (msg.role !== 'assistant') return;

			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const parentMessage = allMessages.find((m) => m.id === msg.parent);
			if (!parentMessage) return;

			this.store.setChatLoading(activeConv.id, true);
			this.store.clearChatStreaming(activeConv.id);

			const newAssistantMessage = await DatabaseService.createMessageBranch(
				{
					convId: msg.convId,
					type: msg.type,
					timestamp: Date.now(),
					role: msg.role,
					content: '',
					thinking: '',
					toolCalls: '',
					children: [],
					model: null
				},
				parentMessage.id
			);
			await conversationsStore.updateCurrentNode(newAssistantMessage.id);
			conversationsStore.updateConversationTimestamp();
			await conversationsStore.refreshActiveMessages();

			const conversationPath = filterByLeafNodeId(
				allMessages,
				parentMessage.id,
				false
			) as DatabaseMessage[];
			const modelToUse = modelOverride || msg.model || undefined;
			await this.streamChatCompletion(
				conversationPath,
				newAssistantMessage,
				undefined,
				undefined,
				modelToUse
			);
		} catch (error) {
			if (!this.isAbortError(error))
				console.error('Failed to regenerate message with branching:', error);
			this.store.setChatLoading(activeConv?.id || '', false);
		}
	}

	/**
	 *
	 *
	 * Message Deletion
	 *
	 *
	 */

	/**
	 * Gets information about messages that would be deleted.
	 * Includes the target message and all its descendants.
	 * @param messageId - ID of the message to analyze
	 * @returns Deletion stats including counts by role
	 */
	async getDeletionInfo(messageId: string): Promise<{
		totalCount: number;
		userMessages: number;
		assistantMessages: number;
		messageTypes: string[];
	}> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv)
			return { totalCount: 0, userMessages: 0, assistantMessages: 0, messageTypes: [] };
		const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
		const descendants = findDescendantMessages(allMessages, messageId);
		const allToDelete = [messageId, ...descendants];
		const messagesToDelete = allMessages.filter((m) => allToDelete.includes(m.id));
		let userMessages = 0,
			assistantMessages = 0;
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
		return { totalCount: allToDelete.length, userMessages, assistantMessages, messageTypes };
	}

	/**
	 * Deletes a message and all its descendants.
	 * Handles branch navigation if deleted message is in current path.
	 * @param messageId - ID of the message to delete
	 */
	async deleteMessage(messageId: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv) return;
		try {
			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const messageToDelete = allMessages.find((m) => m.id === messageId);
			if (!messageToDelete) return;

			const currentPath = filterByLeafNodeId(allMessages, activeConv.currNode || '', false);
			const isInCurrentPath = currentPath.some((m) => m.id === messageId);

			if (isInCurrentPath && messageToDelete.parent) {
				const siblings = allMessages.filter(
					(m) => m.parent === messageToDelete.parent && m.id !== messageId
				);

				if (siblings.length > 0) {
					const latestSibling = siblings.reduce((latest, sibling) =>
						sibling.timestamp > latest.timestamp ? sibling : latest
					);
					await conversationsStore.updateCurrentNode(findLeafNode(allMessages, latestSibling.id));
				} else if (messageToDelete.parent) {
					await conversationsStore.updateCurrentNode(
						findLeafNode(allMessages, messageToDelete.parent)
					);
				}
			}
			await DatabaseService.deleteMessageCascading(activeConv.id, messageId);
			await conversationsStore.refreshActiveMessages();

			conversationsStore.updateConversationTimestamp();
		} catch (error) {
			console.error('Failed to delete message:', error);
		}
	}

	/**
	 *
	 *
	 * Continue Generation
	 *
	 *
	 */

	/**
	 * Continues generating content for an existing assistant message.
	 * Appends new content to the existing message.
	 * @param messageId - ID of the assistant message to continue
	 */
	async continueAssistantMessage(messageId: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv || this.isChatLoading(activeConv.id)) return;

		const result = this.getMessageByIdWithRole(messageId, 'assistant');
		if (!result) return;
		const { message: msg, index: idx } = result;

		try {
			this.store.showErrorDialog(null);
			this.store.setChatLoading(activeConv.id, true);
			this.store.clearChatStreaming(activeConv.id);

			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const dbMessage = allMessages.find((m) => m.id === messageId);

			if (!dbMessage) {
				this.store.setChatLoading(activeConv.id, false);
				return;
			}

			const originalContent = dbMessage.content;
			const originalThinking = dbMessage.thinking || '';

			const conversationContext = conversationsStore.activeMessages.slice(0, idx);
			const contextWithContinue = [
				...conversationContext,
				{ role: 'assistant' as const, content: originalContent }
			];

			let appendedContent = '',
				appendedThinking = '',
				hasReceivedContent = false;

			const abortController = this.store.getAbortController(msg.convId);

			await ChatService.sendMessage(
				contextWithContinue,
				{
					...this.getApiOptions(),

					onChunk: (chunk: string) => {
						hasReceivedContent = true;
						appendedContent += chunk;
						const fullContent = originalContent + appendedContent;
						this.store.setChatStreaming(msg.convId, fullContent, msg.id);
						conversationsStore.updateMessageAtIndex(idx, { content: fullContent });
					},

					onReasoningChunk: (reasoningChunk: string) => {
						hasReceivedContent = true;
						appendedThinking += reasoningChunk;
						conversationsStore.updateMessageAtIndex(idx, {
							thinking: originalThinking + appendedThinking
						});
					},

					onTimings: (timings?: ChatMessageTimings, promptProgress?: ChatMessagePromptProgress) => {
						const tokensPerSecond =
							timings?.predicted_ms && timings?.predicted_n
								? (timings.predicted_n / timings.predicted_ms) * 1000
								: 0;
						this.updateProcessingStateFromTimings(
							{
								prompt_n: timings?.prompt_n || 0,
								prompt_ms: timings?.prompt_ms,
								predicted_n: timings?.predicted_n || 0,
								predicted_per_second: tokensPerSecond,
								cache_n: timings?.cache_n || 0,
								prompt_progress: promptProgress
							},
							msg.convId
						);
					},

					onComplete: async (
						finalContent?: string,
						reasoningContent?: string,
						timings?: ChatMessageTimings
					) => {
						const fullContent = originalContent + (finalContent || appendedContent);
						const fullThinking = originalThinking + (reasoningContent || appendedThinking);
						await DatabaseService.updateMessage(msg.id, {
							content: fullContent,
							thinking: fullThinking,
							timestamp: Date.now(),
							timings
						});
						conversationsStore.updateMessageAtIndex(idx, {
							content: fullContent,
							thinking: fullThinking,
							timestamp: Date.now(),
							timings
						});
						conversationsStore.updateConversationTimestamp();
						this.store.setChatLoading(msg.convId, false);
						this.store.clearChatStreaming(msg.convId);
						this.store.setProcessingState(msg.convId, null);
					},

					onError: async (error: Error) => {
						if (this.isAbortError(error)) {
							if (hasReceivedContent && appendedContent) {
								await DatabaseService.updateMessage(msg.id, {
									content: originalContent + appendedContent,
									thinking: originalThinking + appendedThinking,
									timestamp: Date.now()
								});
								conversationsStore.updateMessageAtIndex(idx, {
									content: originalContent + appendedContent,
									thinking: originalThinking + appendedThinking,
									timestamp: Date.now()
								});
							}
							this.store.setChatLoading(msg.convId, false);
							this.store.clearChatStreaming(msg.convId);
							this.store.setProcessingState(msg.convId, null);
							return;
						}
						console.error('Continue generation error:', error);
						conversationsStore.updateMessageAtIndex(idx, {
							content: originalContent,
							thinking: originalThinking
						});
						await DatabaseService.updateMessage(msg.id, {
							content: originalContent,
							thinking: originalThinking
						});
						this.store.setChatLoading(msg.convId, false);
						this.store.clearChatStreaming(msg.convId);
						this.store.setProcessingState(msg.convId, null);
						this.store.showErrorDialog({
							type: error.name === 'TimeoutError' ? 'timeout' : 'server',
							message: error.message
						});
					}
				},
				msg.convId,
				abortController.signal
			);
		} catch (error) {
			if (!this.isAbortError(error)) console.error('Failed to continue message:', error);
			if (activeConv) this.store.setChatLoading(activeConv.id, false);
		}
	}

	/**
	 * Edits an assistant message content.
	 * Can either replace in-place or create a new branch.
	 * @param messageId - ID of the assistant message to edit
	 * @param newContent - New message content
	 * @param shouldBranch - If true, creates a sibling; if false, replaces in-place
	 */
	async editAssistantMessage(
		messageId: string,
		newContent: string,
		shouldBranch: boolean
	): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv || this.isChatLoading(activeConv.id)) return;

		const result = this.getMessageByIdWithRole(messageId, 'assistant');
		if (!result) return;
		const { message: msg, index: idx } = result;

		try {
			if (shouldBranch) {
				const newMessage = await DatabaseService.createMessageBranch(
					{
						convId: msg.convId,
						type: msg.type,
						timestamp: Date.now(),
						role: msg.role,
						content: newContent,
						thinking: msg.thinking || '',
						toolCalls: msg.toolCalls || '',
						children: [],
						model: msg.model
					},
					msg.parent!
				);
				await conversationsStore.updateCurrentNode(newMessage.id);
			} else {
				await DatabaseService.updateMessage(msg.id, { content: newContent });
				await conversationsStore.updateCurrentNode(msg.id);
				conversationsStore.updateMessageAtIndex(idx, {
					content: newContent
				});
			}
			conversationsStore.updateConversationTimestamp();
			await conversationsStore.refreshActiveMessages();
		} catch (error) {
			console.error('Failed to edit assistant message:', error);
		}
	}

	/**
	 * Edits a user message without regenerating responses.
	 * Preserves all child messages (assistant responses).
	 * @param messageId - ID of the user message to edit
	 * @param newContent - New message content
	 * @param newExtras - Optional new attachments
	 */
	async editUserMessagePreserveResponses(
		messageId: string,
		newContent: string,
		newExtras?: DatabaseMessageExtra[]
	): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv) return;

		const result = this.getMessageByIdWithRole(messageId, 'user');
		if (!result) return;
		const { message: msg, index: idx } = result;

		try {
			const updateData: Partial<DatabaseMessage> = {
				content: newContent
			};

			if (newExtras !== undefined) {
				updateData.extra = JSON.parse(JSON.stringify(newExtras));
			}

			await DatabaseService.updateMessage(messageId, updateData);
			conversationsStore.updateMessageAtIndex(idx, updateData);

			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);

			if (rootMessage && msg.parent === rootMessage.id && newContent.trim()) {
				await conversationsStore.updateConversationTitleWithConfirmation(
					activeConv.id,
					newContent.trim()
				);
			}
			conversationsStore.updateConversationTimestamp();
		} catch (error) {
			console.error('Failed to edit user message:', error);
		}
	}

	/**
	 * Edits a user or system message by creating a new branch.
	 * For user messages, also generates a new AI response.
	 * @param messageId - ID of the message to edit
	 * @param newContent - New message content
	 * @param newExtras - Optional new attachments
	 */
	async editMessageWithBranching(
		messageId: string,
		newContent: string,
		newExtras?: DatabaseMessageExtra[]
	): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv || this.isChatLoading(activeConv.id)) return;

		let result = this.getMessageByIdWithRole(messageId, 'user');

		if (!result) {
			result = this.getMessageByIdWithRole(messageId, 'system');
		}

		if (!result) return;
		const { message: msg } = result;

		try {
			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);
			const isFirstUserMessage =
				msg.role === 'user' && rootMessage && msg.parent === rootMessage.id;

			const parentId = msg.parent || rootMessage?.id;
			if (!parentId) return;

			const extrasToUse =
				newExtras !== undefined
					? JSON.parse(JSON.stringify(newExtras))
					: msg.extra
						? JSON.parse(JSON.stringify(msg.extra))
						: undefined;

			const newMessage = await DatabaseService.createMessageBranch(
				{
					convId: msg.convId,
					type: msg.type,
					timestamp: Date.now(),
					role: msg.role,
					content: newContent,
					thinking: msg.thinking || '',
					toolCalls: msg.toolCalls || '',
					children: [],
					extra: extrasToUse,
					model: msg.model
				},
				parentId
			);
			await conversationsStore.updateCurrentNode(newMessage.id);
			conversationsStore.updateConversationTimestamp();

			if (isFirstUserMessage && newContent.trim()) {
				await conversationsStore.updateConversationTitleWithConfirmation(
					activeConv.id,
					newContent.trim()
				);
			}
			await conversationsStore.refreshActiveMessages();

			if (msg.role === 'user') {
				await this.generateResponseForMessage(newMessage.id);
			}
		} catch (error) {
			console.error('Failed to edit message with branching:', error);
		}
	}

	private async generateResponseForMessage(userMessageId: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;

		if (!activeConv) return;

		this.store.showErrorDialog(null);
		this.store.setChatLoading(activeConv.id, true);
		this.store.clearChatStreaming(activeConv.id);

		try {
			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const conversationPath = filterByLeafNodeId(
				allMessages,
				userMessageId,
				false
			) as DatabaseMessage[];
			const assistantMessage = await DatabaseService.createMessageBranch(
				{
					convId: activeConv.id,
					type: 'text',
					timestamp: Date.now(),
					role: 'assistant',
					content: '',
					thinking: '',
					toolCalls: '',
					children: [],
					model: null
				},
				userMessageId
			);
			conversationsStore.addMessageToActive(assistantMessage);
			await this.streamChatCompletion(conversationPath, assistantMessage);
		} catch (error) {
			console.error('Failed to generate response:', error);
			this.store.setChatLoading(activeConv.id, false);
		}
	}

	/**
	 *
	 *
	 * Processing State
	 *
	 *
	 */

	/**
	 * Gets the total context size for the current model.
	 * Priority: active state > router model > server props > default.
	 */
	private getContextTotal(): number {
		const activeConvId = this.store.getActiveConversationId();
		const activeState = activeConvId ? this.store.getProcessingState(activeConvId) : null;

		if (activeState && activeState.contextTotal > 0) {
			return activeState.contextTotal;
		}

		if (isRouterMode()) {
			const modelContextSize = selectedModelContextSize();
			if (modelContextSize && modelContextSize > 0) {
				return modelContextSize;
			}
		}

		const propsContextSize = contextSize();
		if (propsContextSize && propsContextSize > 0) {
			return propsContextSize;
		}

		return DEFAULT_CONTEXT;
	}

	/**
	 * Updates processing state from streaming timing data.
	 * Called during streaming to update tokens/sec, context usage, etc.
	 * @param timingData - Timing information from the streaming response
	 * @param conversationId - Optional conversation ID (defaults to active)
	 */
	updateProcessingStateFromTimings(
		timingData: {
			prompt_n: number;
			prompt_ms?: number;
			predicted_n: number;
			predicted_per_second: number;
			cache_n: number;
			prompt_progress?: ChatMessagePromptProgress;
		},
		conversationId?: string
	): void {
		const processingState = this.parseTimingData(timingData);

		if (processingState === null) {
			console.warn('Failed to parse timing data - skipping update');
			return;
		}

		const targetId = conversationId || this.store.getActiveConversationId();
		if (targetId) {
			this.store.setProcessingState(targetId, processingState);
		}
	}

	private parseTimingData(timingData: Record<string, unknown>): ApiProcessingState | null {
		const promptTokens = (timingData.prompt_n as number) || 0;
		const promptMs = (timingData.prompt_ms as number) || undefined;
		const predictedTokens = (timingData.predicted_n as number) || 0;
		const tokensPerSecond = (timingData.predicted_per_second as number) || 0;
		const cacheTokens = (timingData.cache_n as number) || 0;
		const promptProgress = timingData.prompt_progress as
			| {
					total: number;
					cache: number;
					processed: number;
					time_ms: number;
			  }
			| undefined;

		const contextTotal = this.getContextTotal();
		const currentConfig = config();
		const outputTokensMax = currentConfig.max_tokens || -1;

		const contextUsed = promptTokens + cacheTokens + predictedTokens;
		const outputTokensUsed = predictedTokens;

		const progressCache = promptProgress?.cache || 0;
		const progressActualDone = (promptProgress?.processed ?? 0) - progressCache;
		const progressActualTotal = (promptProgress?.total ?? 0) - progressCache;
		const progressPercent = promptProgress
			? Math.round((progressActualDone / progressActualTotal) * 100)
			: undefined;

		return {
			status: predictedTokens > 0 ? 'generating' : promptProgress ? 'preparing' : 'idle',
			tokensDecoded: predictedTokens,
			tokensRemaining: outputTokensMax - predictedTokens,
			contextUsed,
			contextTotal,
			outputTokensUsed,
			outputTokensMax,
			hasNextToken: predictedTokens > 0,
			tokensPerSecond,
			temperature: currentConfig.temperature ?? 0.8,
			topP: currentConfig.top_p ?? 0.95,
			speculative: false,
			progressPercent,
			promptProgress,
			promptTokens,
			promptMs,
			cacheTokens
		};
	}

	/**
	 * Restores processing state from stored message timings.
	 * Used when loading a conversation to show last known stats.
	 * @param messages - Conversation messages to search for timing data
	 * @param conversationId - Conversation ID to update state for
	 */
	restoreProcessingStateFromMessages(messages: DatabaseMessage[], conversationId: string): void {
		for (let i = messages.length - 1; i >= 0; i--) {
			const message = messages[i];
			if (message.role === 'assistant' && message.timings) {
				const restoredState = this.parseTimingData({
					prompt_n: message.timings.prompt_n || 0,
					prompt_ms: message.timings.prompt_ms,
					predicted_n: message.timings.predicted_n || 0,
					predicted_per_second:
						message.timings.predicted_n && message.timings.predicted_ms
							? (message.timings.predicted_n / message.timings.predicted_ms) * 1000
							: 0,
					cache_n: message.timings.cache_n || 0
				});

				if (restoredState) {
					this.store.setProcessingState(conversationId, restoredState);
					return;
				}
			}
		}
	}

	/**
	 * Gets the model used in a conversation based on the latest assistant message.
	 */
	getConversationModel(messages: DatabaseMessage[]): string | null {
		for (let i = messages.length - 1; i >= 0; i--) {
			const message = messages[i];
			if (message.role === 'assistant' && message.model) {
				return message.model;
			}
		}
		return null;
	}

	/**
	 *
	 *
	 * Utilities
	 *
	 *
	 */

	private isAbortError(error: unknown): boolean {
		return error instanceof Error && (error.name === 'AbortError' || error instanceof DOMException);
	}

	private isChatLoading(convId: string): boolean {
		const streamingState = this.store.getChatStreaming(convId);
		return streamingState !== undefined;
	}

	private getApiOptions(): Record<string, unknown> {
		const currentConfig = config();
		const hasValue = (value: unknown): boolean =>
			value !== undefined && value !== null && value !== '';

		const apiOptions: Record<string, unknown> = { stream: true, timings_per_token: true };

		if (isRouterMode()) {
			const modelName = selectedModelName();
			if (modelName) apiOptions.model = modelName;
		}

		if (currentConfig.systemMessage) apiOptions.systemMessage = currentConfig.systemMessage;
		if (currentConfig.disableReasoningParsing) apiOptions.disableReasoningParsing = true;

		if (hasValue(currentConfig.temperature))
			apiOptions.temperature = Number(currentConfig.temperature);
		if (hasValue(currentConfig.max_tokens))
			apiOptions.max_tokens = Number(currentConfig.max_tokens);
		if (hasValue(currentConfig.dynatemp_range))
			apiOptions.dynatemp_range = Number(currentConfig.dynatemp_range);
		if (hasValue(currentConfig.dynatemp_exponent))
			apiOptions.dynatemp_exponent = Number(currentConfig.dynatemp_exponent);
		if (hasValue(currentConfig.top_k)) apiOptions.top_k = Number(currentConfig.top_k);
		if (hasValue(currentConfig.top_p)) apiOptions.top_p = Number(currentConfig.top_p);
		if (hasValue(currentConfig.min_p)) apiOptions.min_p = Number(currentConfig.min_p);
		if (hasValue(currentConfig.xtc_probability))
			apiOptions.xtc_probability = Number(currentConfig.xtc_probability);
		if (hasValue(currentConfig.xtc_threshold))
			apiOptions.xtc_threshold = Number(currentConfig.xtc_threshold);
		if (hasValue(currentConfig.typ_p)) apiOptions.typ_p = Number(currentConfig.typ_p);
		if (hasValue(currentConfig.repeat_last_n))
			apiOptions.repeat_last_n = Number(currentConfig.repeat_last_n);
		if (hasValue(currentConfig.repeat_penalty))
			apiOptions.repeat_penalty = Number(currentConfig.repeat_penalty);
		if (hasValue(currentConfig.presence_penalty))
			apiOptions.presence_penalty = Number(currentConfig.presence_penalty);
		if (hasValue(currentConfig.frequency_penalty))
			apiOptions.frequency_penalty = Number(currentConfig.frequency_penalty);
		if (hasValue(currentConfig.dry_multiplier))
			apiOptions.dry_multiplier = Number(currentConfig.dry_multiplier);
		if (hasValue(currentConfig.dry_base)) apiOptions.dry_base = Number(currentConfig.dry_base);
		if (hasValue(currentConfig.dry_allowed_length))
			apiOptions.dry_allowed_length = Number(currentConfig.dry_allowed_length);
		if (hasValue(currentConfig.dry_penalty_last_n))
			apiOptions.dry_penalty_last_n = Number(currentConfig.dry_penalty_last_n);
		if (currentConfig.samplers) apiOptions.samplers = currentConfig.samplers;
		if (currentConfig.backend_sampling)
			apiOptions.backend_sampling = currentConfig.backend_sampling;
		if (currentConfig.custom) apiOptions.custom = currentConfig.custom;

		return apiOptions;
	}
}

export const chatClient = new ChatClient();
