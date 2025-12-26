import { DatabaseService, ChatService } from '$lib/services';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import { config } from '$lib/stores/settings.svelte';
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
import { SvelteMap } from 'svelte/reactivity';
import { DEFAULT_CONTEXT } from '$lib/constants/default-context';

/**
 * chatStore - Active AI interaction and streaming state management
 *
 * **Terminology - Chat vs Conversation:**
 * - **Chat**: The active interaction space with the Chat Completions API. Represents the
 *   real-time streaming session, loading states, and UI visualization of AI communication.
 *   A "chat" is ephemeral - it exists only while the user is actively interacting with the AI.
 * - **Conversation**: The persistent database entity storing all messages and metadata.
 *   Managed by conversationsStore, conversations persist across sessions and page reloads.
 *
 * This store manages all active AI interactions including real-time streaming, response
 * generation, and per-chat loading states. It handles the runtime layer between UI and
 * AI backend, supporting concurrent streaming across multiple conversations.
 *
 * **Architecture & Relationships:**
 * - **chatStore** (this class): Active AI session and streaming management
 *   - Manages real-time AI response streaming via ChatService
 *   - Tracks per-chat loading and streaming states for concurrent sessions
 *   - Handles message operations (send, edit, regenerate, branch)
 *   - Coordinates with conversationsStore for persistence
 *
 * - **conversationsStore**: Provides conversation data and message arrays for chat context
 * - **ChatService**: Low-level API communication with llama.cpp server
 * - **DatabaseService**: Message persistence and retrieval
 *
 * **Key Features:**
 * - **AI Streaming**: Real-time token streaming with abort support
 * - **Concurrent Chats**: Independent loading/streaming states per conversation
 * - **Message Branching**: Edit, regenerate, and branch conversation trees
 * - **Error Handling**: Timeout and server error recovery with user feedback
 * - **Graceful Stop**: Save partial responses when stopping generation
 *
 * **State Management:**
 * - Global `isLoading` and `currentResponse` for active chat UI
 * - `chatLoadingStates` Map for per-conversation streaming tracking
 * - `chatStreamingStates` Map for per-conversation streaming content
 * - `processingStates` Map for per-conversation processing state (timing/context info)
 * - Automatic state sync when switching between conversations
 */
class ChatStore {
	// ─────────────────────────────────────────────────────────────────────────────
	// State
	// ─────────────────────────────────────────────────────────────────────────────

	activeProcessingState = $state<ApiProcessingState | null>(null);
	currentResponse = $state('');
	errorDialogState = $state<{
		type: 'timeout' | 'server';
		message: string;
		contextInfo?: { n_prompt_tokens: number; n_ctx: number };
	} | null>(null);
	isLoading = $state(false);
	chatLoadingStates = new SvelteMap<string, boolean>();
	chatStreamingStates = new SvelteMap<string, { response: string; messageId: string }>();
	private abortControllers = new SvelteMap<string, AbortController>();
	private processingStates = new SvelteMap<string, ApiProcessingState | null>();
	private activeConversationId = $state<string | null>(null);
	private isStreamingActive = $state(false);
	private isEditModeActive = $state(false);
	private addFilesHandler: ((files: File[]) => void) | null = $state(null);

	// ─────────────────────────────────────────────────────────────────────────────
	// Loading State
	// ─────────────────────────────────────────────────────────────────────────────

	private setChatLoading(convId: string, loading: boolean): void {
		if (loading) {
			this.chatLoadingStates.set(convId, true);
			if (conversationsStore.activeConversation?.id === convId) this.isLoading = true;
		} else {
			this.chatLoadingStates.delete(convId);
			if (conversationsStore.activeConversation?.id === convId) this.isLoading = false;
		}
	}

	private isChatLoading(convId: string): boolean {
		return this.chatLoadingStates.get(convId) || false;
	}

	private setChatStreaming(convId: string, response: string, messageId: string): void {
		this.chatStreamingStates.set(convId, { response, messageId });
		if (conversationsStore.activeConversation?.id === convId) this.currentResponse = response;
	}

	private clearChatStreaming(convId: string): void {
		this.chatStreamingStates.delete(convId);
		if (conversationsStore.activeConversation?.id === convId) this.currentResponse = '';
	}

	private getChatStreaming(convId: string): { response: string; messageId: string } | undefined {
		return this.chatStreamingStates.get(convId);
	}

	syncLoadingStateForChat(convId: string): void {
		this.isLoading = this.isChatLoading(convId);
		const streamingState = this.getChatStreaming(convId);
		this.currentResponse = streamingState?.response || '';
	}

	/**
	 * Clears global UI state without affecting background streaming.
	 * Used when navigating to empty/new chat while other chats stream in background.
	 */
	clearUIState(): void {
		this.isLoading = false;
		this.currentResponse = '';
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Processing State
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Set the active conversation for statistics display
	 */
	setActiveProcessingConversation(conversationId: string | null): void {
		this.activeConversationId = conversationId;

		if (conversationId) {
			this.activeProcessingState = this.processingStates.get(conversationId) || null;
		} else {
			this.activeProcessingState = null;
		}
	}

	/**
	 * Get processing state for a specific conversation
	 */
	getProcessingState(conversationId: string): ApiProcessingState | null {
		return this.processingStates.get(conversationId) || null;
	}

	/**
	 * Clear processing state for a specific conversation
	 */
	clearProcessingState(conversationId: string): void {
		this.processingStates.delete(conversationId);

		if (conversationId === this.activeConversationId) {
			this.activeProcessingState = null;
		}
	}

	/**
	 * Get the current processing state for the active conversation (reactive)
	 * Returns the direct reactive state for UI binding
	 */
	getActiveProcessingState(): ApiProcessingState | null {
		return this.activeProcessingState;
	}

	/**
	 * Updates processing state with timing data from streaming response
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

		const targetId = conversationId || this.activeConversationId;
		if (targetId) {
			this.processingStates.set(targetId, processingState);

			if (targetId === this.activeConversationId) {
				this.activeProcessingState = processingState;
			}
		}
	}

	/**
	 * Get current processing state (sync version for reactive access)
	 */
	getCurrentProcessingStateSync(): ApiProcessingState | null {
		return this.activeProcessingState;
	}

	/**
	 * Restore processing state from last assistant message timings
	 * Call this when keepStatsVisible is enabled and we need to show last known stats
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
					this.processingStates.set(conversationId, restoredState);

					if (conversationId === this.activeConversationId) {
						this.activeProcessingState = restoredState;
					}

					return;
				}
			}
		}
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Streaming
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Start streaming session tracking
	 */
	startStreaming(): void {
		this.isStreamingActive = true;
	}

	/**
	 * Stop streaming session tracking
	 */
	stopStreaming(): void {
		this.isStreamingActive = false;
	}

	/**
	 * Check if currently in a streaming session
	 */
	isStreaming(): boolean {
		return this.isStreamingActive;
	}

	private getContextTotal(): number {
		const activeState = this.getActiveProcessingState();

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

		const progressPercent = promptProgress
			? Math.round((promptProgress.processed / promptProgress.total) * 100)
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
			promptTokens,
			promptMs,
			cacheTokens
		};
	}

	/**
	 * Gets the model used in a conversation based on the latest assistant message.
	 * Returns the model from the most recent assistant message that has a model field set.
	 *
	 * @param messages - Array of messages to search through
	 * @returns The model name or null if no model found
	 */
	getConversationModel(messages: DatabaseMessage[]): string | null {
		// Search backwards through messages to find most recent assistant message with model
		for (let i = messages.length - 1; i >= 0; i--) {
			const message = messages[i];
			if (message.role === 'assistant' && message.model) {
				return message.model;
			}
		}
		return null;
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Error Handling
	// ─────────────────────────────────────────────────────────────────────────────

	private isAbortError(error: unknown): boolean {
		return error instanceof Error && (error.name === 'AbortError' || error instanceof DOMException);
	}

	private showErrorDialog(
		type: 'timeout' | 'server',
		message: string,
		contextInfo?: { n_prompt_tokens: number; n_ctx: number }
	): void {
		this.errorDialogState = { type, message, contextInfo };
	}

	dismissErrorDialog(): void {
		this.errorDialogState = null;
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Message Operations
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Finds a message by ID and optionally validates its role.
	 * Returns message and index, or null if not found or role doesn't match.
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

	private async streamChatCompletion(
		allMessages: DatabaseMessage[],
		assistantMessage: DatabaseMessage,
		onComplete?: (content: string) => Promise<void>,
		onError?: (error: Error) => void,
		modelOverride?: string | null
	): Promise<void> {
		// Ensure model props are cached before streaming (for correct n_ctx in processing info)
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

		this.startStreaming();
		this.setActiveProcessingConversation(assistantMessage.convId);

		const abortController = this.getOrCreateAbortController(assistantMessage.convId);

		await ChatService.sendMessage(
			allMessages,
			{
				...this.getApiOptions(),
				...(modelOverride ? { model: modelOverride } : {}),
				onChunk: (chunk: string) => {
					streamedContent += chunk;
					this.setChatStreaming(assistantMessage.convId, streamedContent, assistantMessage.id);
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
				onModel: (modelName: string) => recordModel(modelName),
				onTimings: (timings: ChatMessageTimings, promptProgress?: ChatMessagePromptProgress) => {
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
					this.stopStreaming();

					const updateData: Record<string, unknown> = {
						content: finalContent || streamedContent,
						thinking: reasoningContent || streamedReasoningContent,
						toolCalls: toolCallContent || streamedToolCallContent,
						timings
					};
					if (resolvedModel && !modelPersisted) {
						updateData.model = resolvedModel;
					}
					await DatabaseService.updateMessage(assistantMessage.id, updateData);

					const idx = conversationsStore.findMessageIndex(assistantMessage.id);
					const uiUpdate: Partial<DatabaseMessage> = {
						content: updateData.content as string,
						toolCalls: updateData.toolCalls as string
					};
					if (timings) uiUpdate.timings = timings;
					if (resolvedModel) uiUpdate.model = resolvedModel;

					conversationsStore.updateMessageAtIndex(idx, uiUpdate);
					await conversationsStore.updateCurrentNode(assistantMessage.id);

					if (onComplete) await onComplete(streamedContent);
					this.setChatLoading(assistantMessage.convId, false);
					this.clearChatStreaming(assistantMessage.convId);
					this.clearProcessingState(assistantMessage.convId);

					if (isRouterMode()) {
						modelsStore.fetchRouterModels().catch(console.error);
					}
				},
				onError: (error: Error) => {
					this.stopStreaming();

					if (this.isAbortError(error)) {
						this.setChatLoading(assistantMessage.convId, false);
						this.clearChatStreaming(assistantMessage.convId);
						this.clearProcessingState(assistantMessage.convId);

						return;
					}

					console.error('Streaming error:', error);

					this.setChatLoading(assistantMessage.convId, false);
					this.clearChatStreaming(assistantMessage.convId);
					this.clearProcessingState(assistantMessage.convId);

					const idx = conversationsStore.findMessageIndex(assistantMessage.id);

					if (idx !== -1) {
						const failedMessage = conversationsStore.removeMessageAtIndex(idx);
						if (failedMessage) DatabaseService.deleteMessage(failedMessage.id).catch(console.error);
					}

					const contextInfo = (
						error as Error & { contextInfo?: { n_prompt_tokens: number; n_ctx: number } }
					).contextInfo;

					this.showErrorDialog(
						error.name === 'TimeoutError' ? 'timeout' : 'server',
						error.message,
						contextInfo
					);

					if (onError) onError(error);
				}
			},
			assistantMessage.convId,
			abortController.signal
		);
	}

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

		this.errorDialogState = null;
		this.setChatLoading(currentConv.id, true);
		this.clearChatStreaming(currentConv.id);

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
				this.setChatLoading(currentConv.id, false);
				return;
			}
			console.error('Failed to send message:', error);
			this.setChatLoading(currentConv.id, false);
			if (!this.errorDialogState) {
				const dialogType =
					error instanceof Error && error.name === 'TimeoutError' ? 'timeout' : 'server';
				const contextInfo = (
					error as Error & { contextInfo?: { n_prompt_tokens: number; n_ctx: number } }
				).contextInfo;

				this.showErrorDialog(
					dialogType,
					error instanceof Error ? error.message : 'Unknown error',
					contextInfo
				);
			}
		}
	}

	async stopGeneration(): Promise<void> {
		const activeConv = conversationsStore.activeConversation;

		if (!activeConv) return;

		await this.stopGenerationForChat(activeConv.id);
	}

	async stopGenerationForChat(convId: string): Promise<void> {
		await this.savePartialResponseIfNeeded(convId);

		this.stopStreaming();
		this.abortRequest(convId);
		this.setChatLoading(convId, false);
		this.clearChatStreaming(convId);
		this.clearProcessingState(convId);
	}

	/**
	 * Gets or creates an AbortController for a conversation
	 */
	private getOrCreateAbortController(convId: string): AbortController {
		let controller = this.abortControllers.get(convId);
		if (!controller || controller.signal.aborted) {
			controller = new AbortController();
			this.abortControllers.set(convId, controller);
		}
		return controller;
	}

	/**
	 * Aborts any ongoing request for a conversation
	 */
	private abortRequest(convId?: string): void {
		if (convId) {
			const controller = this.abortControllers.get(convId);
			if (controller) {
				controller.abort();
				this.abortControllers.delete(convId);
			}
		} else {
			for (const controller of this.abortControllers.values()) {
				controller.abort();
			}
			this.abortControllers.clear();
		}
	}

	private async savePartialResponseIfNeeded(convId?: string): Promise<void> {
		const conversationId = convId || conversationsStore.activeConversation?.id;

		if (!conversationId) return;

		const streamingState = this.chatStreamingStates.get(conversationId);

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
				const lastKnownState = this.getProcessingState(conversationId);
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

				lastMessage.content = this.currentResponse;

				if (updateData.thinking) lastMessage.thinking = updateData.thinking;

				if (updateData.timings) lastMessage.timings = updateData.timings;
			} catch (error) {
				lastMessage.content = this.currentResponse;
				console.error('Failed to save partial response:', error);
			}
		}
	}

	async updateMessage(messageId: string, newContent: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv) return;
		if (this.isLoading) this.stopGeneration();

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
					newContent.trim(),
					conversationsStore.titleUpdateConfirmationCallback
				);
			}

			const messagesToRemove = conversationsStore.activeMessages.slice(messageIndex + 1);

			for (const message of messagesToRemove) await DatabaseService.deleteMessage(message.id);

			conversationsStore.sliceActiveMessages(messageIndex + 1);
			conversationsStore.updateConversationTimestamp();

			this.setChatLoading(activeConv.id, true);
			this.clearChatStreaming(activeConv.id);

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

	// ─────────────────────────────────────────────────────────────────────────────
	// Regeneration
	// ─────────────────────────────────────────────────────────────────────────────

	async regenerateMessage(messageId: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv || this.isLoading) return;

		const result = this.getMessageByIdWithRole(messageId, 'assistant');
		if (!result) return;
		const { index: messageIndex } = result;

		try {
			const messagesToRemove = conversationsStore.activeMessages.slice(messageIndex);
			for (const message of messagesToRemove) await DatabaseService.deleteMessage(message.id);
			conversationsStore.sliceActiveMessages(messageIndex);
			conversationsStore.updateConversationTimestamp();

			this.setChatLoading(activeConv.id, true);
			this.clearChatStreaming(activeConv.id);

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
			this.setChatLoading(activeConv?.id || '', false);
		}
	}

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

	// ─────────────────────────────────────────────────────────────────────────────
	// Editing
	// ─────────────────────────────────────────────────────────────────────────────

	clearEditMode(): void {
		this.isEditModeActive = false;
		this.addFilesHandler = null;
	}

	async continueAssistantMessage(messageId: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv || this.isLoading) return;

		const result = this.getMessageByIdWithRole(messageId, 'assistant');
		if (!result) return;
		const { message: msg, index: idx } = result;

		if (this.isChatLoading(activeConv.id)) return;

		try {
			this.errorDialogState = null;
			this.setChatLoading(activeConv.id, true);
			this.clearChatStreaming(activeConv.id);

			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const dbMessage = allMessages.find((m) => m.id === messageId);

			if (!dbMessage) {
				this.setChatLoading(activeConv.id, false);

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

			const abortController = this.getOrCreateAbortController(msg.convId);

			await ChatService.sendMessage(
				contextWithContinue,
				{
					...this.getApiOptions(),

					onChunk: (chunk: string) => {
						hasReceivedContent = true;
						appendedContent += chunk;
						const fullContent = originalContent + appendedContent;
						this.setChatStreaming(msg.convId, fullContent, msg.id);
						conversationsStore.updateMessageAtIndex(idx, { content: fullContent });
					},

					onReasoningChunk: (reasoningChunk: string) => {
						hasReceivedContent = true;
						appendedThinking += reasoningChunk;
						conversationsStore.updateMessageAtIndex(idx, {
							thinking: originalThinking + appendedThinking
						});
					},

					onTimings: (timings: ChatMessageTimings, promptProgress?: ChatMessagePromptProgress) => {
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
						this.setChatLoading(msg.convId, false);
						this.clearChatStreaming(msg.convId);
						this.clearProcessingState(msg.convId);
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
							this.setChatLoading(msg.convId, false);
							this.clearChatStreaming(msg.convId);
							this.clearProcessingState(msg.convId);
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
						this.setChatLoading(msg.convId, false);
						this.clearChatStreaming(msg.convId);
						this.clearProcessingState(msg.convId);
						this.showErrorDialog(
							error.name === 'TimeoutError' ? 'timeout' : 'server',
							error.message
						);
					}
				},
				msg.convId,
				abortController.signal
			);
		} catch (error) {
			if (!this.isAbortError(error)) console.error('Failed to continue message:', error);
			if (activeConv) this.setChatLoading(activeConv.id, false);
		}
	}

	async editAssistantMessage(
		messageId: string,
		newContent: string,
		shouldBranch: boolean
	): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv || this.isLoading) return;

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

			// Update extras if provided (including empty array to clear attachments)
			// Deep clone to avoid Proxy objects from Svelte reactivity
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
					newContent.trim(),
					conversationsStore.titleUpdateConfirmationCallback
				);
			}
			conversationsStore.updateConversationTimestamp();
		} catch (error) {
			console.error('Failed to edit user message:', error);
		}
	}

	async editMessageWithBranching(
		messageId: string,
		newContent: string,
		newExtras?: DatabaseMessageExtra[]
	): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv || this.isLoading) return;

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

			// Use newExtras if provided, otherwise copy existing extras
			// Deep clone to avoid Proxy objects from Svelte reactivity
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
					newContent.trim(),
					conversationsStore.titleUpdateConfirmationCallback
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

	async regenerateMessageWithBranching(messageId: string, modelOverride?: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv || this.isLoading) return;
		try {
			const idx = conversationsStore.findMessageIndex(messageId);
			if (idx === -1) return;
			const msg = conversationsStore.activeMessages[idx];
			if (msg.role !== 'assistant') return;

			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const parentMessage = allMessages.find((m) => m.id === msg.parent);
			if (!parentMessage) return;

			this.setChatLoading(activeConv.id, true);
			this.clearChatStreaming(activeConv.id);

			const newAssistantMessage = await DatabaseService.createMessageBranch(
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
			// Use modelOverride if provided, otherwise use the original message's model
			// If neither is available, don't pass model (will use global selection)
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
			this.setChatLoading(activeConv?.id || '', false);
		}
	}

	private async generateResponseForMessage(userMessageId: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;

		if (!activeConv) return;

		this.errorDialogState = null;
		this.setChatLoading(activeConv.id, true);
		this.clearChatStreaming(activeConv.id);

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
			this.setChatLoading(activeConv.id, false);
		}
	}

	getAddFilesHandler(): ((files: File[]) => void) | null {
		return this.addFilesHandler;
	}

	public getAllLoadingChats(): string[] {
		return Array.from(this.chatLoadingStates.keys());
	}

	public getAllStreamingChats(): string[] {
		return Array.from(this.chatStreamingStates.keys());
	}

	public getChatStreamingPublic(
		convId: string
	): { response: string; messageId: string } | undefined {
		return this.getChatStreaming(convId);
	}

	public isChatLoadingPublic(convId: string): boolean {
		return this.isChatLoading(convId);
	}

	isEditing(): boolean {
		return this.isEditModeActive;
	}

	setEditModeActive(handler: (files: File[]) => void): void {
		this.isEditModeActive = true;
		this.addFilesHandler = handler;
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Utilities
	// ─────────────────────────────────────────────────────────────────────────────

	private getApiOptions(): Record<string, unknown> {
		const currentConfig = config();
		const hasValue = (value: unknown): boolean =>
			value !== undefined && value !== null && value !== '';

		const apiOptions: Record<string, unknown> = { stream: true, timings_per_token: true };

		// Model selection (required in ROUTER mode)
		if (isRouterMode()) {
			const modelName = selectedModelName();
			if (modelName) apiOptions.model = modelName;
		}

		// Config options needed by ChatService
		if (currentConfig.systemMessage) apiOptions.systemMessage = currentConfig.systemMessage;
		if (currentConfig.disableReasoningFormat) apiOptions.disableReasoningFormat = true;

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
		if (currentConfig.custom) apiOptions.custom = currentConfig.custom;

		return apiOptions;
	}
}

export const chatStore = new ChatStore();

export const activeProcessingState = () => chatStore.activeProcessingState;
export const clearEditMode = () => chatStore.clearEditMode();
export const currentResponse = () => chatStore.currentResponse;
export const errorDialog = () => chatStore.errorDialogState;
export const getAddFilesHandler = () => chatStore.getAddFilesHandler();
export const getAllLoadingChats = () => chatStore.getAllLoadingChats();
export const getAllStreamingChats = () => chatStore.getAllStreamingChats();
export const getChatStreaming = (convId: string) => chatStore.getChatStreamingPublic(convId);
export const isChatLoading = (convId: string) => chatStore.isChatLoadingPublic(convId);
export const isChatStreaming = () => chatStore.isStreaming();
export const isEditing = () => chatStore.isEditing();
export const isLoading = () => chatStore.isLoading;
export const setEditModeActive = (handler: (files: File[]) => void) =>
	chatStore.setEditModeActive(handler);
