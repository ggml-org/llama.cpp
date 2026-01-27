/**
 * chatStore - Reactive State Store for Chat Operations
 *
 * This store contains ONLY reactive state ($state, $derived).
 * All business logic is delegated to ChatClient.
 *
 * **Architecture & Relationships:**
 * - **ChatClient**: Business logic facade (streaming, message ops, branching)
 * - **ChatService**: Stateless API layer (sendMessage)
 * - **chatStore** (this): Reactive state for UI components
 *
 * **Responsibilities:**
 * - Hold reactive state for UI binding
 * - Provide getters for computed values
 * - Expose setters for ChatClient to update state
 * - Forward method calls to ChatClient
 *
 * @see ChatClient in clients/chat/ for business logic
 * @see ChatService in services/chat.ts for API operations
 */

import { SvelteMap } from 'svelte/reactivity';
import { browser } from '$app/environment';
import { chatClient } from '$lib/clients';
import type {
	ApiProcessingState,
	ErrorDialogState,
	DatabaseMessage,
	DatabaseMessageExtra
} from '$lib/types';
import { MessageRole, MessageType } from '$lib/enums';
import {
	MAX_INACTIVE_CONVERSATION_STATES,
	INACTIVE_CONVERSATION_STATE_MAX_AGE_MS
} from '$lib/constants/cache';
import { isActiveConversation } from '$lib/stores/shared';

interface ConversationStateEntry {
	lastAccessed: number;
}

class ChatStore {
	activeProcessingState = $state<ApiProcessingState | null>(null);
	currentResponse = $state('');
	errorDialogState = $state<ErrorDialogState | null>(null);
	isLoading = $state(false);
	chatLoadingStates = new SvelteMap<string, boolean>();
	chatStreamingStates = new SvelteMap<string, { response: string; messageId: string }>();
	private abortControllers = new SvelteMap<string, AbortController>();
	private processingStates = new SvelteMap<string, ApiProcessingState | null>();

	/** Track when each conversation state was last accessed for cleanup */
	private conversationStateTimestamps = new SvelteMap<string, ConversationStateEntry>();
	private activeConversationId = $state<string | null>(null);
	private isStreamingActive = $state(false);
	private isEditModeActive = $state(false);
	private addFilesHandler: ((files: File[]) => void) | null = $state(null);
	pendingEditMessageId = $state<string | null>(null);

	/**
	 * Callback for updating message content in conversationsStore.
	 * Registered by conversationsStore to avoid circular dependency.
	 */
	private messageUpdateCallback:
		| ((messageId: string, updates: Partial<DatabaseMessage>) => void)
		| null = null;

	// Draft preservation for navigation (e.g., when adding system prompt from welcome page)
	private _pendingDraftMessage = $state<string>('');
	private _pendingDraftFiles = $state<ChatUploadedFile[]>([]);

	constructor() {
		if (browser) {
			chatClient.setStoreCallbacks({
				setChatLoading: (convId, loading) => this.setChatLoading(convId, loading),
				setChatStreaming: (convId, response, messageId) =>
					this.setChatStreaming(convId, response, messageId),
				clearChatStreaming: (convId) => this.clearChatStreaming(convId),
				getChatStreaming: (convId) => this.getChatStreaming(convId),
				setProcessingState: (convId, state) => this.setProcessingState(convId, state),
				getProcessingState: (convId) => this.getProcessingState(convId),
				setActiveProcessingConversation: (convId) => this.setActiveProcessingConversation(convId),
				setStreamingActive: (active) => this.setStreamingActive(active),
				showErrorDialog: (state) => this.showErrorDialog(state),
				getAbortController: (convId) => this.getOrCreateAbortController(convId),
				abortRequest: (convId) => this.abortRequest(convId),
				setPendingEditMessageId: (messageId) => (this.pendingEditMessageId = messageId),
				getActiveConversationId: () => this.activeConversationId,
				getCurrentResponse: () => this.currentResponse
			});
		}
	}

	private setChatLoading(convId: string, loading: boolean): void {
		this.touchConversationState(convId);
		if (loading) {
			this.chatLoadingStates.set(convId, true);
			if (isActiveConversation(convId)) this.isLoading = true;
		} else {
			this.chatLoadingStates.delete(convId);
			if (isActiveConversation(convId)) this.isLoading = false;
		}
	}

	private setChatStreaming(convId: string, response: string, messageId: string): void {
		this.touchConversationState(convId);
		this.chatStreamingStates.set(convId, { response, messageId });
		if (isActiveConversation(convId)) this.currentResponse = response;
	}

	private clearChatStreaming(convId: string): void {
		this.chatStreamingStates.delete(convId);
		if (isActiveConversation(convId)) this.currentResponse = '';
	}

	private getChatStreaming(convId: string): { response: string; messageId: string } | undefined {
		return this.chatStreamingStates.get(convId);
	}

	syncLoadingStateForChat(convId: string): void {
		this.isLoading = this.chatLoadingStates.get(convId) || false;
		const streamingState = this.chatStreamingStates.get(convId);
		this.currentResponse = streamingState?.response || '';

		// If there's an active stream for this conversation, update the message content
		// This ensures streaming content is visible when switching back to a conversation
		if (streamingState?.response && streamingState?.messageId && this.messageUpdateCallback) {
			this.messageUpdateCallback(streamingState.messageId, { content: streamingState.response });
		}
	}

	/**
	 * Register a callback for updating message content.
	 * Called by conversationsStore during initialization to avoid circular dependency.
	 */
	registerMessageUpdateCallback(
		callback: (messageId: string, updates: Partial<DatabaseMessage>) => void
	): void {
		this.messageUpdateCallback = callback;
	}

	clearUIState(): void {
		this.isLoading = false;
		this.currentResponse = '';
	}

	setActiveProcessingConversation(conversationId: string | null): void {
		this.activeConversationId = conversationId;

		if (conversationId) {
			this.activeProcessingState = this.processingStates.get(conversationId) || null;
		} else {
			this.activeProcessingState = null;
		}
	}

	getProcessingState(conversationId: string): ApiProcessingState | null {
		return this.processingStates.get(conversationId) || null;
	}

	private setProcessingState(conversationId: string, state: ApiProcessingState | null): void {
		if (state === null) {
			this.processingStates.delete(conversationId);
		} else {
			this.processingStates.set(conversationId, state);
		}

		if (conversationId === this.activeConversationId) {
			this.activeProcessingState = state;
		}
	}

	clearProcessingState(conversationId: string): void {
		this.processingStates.delete(conversationId);

		if (conversationId === this.activeConversationId) {
			this.activeProcessingState = null;
		}
	}

	getActiveProcessingState(): ApiProcessingState | null {
		return this.activeProcessingState;
	}

	getCurrentProcessingStateSync(): ApiProcessingState | null {
		return this.activeProcessingState;
	}

	private setStreamingActive(active: boolean): void {
		this.isStreamingActive = active;
	}

	isStreaming(): boolean {
		return this.isStreamingActive;
	}

	private getOrCreateAbortController(convId: string): AbortController {
		let controller = this.abortControllers.get(convId);
		if (!controller || controller.signal.aborted) {
			controller = new AbortController();
			this.abortControllers.set(convId, controller);
		}
		return controller;
	}

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

	private showErrorDialog(state: ErrorDialogState | null): void {
		this.errorDialogState = state;
	}

	dismissErrorDialog(): void {
		this.errorDialogState = null;
	}

	clearEditMode(): void {
		this.isEditModeActive = false;
		this.addFilesHandler = null;
	}

	isEditing(): boolean {
		return this.isEditModeActive;
	}

	setEditModeActive(handler: (files: File[]) => void): void {
		this.isEditModeActive = true;
		this.addFilesHandler = handler;
	}

	getAddFilesHandler(): ((files: File[]) => void) | null {
		return this.addFilesHandler;
	}

	clearPendingEditMessageId(): void {
		this.pendingEditMessageId = null;
	}

	savePendingDraft(message: string, files: ChatUploadedFile[]): void {
		this._pendingDraftMessage = message;
		this._pendingDraftFiles = [...files];
	}

	consumePendingDraft(): { message: string; files: ChatUploadedFile[] } | null {
		if (!this._pendingDraftMessage && this._pendingDraftFiles.length === 0) {
			return null;
		}

		const draft = {
			message: this._pendingDraftMessage,
			files: [...this._pendingDraftFiles]
		};

		this._pendingDraftMessage = '';
		this._pendingDraftFiles = [];

		return draft;
	}

	hasPendingDraft(): boolean {
		return Boolean(this._pendingDraftMessage) || this._pendingDraftFiles.length > 0;
	}

	getAllLoadingChats(): string[] {
		return Array.from(this.chatLoadingStates.keys());
	}

	getAllStreamingChats(): string[] {
		return Array.from(this.chatStreamingStates.keys());
	}

	getChatStreamingPublic(convId: string): { response: string; messageId: string } | undefined {
		return this.getChatStreaming(convId);
	}

	isChatLoadingPublic(convId: string): boolean {
		return this.chatLoadingStates.get(convId) || false;
	}

	/**
	 * Update last accessed timestamp for a conversation.
	 */
	private touchConversationState(convId: string): void {
		this.conversationStateTimestamps.set(convId, { lastAccessed: Date.now() });
	}

	/**
	 * Clean up states for old/inactive conversations to prevent memory bloat.
	 * This removes loading states, streaming states, abort controllers, and processing states
	 * for conversations that haven't been accessed recently.
	 *
	 * @param activeConversationIds - Set of conversation IDs that should not be cleaned up
	 * @returns Number of conversation states cleaned up
	 */
	cleanupOldConversationStates(activeConversationIds?: string[]): number {
		const now = Date.now();
		const activeIdsList = activeConversationIds ?? [];

		// Always preserve the currently active conversation
		const preserveIds = this.activeConversationId
			? [...activeIdsList, this.activeConversationId]
			: activeIdsList;

		// Collect all unique conversation IDs that have any state
		const allConvIdsArray = [
			...this.chatLoadingStates.keys(),
			...this.chatStreamingStates.keys(),
			...this.abortControllers.keys(),
			...this.processingStates.keys(),
			...this.conversationStateTimestamps.keys()
		];
		const allConvIds = [...new Map(allConvIdsArray.map((id) => [id, true])).keys()];

		// Filter to candidates for cleanup (not active, not currently loading/streaming)
		const cleanupCandidates: Array<{ convId: string; lastAccessed: number }> = [];

		for (const convId of allConvIds) {
			// Never clean up active conversations or those with active streams
			if (preserveIds.includes(convId)) continue;
			if (this.chatLoadingStates.get(convId)) continue;
			if (this.chatStreamingStates.has(convId)) continue;

			const timestamp = this.conversationStateTimestamps.get(convId);
			const lastAccessed = timestamp?.lastAccessed ?? 0;

			cleanupCandidates.push({ convId, lastAccessed });
		}

		// Sort by last accessed (oldest first)
		cleanupCandidates.sort((a, b) => a.lastAccessed - b.lastAccessed);

		let cleanedUp = 0;

		for (const { convId, lastAccessed } of cleanupCandidates) {
			// Clean up if:
			// 1. Too many inactive states, OR
			// 2. State is older than max age
			const tooManyStates = cleanupCandidates.length - cleanedUp > MAX_INACTIVE_CONVERSATION_STATES;
			const tooOld = now - lastAccessed > INACTIVE_CONVERSATION_STATE_MAX_AGE_MS;

			if (tooManyStates || tooOld) {
				this.cleanupConversationState(convId);
				cleanedUp++;
			}
		}

		return cleanedUp;
	}

	/**
	 * Clean up all state for a specific conversation.
	 */
	private cleanupConversationState(convId: string): void {
		// Abort any pending request
		const controller = this.abortControllers.get(convId);
		if (controller && !controller.signal.aborted) {
			controller.abort();
		}

		// Remove all state
		this.chatLoadingStates.delete(convId);
		this.chatStreamingStates.delete(convId);
		this.abortControllers.delete(convId);
		this.processingStates.delete(convId);
		this.conversationStateTimestamps.delete(convId);
	}

	/**
	 * Get the number of tracked conversation states (for debugging/monitoring).
	 */
	getTrackedConversationCount(): number {
		return new Set([
			...this.chatLoadingStates.keys(),
			...this.chatStreamingStates.keys(),
			...this.abortControllers.keys(),
			...this.processingStates.keys()
		]).size;
	}

	async addMessage(
		role: MessageRole,
		content: string,
		type: MessageType = MessageType.TEXT,
		parent: string = '-1',
		extras?: DatabaseMessageExtra[]
	): Promise<DatabaseMessage> {
		return chatClient.addMessage(role, content, type, parent, extras);
	}

	async addSystemPrompt(): Promise<void> {
		return chatClient.addSystemPrompt();
	}

	async removeSystemPromptPlaceholder(messageId: string): Promise<boolean> {
		return chatClient.removeSystemPromptPlaceholder(messageId);
	}

	async sendMessage(content: string, extras?: DatabaseMessageExtra[]): Promise<void> {
		return chatClient.sendMessage(content, extras);
	}

	async stopGeneration(): Promise<void> {
		return chatClient.stopGeneration();
	}

	async stopGenerationForChat(convId: string): Promise<void> {
		return chatClient.stopGenerationForChat(convId);
	}

	async updateMessage(messageId: string, newContent: string): Promise<void> {
		return chatClient.updateMessage(messageId, newContent);
	}

	async regenerateMessage(messageId: string): Promise<void> {
		return chatClient.regenerateMessage(messageId);
	}

	async regenerateMessageWithBranching(messageId: string, modelOverride?: string): Promise<void> {
		return chatClient.regenerateMessageWithBranching(messageId, modelOverride);
	}

	async getDeletionInfo(messageId: string): Promise<{
		totalCount: number;
		userMessages: number;
		assistantMessages: number;
		messageTypes: string[];
	}> {
		return chatClient.getDeletionInfo(messageId);
	}

	async deleteMessage(messageId: string): Promise<void> {
		return chatClient.deleteMessage(messageId);
	}

	async continueAssistantMessage(messageId: string): Promise<void> {
		return chatClient.continueAssistantMessage(messageId);
	}

	async editAssistantMessage(
		messageId: string,
		newContent: string,
		shouldBranch: boolean
	): Promise<void> {
		return chatClient.editAssistantMessage(messageId, newContent, shouldBranch);
	}

	async editUserMessagePreserveResponses(
		messageId: string,
		newContent: string,
		newExtras?: DatabaseMessageExtra[]
	): Promise<void> {
		return chatClient.editUserMessagePreserveResponses(messageId, newContent, newExtras);
	}

	async editMessageWithBranching(
		messageId: string,
		newContent: string,
		newExtras?: DatabaseMessageExtra[]
	): Promise<void> {
		return chatClient.editMessageWithBranching(messageId, newContent, newExtras);
	}

	updateProcessingStateFromTimings(
		timingData: {
			prompt_n: number;
			prompt_ms?: number;
			predicted_n: number;
			predicted_per_second: number;
			cache_n: number;
			prompt_progress?: {
				total: number;
				cache: number;
				processed: number;
				time_ms: number;
			};
		},
		conversationId?: string
	): void {
		chatClient.updateProcessingStateFromTimings(timingData, conversationId);
	}

	restoreProcessingStateFromMessages(messages: DatabaseMessage[], conversationId: string): void {
		chatClient.restoreProcessingStateFromMessages(messages, conversationId);
	}

	getConversationModel(messages: DatabaseMessage[]): string | null {
		return chatClient.getConversationModel(messages);
	}
}

export const chatStore = new ChatStore();

// State access functions (getters only - use chatStore.method() for actions)
export const activeProcessingState = () => chatStore.activeProcessingState;
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
export const pendingEditMessageId = () => chatStore.pendingEditMessageId;
