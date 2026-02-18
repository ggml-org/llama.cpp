/**
 * chatStore - Reactive State Store for Chat Operations
 *
 * Manages chat lifecycle, streaming, message operations, and processing state.
 *
 * **Architecture & Relationships:**
 * - **ChatService**: Stateless API layer (sendMessage, streaming)
 * - **chatStore** (this): Reactive state + business logic
 * - **conversationsStore**: Conversation persistence and navigation
 *
 * @see ChatService in services/chat.service.ts for API operations
 */

import { SvelteMap } from 'svelte/reactivity';
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
	findLeafNode,
	isAbortError
} from '$lib/utils';
import { SYSTEM_MESSAGE_PLACEHOLDER } from '$lib/constants/ui';
import { REASONING_TAGS } from '$lib/constants/agentic';
import {
	MAX_INACTIVE_CONVERSATION_STATES,
	INACTIVE_CONVERSATION_STATE_MAX_AGE_MS
} from '$lib/constants/cache';
import type {
	ChatMessageTimings,
	ChatMessagePromptProgress,
	ChatStreamCallbacks,
	ErrorDialogState
} from '$lib/types/chat';
import type { ApiProcessingState, DatabaseMessage, DatabaseMessageExtra } from '$lib/types';
import { ErrorDialogType, MessageRole, MessageType } from '$lib/enums';

interface ConversationStateEntry {
	lastAccessed: number;
}

const countOccurrences = (source: string, token: string): number =>
	source ? source.split(token).length - 1 : 0;
const hasUnclosedReasoningTag = (content: string): boolean =>
	countOccurrences(content, REASONING_TAGS.START) > countOccurrences(content, REASONING_TAGS.END);
const wrapReasoningContent = (content: string, reasoningContent?: string): string => {
	if (!reasoningContent) return content;
	return `${REASONING_TAGS.START}${reasoningContent}${REASONING_TAGS.END}${content}`;
};

class ChatStore {
	activeProcessingState = $state<ApiProcessingState | null>(null);
	currentResponse = $state('');
	errorDialogState = $state<ErrorDialogState | null>(null);
	isLoading = $state(false);
	chatLoadingStates = new SvelteMap<string, boolean>();
	chatStreamingStates = new SvelteMap<string, { response: string; messageId: string }>();
	private abortControllers = new SvelteMap<string, AbortController>();
	private processingStates = new SvelteMap<string, ApiProcessingState | null>();
	private conversationStateTimestamps = new SvelteMap<string, ConversationStateEntry>();
	private activeConversationId = $state<string | null>(null);
	private isStreamingActive = $state(false);
	private isEditModeActive = $state(false);
	private addFilesHandler: ((files: File[]) => void) | null = $state(null);
	pendingEditMessageId = $state<string | null>(null);
	private messageUpdateCallback:
		| ((messageId: string, updates: Partial<DatabaseMessage>) => void)
		| null = null;
	private _pendingDraftMessage = $state<string>('');
	private _pendingDraftFiles = $state<ChatUploadedFile[]>([]);

	private setChatLoading(convId: string, loading: boolean): void {
		this.touchConversationState(convId);
		if (loading) {
			this.chatLoadingStates.set(convId, true);
			if (convId === conversationsStore.activeConversation?.id) this.isLoading = true;
		} else {
			this.chatLoadingStates.delete(convId);
			if (convId === conversationsStore.activeConversation?.id) this.isLoading = false;
		}
	}
	private setChatStreaming(convId: string, response: string, messageId: string): void {
		this.touchConversationState(convId);
		this.chatStreamingStates.set(convId, { response, messageId });
		if (convId === conversationsStore.activeConversation?.id) this.currentResponse = response;
	}
	private clearChatStreaming(convId: string): void {
		this.chatStreamingStates.delete(convId);
		if (convId === conversationsStore.activeConversation?.id) this.currentResponse = '';
	}
	private getChatStreaming(convId: string): { response: string; messageId: string } | undefined {
		return this.chatStreamingStates.get(convId);
	}
	syncLoadingStateForChat(convId: string): void {
		this.isLoading = this.chatLoadingStates.get(convId) || false;
		const s = this.chatStreamingStates.get(convId);
		this.currentResponse = s?.response || '';
		this.isStreamingActive = s !== undefined;
		this.setActiveProcessingConversation(convId);
		// Sync streaming content to activeMessages so UI displays current content
		if (s?.response && s?.messageId) {
			const idx = conversationsStore.findMessageIndex(s.messageId);
			if (idx !== -1) {
				conversationsStore.updateMessageAtIndex(idx, { content: s.response });
			}
		}
	}

	clearUIState(): void {
		this.isLoading = false;
		this.currentResponse = '';
		this.isStreamingActive = false;
	}

	setActiveProcessingConversation(conversationId: string | null): void {
		this.activeConversationId = conversationId;
		this.activeProcessingState = conversationId
			? this.processingStates.get(conversationId) || null
			: null;
	}

	getProcessingState(conversationId: string): ApiProcessingState | null {
		return this.processingStates.get(conversationId) || null;
	}

	private setProcessingState(conversationId: string, state: ApiProcessingState | null): void {
		if (state === null) this.processingStates.delete(conversationId);
		else this.processingStates.set(conversationId, state);
		if (conversationId === this.activeConversationId) this.activeProcessingState = state;
	}

	clearProcessingState(conversationId: string): void {
		this.processingStates.delete(conversationId);
		if (conversationId === this.activeConversationId) this.activeProcessingState = null;
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
		let c = this.abortControllers.get(convId);
		if (!c || c.signal.aborted) {
			c = new AbortController();
			this.abortControllers.set(convId, c);
		}
		return c;
	}

	private abortRequest(convId?: string): void {
		if (convId) {
			const c = this.abortControllers.get(convId);
			if (c) {
				c.abort();
				this.abortControllers.delete(convId);
			}
		} else {
			for (const c of this.abortControllers.values()) c.abort();
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
		if (!this._pendingDraftMessage && this._pendingDraftFiles.length === 0) return null;
		const d = { message: this._pendingDraftMessage, files: [...this._pendingDraftFiles] };
		this._pendingDraftMessage = '';
		this._pendingDraftFiles = [];
		return d;
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

	private isChatLoadingInternal(convId: string): boolean {
		return this.chatStreamingStates.has(convId);
	}

	private touchConversationState(convId: string): void {
		this.conversationStateTimestamps.set(convId, { lastAccessed: Date.now() });
	}

	cleanupOldConversationStates(activeConversationIds?: string[]): number {
		const now = Date.now();
		const activeIdsList = activeConversationIds ?? [];
		const preserveIds = this.activeConversationId
			? [...activeIdsList, this.activeConversationId]
			: activeIdsList;
		const allConvIds = [
			...new Set([
				...this.chatLoadingStates.keys(),
				...this.chatStreamingStates.keys(),
				...this.abortControllers.keys(),
				...this.processingStates.keys(),
				...this.conversationStateTimestamps.keys()
			])
		];
		const cleanupCandidates: Array<{ convId: string; lastAccessed: number }> = [];
		for (const convId of allConvIds) {
			if (preserveIds.includes(convId)) continue;
			if (this.chatLoadingStates.get(convId)) continue;
			if (this.chatStreamingStates.has(convId)) continue;
			const ts = this.conversationStateTimestamps.get(convId);
			cleanupCandidates.push({ convId, lastAccessed: ts?.lastAccessed ?? 0 });
		}
		cleanupCandidates.sort((a, b) => a.lastAccessed - b.lastAccessed);
		let cleanedUp = 0;
		for (const { convId, lastAccessed } of cleanupCandidates) {
			if (
				cleanupCandidates.length - cleanedUp > MAX_INACTIVE_CONVERSATION_STATES ||
				now - lastAccessed > INACTIVE_CONVERSATION_STATE_MAX_AGE_MS
			) {
				this.cleanupConversationState(convId);
				cleanedUp++;
			}
		}
		return cleanedUp;
	}
	private cleanupConversationState(convId: string): void {
		const c = this.abortControllers.get(convId);
		if (c && !c.signal.aborted) c.abort();
		this.chatLoadingStates.delete(convId);
		this.chatStreamingStates.delete(convId);
		this.abortControllers.delete(convId);
		this.processingStates.delete(convId);
		this.conversationStateTimestamps.delete(convId);
	}
	getTrackedConversationCount(): number {
		return new Set([
			...this.chatLoadingStates.keys(),
			...this.chatStreamingStates.keys(),
			...this.abortControllers.keys(),
			...this.processingStates.keys()
		]).size;
	}

	private getMessageByIdWithRole(
		messageId: string,
		expectedRole?: MessageRole
	): { message: DatabaseMessage; index: number } | null {
		const index = conversationsStore.findMessageIndex(messageId);
		if (index === -1) return null;
		const message = conversationsStore.activeMessages[index];
		if (expectedRole && message.role !== expectedRole) return null;
		return { message, index };
	}

	async addMessage(
		role: MessageRole,
		content: string,
		type: MessageType = MessageType.TEXT,
		parent: string = '-1',
		extras?: DatabaseMessageExtra[]
	): Promise<DatabaseMessage> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv) throw new Error('No active conversation');
		let parentId: string | null = null;
		if (parent === '-1') {
			const am = conversationsStore.activeMessages;
			if (am.length > 0) parentId = am[am.length - 1].id;
			else {
				const all = await conversationsStore.getConversationMessages(activeConv.id);
				const r = all.find((m) => m.parent === null && m.type === 'root');
				parentId = r ? r.id : await DatabaseService.createRootMessage(activeConv.id);
			}
		} else parentId = parent;
		const message = await DatabaseService.createMessageBranch(
			{
				convId: activeConv.id,
				role,
				content,
				type,
				timestamp: Date.now(),
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
	}

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
			const rootId = rootMessage
				? rootMessage.id
				: await DatabaseService.createRootMessage(activeConv.id);
			const existingSystemMessage = allMessages.find(
				(m) => m.role === MessageRole.SYSTEM && m.parent === rootId
			);
			if (existingSystemMessage) {
				this.pendingEditMessageId = existingSystemMessage.id;
				if (!conversationsStore.activeMessages.some((m) => m.id === existingSystemMessage.id))
					conversationsStore.activeMessages.unshift(existingSystemMessage);
				return;
			}
			const am = conversationsStore.activeMessages;
			const firstActiveMessage = am.find((m) => m.parent === rootId);
			const systemMessage = await DatabaseService.createSystemMessage(
				activeConv.id,
				SYSTEM_MESSAGE_PLACEHOLDER,
				rootId
			);
			if (firstActiveMessage) {
				await DatabaseService.updateMessage(firstActiveMessage.id, { parent: systemMessage.id });
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
				if (firstMsgIndex !== -1)
					conversationsStore.updateMessageAtIndex(firstMsgIndex, { parent: systemMessage.id });
			}
			conversationsStore.activeMessages.unshift(systemMessage);
			this.pendingEditMessageId = systemMessage.id;
			conversationsStore.updateConversationTimestamp();
		} catch (error) {
			console.error('Failed to add system prompt:', error);
		}
	}

	async removeSystemPromptPlaceholder(messageId: string): Promise<boolean> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv) return false;
		try {
			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const systemMessage = allMessages.find((m) => m.id === messageId);
			if (!systemMessage || systemMessage.role !== MessageRole.SYSTEM) return false;
			const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);
			if (!rootMessage) return false;
			if (allMessages.length === 2 && systemMessage.children.length === 0) {
				await conversationsStore.deleteConversation(activeConv.id);
				return true;
			}
			for (const childId of systemMessage.children) {
				await DatabaseService.updateMessage(childId, { parent: rootMessage.id });
				const childIndex = conversationsStore.findMessageIndex(childId);
				if (childIndex !== -1)
					conversationsStore.updateMessageAtIndex(childIndex, { parent: rootMessage.id });
			}
			await DatabaseService.updateMessage(rootMessage.id, {
				children: [
					...rootMessage.children.filter((id: string) => id !== messageId),
					...systemMessage.children
				]
			});
			await DatabaseService.deleteMessage(messageId);
			const systemIndex = conversationsStore.findMessageIndex(messageId);
			if (systemIndex !== -1) conversationsStore.activeMessages.splice(systemIndex, 1);
			conversationsStore.updateConversationTimestamp();
			return false;
		} catch (error) {
			console.error('Failed to remove system prompt placeholder:', error);
			return false;
		}
	}

	private async createAssistantMessage(parentId?: string): Promise<DatabaseMessage> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv) throw new Error('No active conversation');
		return await DatabaseService.createMessageBranch(
			{
				convId: activeConv.id,
				type: MessageType.TEXT,
				role: MessageRole.ASSISTANT,
				content: '',
				timestamp: Date.now(),
				toolCalls: '',
				children: [],
				model: null
			},
			parentId || null
		);
	}

	async sendMessage(content: string, extras?: DatabaseMessageExtra[]): Promise<void> {
		if (!content.trim() && (!extras || extras.length === 0)) return;
		const activeConv = conversationsStore.activeConversation;
		if (activeConv && this.isChatLoadingInternal(activeConv.id)) return;

		let isNewConversation = false;
		if (!activeConv) {
			await conversationsStore.createConversation();
			isNewConversation = true;
		}
		const currentConv = conversationsStore.activeConversation;
		if (!currentConv) return;
		this.showErrorDialog(null);
		this.setChatLoading(currentConv.id, true);
		this.clearChatStreaming(currentConv.id);
		try {
			let parentIdForUserMessage: string | undefined;
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
					parentIdForUserMessage = systemMessage.id;
				} else parentIdForUserMessage = rootId;
			}
			const userMessage = await this.addMessage(
				MessageRole.USER,
				content,
				MessageType.TEXT,
				parentIdForUserMessage ?? '-1'
			);
			if (isNewConversation && content)
				await conversationsStore.updateConversationName(currentConv.id, content.trim());
			const assistantMessage = await this.createAssistantMessage(userMessage.id);
			conversationsStore.addMessageToActive(assistantMessage);
			await this.streamChatCompletion(
				conversationsStore.activeMessages.slice(0, -1),
				assistantMessage
			);
		} catch (error) {
			if (isAbortError(error)) {
				this.setChatLoading(currentConv.id, false);
				return;
			}
			console.error('Failed to send message:', error);
			this.setChatLoading(currentConv.id, false);
			const dialogType =
				error instanceof Error && error.name === 'TimeoutError'
					? ErrorDialogType.TIMEOUT
					: ErrorDialogType.SERVER;
			const contextInfo = (
				error as Error & { contextInfo?: { n_prompt_tokens: number; n_ctx: number } }
			).contextInfo;
			this.showErrorDialog({
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
		let effectiveModel = modelOverride;

		if (isRouterMode() && !effectiveModel) {
			const conversationModel = this.getConversationModel(allMessages);
			effectiveModel = selectedModelName() || conversationModel;
		}

		if (isRouterMode() && effectiveModel) {
			if (!modelsStore.getModelProps(effectiveModel))
				await modelsStore.fetchModelProps(effectiveModel);
		}

		let streamedContent = '',
			streamedToolCallContent = '',
			isReasoningOpen = false,
			hasStreamedChunks = false,
			resolvedModel: string | null = null,
			modelPersisted = false;
		let streamedExtras: DatabaseMessageExtra[] = assistantMessage.extra
			? JSON.parse(JSON.stringify(assistantMessage.extra))
			: [];
		const recordModel = (modelName: string | null | undefined, persistImmediately = true): void => {
			if (!modelName) return;
			const n = normalizeModelName(modelName);
			if (!n || n === resolvedModel) return;
			resolvedModel = n;
			const idx = conversationsStore.findMessageIndex(assistantMessage.id);
			conversationsStore.updateMessageAtIndex(idx, { model: n });
			if (persistImmediately && !modelPersisted) {
				modelPersisted = true;
				DatabaseService.updateMessage(assistantMessage.id, { model: n }).catch(() => {
					modelPersisted = false;
					resolvedModel = null;
				});
			}
		};
		const updateStreamingContent = () => {
			this.setChatStreaming(assistantMessage.convId, streamedContent, assistantMessage.id);
			const idx = conversationsStore.findMessageIndex(assistantMessage.id);
			conversationsStore.updateMessageAtIndex(idx, { content: streamedContent });
		};
		const appendContentChunk = (chunk: string) => {
			if (isReasoningOpen) {
				streamedContent += REASONING_TAGS.END;
				isReasoningOpen = false;
			}
			streamedContent += chunk;
			hasStreamedChunks = true;
			updateStreamingContent();
		};
		const appendReasoningChunk = (chunk: string) => {
			if (!isReasoningOpen) {
				streamedContent += REASONING_TAGS.START;
				isReasoningOpen = true;
			}
			streamedContent += chunk;
			hasStreamedChunks = true;
			updateStreamingContent();
		};
		const finalizeReasoning = () => {
			if (isReasoningOpen) {
				streamedContent += REASONING_TAGS.END;
				isReasoningOpen = false;
			}
		};
		this.setStreamingActive(true);
		this.setActiveProcessingConversation(assistantMessage.convId);
		const abortController = this.getOrCreateAbortController(assistantMessage.convId);
		const streamCallbacks: ChatStreamCallbacks = {
			onChunk: (chunk: string) => appendContentChunk(chunk),
			onReasoningChunk: (chunk: string) => appendReasoningChunk(chunk),
			onToolCallChunk: (chunk: string) => {
				const c = chunk.trim();
				if (!c) return;
				streamedToolCallContent = c;
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
				this.setStreamingActive(false);
				finalizeReasoning();
				const combinedContent = hasStreamedChunks
					? streamedContent
					: wrapReasoningContent(finalContent || '', reasoningContent);
				const updateData: Record<string, unknown> = {
					content: combinedContent,
					toolCalls: toolCallContent || streamedToolCallContent,
					timings
				};
				if (streamedExtras.length > 0) updateData.extra = streamedExtras;
				if (resolvedModel && !modelPersisted) updateData.model = resolvedModel;
				await DatabaseService.updateMessage(assistantMessage.id, updateData);
				const idx = conversationsStore.findMessageIndex(assistantMessage.id);
				const uiUpdate: Partial<DatabaseMessage> = {
					content: combinedContent,
					toolCalls: updateData.toolCalls as string
				};
				if (streamedExtras.length > 0) uiUpdate.extra = streamedExtras;
				if (timings) uiUpdate.timings = timings;
				if (resolvedModel) uiUpdate.model = resolvedModel;
				conversationsStore.updateMessageAtIndex(idx, uiUpdate);
				await conversationsStore.updateCurrentNode(assistantMessage.id);
				if (onComplete) await onComplete(combinedContent);
				this.setChatLoading(assistantMessage.convId, false);
				this.clearChatStreaming(assistantMessage.convId);
				this.setProcessingState(assistantMessage.convId, null);
				if (isRouterMode()) modelsStore.fetchRouterModels().catch(console.error);
			},
			onError: (error: Error) => {
				this.setStreamingActive(false);
				if (isAbortError(error)) {
					this.setChatLoading(assistantMessage.convId, false);
					this.clearChatStreaming(assistantMessage.convId);
					this.setProcessingState(assistantMessage.convId, null);
					return;
				}
				console.error('Streaming error:', error);
				this.setChatLoading(assistantMessage.convId, false);
				this.clearChatStreaming(assistantMessage.convId);
				this.setProcessingState(assistantMessage.convId, null);
				const idx = conversationsStore.findMessageIndex(assistantMessage.id);
				if (idx !== -1) {
					const failedMessage = conversationsStore.removeMessageAtIndex(idx);
					if (failedMessage) DatabaseService.deleteMessage(failedMessage.id).catch(console.error);
				}
				const contextInfo = (
					error as Error & { contextInfo?: { n_prompt_tokens: number; n_ctx: number } }
				).contextInfo;
				this.showErrorDialog({
					type: error.name === 'TimeoutError' ? ErrorDialogType.TIMEOUT : ErrorDialogType.SERVER,
					message: error.message,
					contextInfo
				});
				if (onError) onError(error);
			}
		};

		const completionOptions = {
			...this.getApiOptions(),
			...(effectiveModel ? { model: effectiveModel } : {}),
			...streamCallbacks
		};

		await ChatService.sendMessage(
			allMessages,
			completionOptions,
			assistantMessage.convId,
			abortController.signal
		);
	}

	async stopGeneration(): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv) return;
		await this.stopGenerationForChat(activeConv.id);
	}
	async stopGenerationForChat(convId: string): Promise<void> {
		await this.savePartialResponseIfNeeded(convId);
		this.setStreamingActive(false);
		this.abortRequest(convId);
		this.setChatLoading(convId, false);
		this.clearChatStreaming(convId);
		this.setProcessingState(convId, null);
	}
	private async savePartialResponseIfNeeded(convId?: string): Promise<void> {
		const conversationId = convId || conversationsStore.activeConversation?.id;
		if (!conversationId) return;
		const streamingState = this.getChatStreaming(conversationId);
		if (!streamingState || !streamingState.response.trim()) return;
		const messages =
			conversationId === conversationsStore.activeConversation?.id
				? conversationsStore.activeMessages
				: await conversationsStore.getConversationMessages(conversationId);
		if (!messages.length) return;
		const lastMessage = messages[messages.length - 1];
		if (lastMessage?.role === MessageRole.ASSISTANT) {
			try {
				const updateData: { content: string; timings?: ChatMessageTimings } = {
					content: streamingState.response
				};
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
				lastMessage.content = streamingState.response;
				if (updateData.timings) lastMessage.timings = updateData.timings;
			} catch (error) {
				lastMessage.content = streamingState.response;
				console.error('Failed to save partial response:', error);
			}
		}
	}

	async updateMessage(messageId: string, newContent: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv) return;
		if (this.isChatLoadingInternal(activeConv.id)) await this.stopGeneration();
		const result = this.getMessageByIdWithRole(messageId, MessageRole.USER);
		if (!result) return;
		const { message: messageToUpdate, index: messageIndex } = result;
		const originalContent = messageToUpdate.content;
		try {
			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);
			const isFirstUserMessage = rootMessage && messageToUpdate.parent === rootMessage.id;
			conversationsStore.updateMessageAtIndex(messageIndex, { content: newContent });
			await DatabaseService.updateMessage(messageId, { content: newContent });
			if (isFirstUserMessage && newContent.trim())
				await conversationsStore.updateConversationTitleWithConfirmation(
					activeConv.id,
					newContent.trim()
				);
			const messagesToRemove = conversationsStore.activeMessages.slice(messageIndex + 1);
			for (const message of messagesToRemove) await DatabaseService.deleteMessage(message.id);
			conversationsStore.sliceActiveMessages(messageIndex + 1);
			conversationsStore.updateConversationTimestamp();
			this.setChatLoading(activeConv.id, true);
			this.clearChatStreaming(activeConv.id);
			const assistantMessage = await this.createAssistantMessage();
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
			if (!isAbortError(error)) console.error('Failed to update message:', error);
		}
	}

	async regenerateMessage(messageId: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv || this.isChatLoadingInternal(activeConv.id)) return;
		const result = this.getMessageByIdWithRole(messageId, MessageRole.ASSISTANT);
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
			conversationsStore.addMessageToActive(assistantMessage);
			await this.streamChatCompletion(
				conversationsStore.activeMessages.slice(0, -1),
				assistantMessage
			);
		} catch (error) {
			if (!isAbortError(error)) console.error('Failed to regenerate message:', error);
			this.setChatLoading(activeConv?.id || '', false);
		}
	}

	async regenerateMessageWithBranching(messageId: string, modelOverride?: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv || this.isChatLoadingInternal(activeConv.id)) return;
		try {
			const idx = conversationsStore.findMessageIndex(messageId);
			if (idx === -1) return;
			const msg = conversationsStore.activeMessages[idx];
			if (msg.role !== MessageRole.ASSISTANT) return;
			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const parentMessage = allMessages.find((m) => m.id === msg.parent);
			if (!parentMessage) return;
			this.setChatLoading(activeConv.id, true);
			this.clearChatStreaming(activeConv.id);
			const newAssistantMessage = await DatabaseService.createMessageBranch(
				{
					convId: msg.convId,
					type: msg.type,
					timestamp: Date.now(),
					role: msg.role,
					content: '',
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
			if (!isAbortError(error))
				console.error('Failed to regenerate message with branching:', error);
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
		const messageToDelete = allMessages.find((m) => m.id === messageId);

		// For system messages, don't count descendants as they will be preserved (reparented to root)
		if (messageToDelete?.role === MessageRole.SYSTEM) {
			const messagesToDelete = allMessages.filter((m) => m.id === messageId);
			let userMessages = 0,
				assistantMessages = 0;
			const messageTypes: string[] = [];

			for (const msg of messagesToDelete) {
				if (msg.role === MessageRole.USER) {
					userMessages++;
					if (!messageTypes.includes('user message')) messageTypes.push('user message');
				} else if (msg.role === MessageRole.ASSISTANT) {
					assistantMessages++;
					if (!messageTypes.includes('assistant response')) messageTypes.push('assistant response');
				}
			}

			return { totalCount: 1, userMessages, assistantMessages, messageTypes };
		}

		const descendants = findDescendantMessages(allMessages, messageId);
		const allToDelete = [messageId, ...descendants];
		const messagesToDelete = allMessages.filter((m) => allToDelete.includes(m.id));
		let userMessages = 0,
			assistantMessages = 0;
		const messageTypes: string[] = [];

		for (const msg of messagesToDelete) {
			if (msg.role === MessageRole.USER) {
				userMessages++;
				if (!messageTypes.includes('user message')) messageTypes.push('user message');
			} else if (msg.role === MessageRole.ASSISTANT) {
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

	async continueAssistantMessage(messageId: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv || this.isChatLoadingInternal(activeConv.id)) return;
		const result = this.getMessageByIdWithRole(messageId, MessageRole.ASSISTANT);

		if (!result) return;

		const { message: msg, index: idx } = result;

		try {
			this.showErrorDialog(null);
			this.setChatLoading(activeConv.id, true);
			this.clearChatStreaming(activeConv.id);

			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const dbMessage = allMessages.find((m) => m.id === messageId);

			if (!dbMessage) {
				this.setChatLoading(activeConv.id, false);
				return;
			}

			const originalContent = dbMessage.content;
			const conversationContext = conversationsStore.activeMessages.slice(0, idx);
			const contextWithContinue = [
				...conversationContext,
				{ role: MessageRole.ASSISTANT as const, content: originalContent }
			];

			let appendedContent = '',
				hasReceivedContent = false,
				isReasoningOpen = hasUnclosedReasoningTag(originalContent);

			const updateStreamingContent = (fullContent: string) => {
				this.setChatStreaming(msg.convId, fullContent, msg.id);
				conversationsStore.updateMessageAtIndex(idx, { content: fullContent });
			};

			const appendContentChunk = (chunk: string) => {
				if (isReasoningOpen) {
					appendedContent += REASONING_TAGS.END;
					isReasoningOpen = false;
				}
				appendedContent += chunk;
				hasReceivedContent = true;
				updateStreamingContent(originalContent + appendedContent);
			};

			const appendReasoningChunk = (chunk: string) => {
				if (!isReasoningOpen) {
					appendedContent += REASONING_TAGS.START;
					isReasoningOpen = true;
				}
				appendedContent += chunk;
				hasReceivedContent = true;
				updateStreamingContent(originalContent + appendedContent);
			};

			const finalizeReasoning = () => {
				if (isReasoningOpen) {
					appendedContent += REASONING_TAGS.END;
					isReasoningOpen = false;
				}
			};

			const abortController = this.getOrCreateAbortController(msg.convId);

			await ChatService.sendMessage(
				contextWithContinue,
				{
					...this.getApiOptions(),
					onChunk: (chunk: string) => appendContentChunk(chunk),
					onReasoningChunk: (chunk: string) => appendReasoningChunk(chunk),
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
						finalizeReasoning();

						const appendedFromCompletion = hasReceivedContent
							? appendedContent
							: wrapReasoningContent(finalContent || '', reasoningContent);
						const fullContent = originalContent + appendedFromCompletion;

						await DatabaseService.updateMessage(msg.id, {
							content: fullContent,
							timestamp: Date.now(),
							timings
						});

						conversationsStore.updateMessageAtIndex(idx, {
							content: fullContent,
							timestamp: Date.now(),
							timings
						});

						conversationsStore.updateConversationTimestamp();

						this.setChatLoading(msg.convId, false);
						this.clearChatStreaming(msg.convId);
						this.setProcessingState(msg.convId, null);
					},
					onError: async (error: Error) => {
						if (isAbortError(error)) {
							if (hasReceivedContent && appendedContent) {
								await DatabaseService.updateMessage(msg.id, {
									content: originalContent + appendedContent,
									timestamp: Date.now()
								});

								conversationsStore.updateMessageAtIndex(idx, {
									content: originalContent + appendedContent,
									timestamp: Date.now()
								});
							}

							this.setChatLoading(msg.convId, false);
							this.clearChatStreaming(msg.convId);
							this.setProcessingState(msg.convId, null);

							return;
						}

						console.error('Continue generation error:', error);
						conversationsStore.updateMessageAtIndex(idx, { content: originalContent });

						await DatabaseService.updateMessage(msg.id, { content: originalContent });

						this.setChatLoading(msg.convId, false);
						this.clearChatStreaming(msg.convId);
						this.setProcessingState(msg.convId, null);
						this.showErrorDialog({
							type:
								error.name === 'TimeoutError' ? ErrorDialogType.TIMEOUT : ErrorDialogType.SERVER,
							message: error.message
						});
					}
				},

				msg.convId,
				abortController.signal
			);
		} catch (error) {
			if (!isAbortError(error)) console.error('Failed to continue message:', error);
			if (activeConv) this.setChatLoading(activeConv.id, false);
		}
	}

	async editAssistantMessage(
		messageId: string,
		newContent: string,
		shouldBranch: boolean
	): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv || this.isChatLoadingInternal(activeConv.id)) return;

		const result = this.getMessageByIdWithRole(messageId, MessageRole.ASSISTANT);
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
				conversationsStore.updateMessageAtIndex(idx, { content: newContent });
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

		const result = this.getMessageByIdWithRole(messageId, MessageRole.USER);
		if (!result) return;

		const { message: msg, index: idx } = result;
		try {
			const updateData: Partial<DatabaseMessage> = { content: newContent };

			if (newExtras !== undefined) updateData.extra = JSON.parse(JSON.stringify(newExtras));

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

	async editMessageWithBranching(
		messageId: string,
		newContent: string,
		newExtras?: DatabaseMessageExtra[]
	): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv || this.isChatLoadingInternal(activeConv.id)) return;
		let result = this.getMessageByIdWithRole(messageId, MessageRole.USER);
		if (!result) result = this.getMessageByIdWithRole(messageId, MessageRole.SYSTEM);
		if (!result) return;
		const { message: msg } = result;
		try {
			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);
			const isFirstUserMessage =
				msg.role === MessageRole.USER && rootMessage && msg.parent === rootMessage.id;
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
					toolCalls: msg.toolCalls || '',
					children: [],
					extra: extrasToUse,
					model: msg.model
				},
				parentId
			);
			await conversationsStore.updateCurrentNode(newMessage.id);
			conversationsStore.updateConversationTimestamp();
			if (isFirstUserMessage && newContent.trim())
				await conversationsStore.updateConversationTitleWithConfirmation(
					activeConv.id,
					newContent.trim()
				);
			await conversationsStore.refreshActiveMessages();
			if (msg.role === MessageRole.USER) await this.generateResponseForMessage(newMessage.id);
		} catch (error) {
			console.error('Failed to edit message with branching:', error);
		}
	}

	private async generateResponseForMessage(userMessageId: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv) return;

		this.showErrorDialog(null);
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
					type: MessageType.TEXT,
					timestamp: Date.now(),
					role: MessageRole.ASSISTANT,
					content: '',
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

	private getContextTotal(): number | null {
		const activeConvId = this.activeConversationId;
		const activeState = activeConvId ? this.getProcessingState(activeConvId) : null;

		if (activeState && typeof activeState.contextTotal === 'number' && activeState.contextTotal > 0)
			return activeState.contextTotal;

		if (isRouterMode()) {
			const modelContextSize = selectedModelContextSize();

			if (typeof modelContextSize === 'number' && modelContextSize > 0) {
				return modelContextSize;
			}
		} else {
			const propsContextSize = contextSize();

			if (typeof propsContextSize === 'number' && propsContextSize > 0) {
				return propsContextSize;
			}
		}

		return null;
	}

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
			this.setProcessingState(targetId, processingState);
		}
	}

	private parseTimingData(timingData: Record<string, unknown>): ApiProcessingState | null {
		const promptTokens = (timingData.prompt_n as number) || 0,
			promptMs = (timingData.prompt_ms as number) || undefined,
			predictedTokens = (timingData.predicted_n as number) || 0,
			tokensPerSecond = (timingData.predicted_per_second as number) || 0,
			cacheTokens = (timingData.cache_n as number) || 0;
		const promptProgress = timingData.prompt_progress as
			| { total: number; cache: number; processed: number; time_ms: number }
			| undefined;
		const contextTotal = this.getContextTotal();
		const currentConfig = config();
		const outputTokensMax = currentConfig.max_tokens || -1;
		const contextUsed = promptTokens + cacheTokens + predictedTokens,
			outputTokensUsed = predictedTokens;
		const progressCache = promptProgress?.cache || 0,
			progressActualDone = (promptProgress?.processed ?? 0) - progressCache,
			progressActualTotal = (promptProgress?.total ?? 0) - progressCache;
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

	restoreProcessingStateFromMessages(messages: DatabaseMessage[], conversationId: string): void {
		for (let i = messages.length - 1; i >= 0; i--) {
			const message = messages[i];
			if (message.role === MessageRole.ASSISTANT && message.timings) {
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
					this.setProcessingState(conversationId, restoredState);
					return;
				}
			}
		}
	}

	getConversationModel(messages: DatabaseMessage[]): string | null {
		for (let i = messages.length - 1; i >= 0; i--) {
			const message = messages[i];
			if (message.role === MessageRole.ASSISTANT && message.model) return message.model;
		}
		return null;
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

export const chatStore = new ChatStore();

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
