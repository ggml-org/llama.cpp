/**
 * ChatMessageFlows - Message-level flows for the active conversation
 *
 * Owns the operations that mutate chat history and (re)stream a response:
 * editing, regeneration, continuation and deletion of messages. Created and
 * owned by chatStore; the host exposes the streaming core and the
 * per-conversation state setters these flows drive.
 */

import {
	ContinueIntentKind,
	ErrorDialogType,
	MessageRole,
	MessageType,
	StreamConnectionState
} from '$lib/enums';
import { ChatService } from '$lib/services/chat.service';
import { DatabaseService } from '$lib/services/database.service';
import type { ChatProcessingStore } from '$lib/stores/chat/processing.svelte';
// direct imports between stores, not via the barrel, to avoid circular deps
import { conversationsStore } from '$lib/stores/conversations/index.svelte';
import type {
	ChatMessagePromptProgress,
	ChatMessageTimings,
	DatabaseMessage,
	DatabaseMessageExtra,
	ErrorDialogState
} from '$lib/types';
import {
	classifyContinueIntent,
	filterByLeafNodeId,
	findDescendantMessages,
	findLeafNode,
	findMessageById,
	isAbortError,
	sliceAtLastCompaction
} from '$lib/utils';

/**
 * The slice of chatStore the flows drive. Kept narrow on purpose so the flows
 * cannot reach around the host's full surface; chatStore implements this
 * structurally.
 */
export interface ChatFlowsHost {
	processing: ChatProcessingStore;
	streamConnectionState: StreamConnectionState;
	cancelPreEncode(): void;
	clearChatStreaming(convId: string, messageId?: string): void;
	cleanupStreaming(convId: string): void;
	createAssistantMessage(parentId?: string): Promise<DatabaseMessage>;
	getApiOptions(): Record<string, unknown>;
	getOrCreateAbortController(convId: string): AbortController;
	isChatLoadingInternal(convId: string): boolean;
	setChatLoading(convId: string, loading: boolean): void;
	setChatReasoning(convId: string, reasoning: boolean): void;
	setChatStreaming(
		convId: string,
		response: string,
		messageId: string,
		model?: string | null
	): void;
	showErrorDialog(state: ErrorDialogState | null): void;
	stopGeneration(): Promise<void>;
	streamChatCompletion(
		allMessages: DatabaseMessage[],
		assistantMessage: DatabaseMessage,
		onComplete?: (content: string) => Promise<void>,
		onError?: (error: Error) => void,
		modelOverride?: string | null,
		firstUserMessageContent?: string
	): Promise<void>;
}

export class ChatMessageFlows {
	constructor(private host: ChatFlowsHost) {}

	async continueAssistantMessage(messageId: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;

		if (!activeConv || this.host.isChatLoadingInternal(activeConv.id)) return;

		const result = this.getMessageByIdWithRole(messageId, MessageRole.ASSISTANT);

		if (!result) return;

		const { index: idx, message: msg } = result;
		// Decide which resume path applies. tool_calls without tool results can
		// not be resumed mid sequence by continue_final_message, branch instead.
		// tool_calls already paired with tool results need a fresh next turn,
		// not a token level continuation of the target assistant.
		const intent = classifyContinueIntent(conversationsStore.activeMessages, idx);

		if (intent.kind === ContinueIntentKind.RERUN_TURN) {
			return this.regenerateMessageWithBranching(messageId);
		}

		if (intent.kind === ContinueIntentKind.NEXT_TURN) {
			return this.continueAsNextAgenticTurn(intent.truncateAfter);
		}

		try {
			this.host.showErrorDialog(null);
			this.host.setChatLoading(activeConv.id, true);
			this.host.clearChatStreaming(activeConv.id);

			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const dbMessage = findMessageById(allMessages, messageId);

			if (!dbMessage) {
				this.host.setChatLoading(activeConv.id, false);

				return;
			}

			const originalContent = dbMessage.content;
			const originalReasoning = dbMessage.reasoningContent || '';
			// Hand the persisted DatabaseMessage straight to sendMessage so its
			// internal converter preserves tool_calls and extras when present.
			// Reconstructing a bare {role, content} here would drop those fields
			// and break continue_final_message for messages with tool calls.
			const contextWithContinue = sliceAtLastCompaction(
				conversationsStore.activeMessages.slice(0, idx + 1)
			);

			let appendedContent = '';
			let appendedReasoning = '';
			let hasReceivedContent = false;

			const updateStreamingContent = (fullContent: string) => {
				this.host.setChatStreaming(msg.convId, fullContent, msg.id);
				// resolve the row by id on every write, switching to another conv mid continue makes
				// this a no op instead of writing positionally into the now displayed conversation
				conversationsStore.updateMessageAtIndex(conversationsStore.findMessageIndex(msg.id), {
					content: fullContent
				});
			};
			const abortController = this.host.getOrCreateAbortController(msg.convId);

			await ChatService.sendMessage(
				contextWithContinue,
				{
					...this.host.getApiOptions(),
					continueFinalMessage: true,
					onChunk: (chunk: string) => {
						appendedContent += chunk;
						hasReceivedContent = true;
						updateStreamingContent(originalContent + appendedContent);
						this.host.setChatReasoning(msg.convId, false);
					},
					onComplete: async (
						finalContent?: string,
						reasoningContent?: string,
						timings?: ChatMessageTimings
					) => {
						const finalAppendedContent = hasReceivedContent ? appendedContent : finalContent || '';
						const finalAppendedReasoning = hasReceivedContent
							? appendedReasoning
							: reasoningContent || '';
						const fullContent = originalContent + finalAppendedContent;
						const fullReasoning = originalReasoning + finalAppendedReasoning || undefined;

						await DatabaseService.updateMessage(msg.id, {
							content: fullContent,
							reasoningContent: fullReasoning,
							timestamp: Date.now(),
							timings
						});

						conversationsStore.updateMessageAtIndex(conversationsStore.findMessageIndex(msg.id), {
							content: fullContent,
							reasoningContent: fullReasoning,
							timestamp: Date.now(),
							timings
						});

						conversationsStore.updateConversationTimestamp(msg.convId);

						this.host.cleanupStreaming(msg.convId);
					},
					onCompletionId: (id: string) => {
						if (!id) return;

						// refresh the message id so a later skip targets the live slot after a continue
						conversationsStore.updateMessageAtIndex(conversationsStore.findMessageIndex(msg.id), {
							completionId: id
						});
						DatabaseService.updateMessage(msg.id, { completionId: id }).catch(() => {});
					},
					onConnectionState: (state: StreamConnectionState) => {
						if (msg.convId === conversationsStore.activeConversation?.id) {
							this.host.streamConnectionState = state;
						}
					},
					onError: async (error: Error) => {
						if (isAbortError(error)) {
							if (hasReceivedContent && appendedContent) {
								await DatabaseService.updateMessage(msg.id, {
									content: originalContent + appendedContent,
									reasoningContent: originalReasoning + appendedReasoning || undefined,
									timestamp: Date.now()
								});

								conversationsStore.updateMessageAtIndex(
									conversationsStore.findMessageIndex(msg.id),
									{
										content: originalContent + appendedContent,
										reasoningContent: originalReasoning + appendedReasoning || undefined,
										timestamp: Date.now()
									}
								);
							}

							this.host.cleanupStreaming(msg.convId);

							return;
						}

						console.error('Continue generation error:', error);
						// keep whatever was appended so far, the message stays in memory and in DB
						await DatabaseService.updateMessage(msg.id, {
							content: originalContent + appendedContent,
							reasoningContent: originalReasoning + appendedReasoning || undefined,
							timestamp: Date.now()
						});
						conversationsStore.updateMessageAtIndex(conversationsStore.findMessageIndex(msg.id), {
							content: originalContent + appendedContent,
							reasoningContent: originalReasoning + appendedReasoning || undefined,
							timestamp: Date.now()
						});

						this.host.cleanupStreaming(msg.convId);
						this.host.showErrorDialog({
							message: error.message,
							type: error.name === 'TimeoutError' ? ErrorDialogType.TIMEOUT : ErrorDialogType.SERVER
						});
					},
					onReasoningChunk: (chunk: string) => {
						appendedReasoning += chunk;
						hasReceivedContent = true;
						// mark streaming state so a stop mid-thinking can persist the partial reasoning
						this.host.setChatStreaming(msg.convId, originalContent + appendedContent, msg.id);
						conversationsStore.updateMessageAtIndex(conversationsStore.findMessageIndex(msg.id), {
							reasoningContent: originalReasoning + appendedReasoning
						});
						this.host.setChatReasoning(msg.convId, true);
					},
					onTimings: (timings?: ChatMessageTimings, promptProgress?: ChatMessagePromptProgress) => {
						this.host.processing.applyStreamTimings(timings, promptProgress, msg.convId);
					}
				},

				msg.convId,
				abortController.signal
			);
		} catch (error) {
			if (!isAbortError(error)) console.error('Failed to continue message:', error);

			if (activeConv) this.host.setChatLoading(activeConv.id, false);
		}
	}

	async deleteMessage(messageId: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;

		if (!activeConv) return;

		try {
			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const messageToDelete = findMessageById(allMessages, messageId);

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

	async editAssistantMessage(
		messageId: string,
		newContent: string,
		shouldBranch: boolean
	): Promise<void> {
		const activeConv = conversationsStore.activeConversation;

		if (!activeConv || this.host.isChatLoadingInternal(activeConv.id)) return;

		const result = this.getMessageByIdWithRole(messageId, MessageRole.ASSISTANT);

		if (!result) return;

		const { index: idx, message: msg } = result;

		try {
			if (shouldBranch) {
				const newMessage = await DatabaseService.createMessageBranch(
					{
						children: [],
						content: newContent,
						convId: msg.convId,
						model: msg.model,
						role: msg.role,
						timestamp: Date.now(),
						toolCalls: msg.toolCalls || '',
						type: msg.type
					},
					msg.parent!
				);

				await conversationsStore.updateCurrentNode(newMessage.id);
			} else {
				await DatabaseService.updateMessage(msg.id, { content: newContent });
				conversationsStore.updateMessageAtIndex(idx, { content: newContent });
			}

			conversationsStore.updateConversationTimestamp();

			await conversationsStore.refreshActiveMessages();
		} catch (error) {
			console.error('Failed to edit assistant message:', error);
		}
	}

	async editMessageWithBranching(
		messageId: string,
		newContent: string,
		newExtras?: DatabaseMessageExtra[]
	): Promise<void> {
		const activeConv = conversationsStore.activeConversation;

		if (!activeConv || this.host.isChatLoadingInternal(activeConv.id)) return;

		let result = this.getMessageByIdWithRole(messageId, MessageRole.USER);

		if (!result) result = this.getMessageByIdWithRole(messageId, MessageRole.SYSTEM);

		if (!result) return;

		const { index: idx, message: msg } = result;

		try {
			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);
			const isFirstUserMessage =
				msg.role === MessageRole.USER && rootMessage && msg.parent === rootMessage.id;
			const extrasToUse =
				newExtras !== undefined
					? JSON.parse(JSON.stringify(newExtras))
					: msg.extra
						? JSON.parse(JSON.stringify(msg.extra))
						: undefined;

			let messageIdForResponse: string;

			const dbMsg = findMessageById(allMessages, msg.id);
			const hasChildren = dbMsg ? dbMsg.children.length > 0 : msg.children.length > 0;

			if (!hasChildren) {
				// No responses after this message - update in place instead of branching
				const updates: Partial<DatabaseMessage> = {
					content: newContent,
					extra: extrasToUse,
					timestamp: Date.now()
				};

				await DatabaseService.updateMessage(msg.id, updates);
				conversationsStore.updateMessageAtIndex(idx, updates);
				messageIdForResponse = msg.id;
			} else {
				// Has children - create a new branch as sibling
				const parentId = msg.parent || rootMessage?.id;

				if (!parentId) return;

				const newMessage = await DatabaseService.createMessageBranch(
					{
						children: [],
						content: newContent,
						convId: msg.convId,
						extra: extrasToUse,
						model: msg.model,
						role: msg.role,
						timestamp: Date.now(),
						toolCalls: msg.toolCalls || '',
						type: msg.type
					},
					parentId
				);

				await conversationsStore.updateCurrentNode(newMessage.id);
				messageIdForResponse = newMessage.id;
			}

			conversationsStore.updateConversationTimestamp();

			if (isFirstUserMessage && newContent.trim())
				await conversationsStore.applyTitleFromContent(activeConv.id, newContent);

			await conversationsStore.refreshActiveMessages();

			if (msg.role === MessageRole.USER)
				await this.generateResponseForMessage(messageIdForResponse);
		} catch (error) {
			console.error('Failed to edit message with branching:', error);
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

		const { index: idx, message: msg } = result;

		try {
			const updateData: Partial<DatabaseMessage> = { content: newContent };

			if (newExtras !== undefined) updateData.extra = JSON.parse(JSON.stringify(newExtras));

			await DatabaseService.updateMessage(messageId, updateData);

			conversationsStore.updateMessageAtIndex(idx, updateData);

			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);

			if (rootMessage && msg.parent === rootMessage.id && newContent.trim()) {
				await conversationsStore.applyTitleFromContent(activeConv.id, newContent);
			}

			conversationsStore.updateConversationTimestamp();
		} catch (error) {
			console.error('Failed to edit user message:', error);
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
			return { assistantMessages: 0, messageTypes: [], totalCount: 0, userMessages: 0 };

		const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
		const messageToDelete = findMessageById(allMessages, messageId);

		// For system messages, don't count descendants as they will be preserved (reparented to root)
		if (messageToDelete?.role === MessageRole.SYSTEM) {
			const messagesToDelete = allMessages.filter((m) => m.id === messageId);

			let assistantMessages = 0,
				userMessages = 0;

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

			return { assistantMessages, messageTypes, totalCount: 1, userMessages };
		}

		const descendants = findDescendantMessages(allMessages, messageId);
		const allToDelete = [messageId, ...descendants];
		const messagesToDelete = allMessages.filter((m) => allToDelete.includes(m.id));

		let assistantMessages = 0,
			userMessages = 0;

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

		return { assistantMessages, messageTypes, totalCount: allToDelete.length, userMessages };
	}

	async regenerateMessage(messageId: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;

		if (!activeConv || this.host.isChatLoadingInternal(activeConv.id)) return;

		this.host.cancelPreEncode();
		const result = this.getMessageByIdWithRole(messageId, MessageRole.ASSISTANT);

		if (!result) return;

		const { index: messageIndex } = result;

		try {
			const messagesToRemove = conversationsStore.activeMessages.slice(messageIndex);

			await DatabaseService.deleteMessageCascading(activeConv.id, messagesToRemove[0].id);
			conversationsStore.sliceActiveMessages(messageIndex);
			conversationsStore.updateConversationTimestamp();
			this.host.setChatLoading(activeConv.id, true);
			this.host.clearChatStreaming(activeConv.id);
			const parentMessageId =
				conversationsStore.activeMessages.length > 0
					? conversationsStore.activeMessages[conversationsStore.activeMessages.length - 1].id
					: undefined;
			const assistantMessage = await this.host.createAssistantMessage(parentMessageId);

			conversationsStore.addMessageToActive(assistantMessage);
			await this.host.streamChatCompletion(
				conversationsStore.activeMessages.slice(0, -1),
				assistantMessage
			);
		} catch (error) {
			if (!isAbortError(error)) console.error('Failed to regenerate message:', error);

			this.host.setChatLoading(activeConv?.id || '', false);
		}
	}

	async regenerateMessageWithBranching(messageId: string, modelOverride?: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;

		if (!activeConv || this.host.isChatLoadingInternal(activeConv.id)) return;

		this.host.cancelPreEncode();
		try {
			const idx = conversationsStore.findMessageIndex(messageId);

			if (idx === -1) return;

			const msg = conversationsStore.activeMessages[idx];

			if (msg.role !== MessageRole.ASSISTANT) return;

			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const parentMessage = findMessageById(allMessages, msg.parent);

			if (!parentMessage) return;

			this.host.setChatLoading(activeConv.id, true);
			this.host.clearChatStreaming(activeConv.id);
			const newAssistantMessage = await DatabaseService.createMessageBranch(
				{
					children: [],
					content: '',
					convId: msg.convId,
					model: null,
					role: msg.role,
					timestamp: Date.now(),
					toolCalls: '',
					type: msg.type
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

			await this.host.streamChatCompletion(
				conversationPath,
				newAssistantMessage,
				undefined,
				undefined,
				modelToUse
			);
		} catch (error) {
			if (!isAbortError(error))
				console.error('Failed to regenerate message with branching:', error);

			this.host.setChatLoading(activeConv?.id || '', false);
		}
	}

	async updateMessage(messageId: string, newContent: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;

		if (!activeConv) return;

		if (this.host.isChatLoadingInternal(activeConv.id)) await this.host.stopGeneration();

		const result = this.getMessageByIdWithRole(messageId, MessageRole.USER);

		if (!result) return;

		const { index: messageIndex, message: messageToUpdate } = result;
		const originalContent = messageToUpdate.content;

		try {
			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);
			const isFirstUserMessage = rootMessage && messageToUpdate.parent === rootMessage.id;

			conversationsStore.updateMessageAtIndex(messageIndex, { content: newContent });
			await DatabaseService.updateMessage(messageId, { content: newContent });

			if (isFirstUserMessage && newContent.trim())
				await conversationsStore.applyTitleFromContent(activeConv.id, newContent);

			const messagesToRemove = conversationsStore.activeMessages.slice(messageIndex + 1);

			if (messagesToRemove.length > 0)
				await DatabaseService.deleteMessageCascading(activeConv.id, messagesToRemove[0].id);

			conversationsStore.sliceActiveMessages(messageIndex + 1);
			conversationsStore.updateConversationTimestamp();
			this.host.setChatLoading(activeConv.id, true);
			this.host.clearChatStreaming(activeConv.id);
			const assistantMessage = await this.host.createAssistantMessage();

			conversationsStore.addMessageToActive(assistantMessage);
			await conversationsStore.updateCurrentNode(assistantMessage.id);
			await this.host.streamChatCompletion(
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

	/**
	 * Open a fresh assistant turn anchored at the last tool result of a resolved
	 * agentic round and let streamChatCompletion route through runAgenticFlow.
	 * Used by continueAssistantMessage when classifyContinueIntent returns
	 * next_turn, meaning the target assistant already has its tool_calls paired
	 * with trailing tool results and the next thing to generate is a brand new
	 * turn rather than a token level continuation.
	 */
	private async continueAsNextAgenticTurn(anchorIndex: number): Promise<void> {
		const activeConv = conversationsStore.activeConversation;

		if (!activeConv) return;

		const anchor = conversationsStore.activeMessages[anchorIndex];

		if (!anchor) return;

		this.host.cancelPreEncode();
		this.host.setChatLoading(activeConv.id, true);
		this.host.clearChatStreaming(activeConv.id);
		try {
			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const anchorMessage = findMessageById(allMessages, anchor.id);

			if (!anchorMessage) {
				this.host.setChatLoading(activeConv.id, false);

				return;
			}

			const newAssistantMessage = await DatabaseService.createMessageBranch(
				{
					children: [],
					content: '',
					convId: activeConv.id,
					model: null,
					role: MessageRole.ASSISTANT,
					timestamp: Date.now(),
					toolCalls: '',
					type: MessageType.TEXT
				},
				anchorMessage.id
			);

			await conversationsStore.updateCurrentNode(newAssistantMessage.id);
			conversationsStore.updateConversationTimestamp();
			await conversationsStore.refreshActiveMessages();
			const conversationPath = filterByLeafNodeId(
				allMessages,
				anchorMessage.id,
				false
			) as DatabaseMessage[];

			await this.host.streamChatCompletion(conversationPath, newAssistantMessage);
		} catch (error) {
			if (!isAbortError(error)) console.error('Failed to continue agentic turn:', error);

			this.host.setChatLoading(activeConv.id, false);
		}
	}

	private async generateResponseForMessage(userMessageId: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;

		if (!activeConv) return;

		this.host.showErrorDialog(null);
		this.host.setChatLoading(activeConv.id, true);
		this.host.clearChatStreaming(activeConv.id);

		try {
			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const conversationPath = filterByLeafNodeId(
				allMessages,
				userMessageId,
				false
			) as DatabaseMessage[];
			const assistantMessage = await DatabaseService.createMessageBranch(
				{
					children: [],
					content: '',
					convId: activeConv.id,
					model: null,
					role: MessageRole.ASSISTANT,
					timestamp: Date.now(),
					toolCalls: '',
					type: MessageType.TEXT
				},
				userMessageId
			);

			conversationsStore.addMessageToActive(assistantMessage);

			await this.host.streamChatCompletion(conversationPath, assistantMessage);
		} catch (error) {
			console.error('Failed to generate response:', error);
			this.host.setChatLoading(activeConv.id, false);
		}
	}

	private getMessageByIdWithRole(
		messageId: string,
		expectedRole?: MessageRole
	): { message: DatabaseMessage; index: number } | null {
		const index = conversationsStore.findMessageIndex(messageId);

		if (index === -1) return null;

		const message = conversationsStore.activeMessages[index];

		if (expectedRole && message.role !== expectedRole) return null;

		return { index, message };
	}
}
