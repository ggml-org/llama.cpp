<script lang="ts">
	import { ChatMessage, ChatMessageUserPending } from '$lib/components/app';
	import { MessageRole } from '$lib/enums';
	import { agenticStore, chatStore, conversationsStore, settingsStore } from '$lib/stores';
	import type { ChatMessageActions } from '$lib/types';
	import {
		buildSiblingInfoMap,
		copyToClipboard,
		formatMessageForClipboard,
		hasAgenticContent
	} from '$lib/utils';

	interface Props {
		messages?: DatabaseMessage[];
		onUserAction?: () => void;
		onMessagesReady?: (messageCount: number) => void;
	}

	let { messages = [], onMessagesReady, onUserAction }: Props = $props();

	let allConversationMessages = $state<DatabaseMessage[]>([]);

	const currentConfig = settingsStore.config;

	const chatActions: ChatMessageActions = {
		continueAssistantMessage: async (message: DatabaseMessage) => {
			onUserAction?.();
			await chatStore.continueAssistantMessage(message.id);
			refreshAllMessages();
		},

		copy: async (message: DatabaseMessage) => {
			const asPlainText = Boolean(currentConfig.copyTextAttachmentsAsPlainText);
			const clipboardContent = formatMessageForClipboard(
				message.content,
				message.extra,
				asPlainText
			);

			await copyToClipboard(clipboardContent, 'Message copied to clipboard');
		},

		delete: async (message: DatabaseMessage) => {
			await chatStore.deleteMessage(message.id);
			refreshAllMessages();
		},

		editUserMessagePreserveResponses: async (
			message: DatabaseMessage,
			newContent: string,
			newExtras?: DatabaseMessageExtra[]
		) => {
			onUserAction?.();
			await chatStore.editUserMessagePreserveResponses(message.id, newContent, newExtras);
			refreshAllMessages();
		},

		editWithBranching: async (
			message: DatabaseMessage,
			newContent: string,
			newExtras?: DatabaseMessageExtra[]
		) => {
			onUserAction?.();
			await chatStore.editMessageWithBranching(message.id, newContent, newExtras);
			refreshAllMessages();
		},

		editWithReplacement: async (
			message: DatabaseMessage,
			newContent: string,
			shouldBranch: boolean
		) => {
			onUserAction?.();
			await chatStore.editAssistantMessage(message.id, newContent, shouldBranch);
			refreshAllMessages();
		},

		forkConversation: async (
			message: DatabaseMessage,
			options: { name: string; includeAttachments: boolean }
		) => {
			await conversationsStore.forkConversation(message.id, options);
		},

		navigateToSibling: async (siblingId: string) => {
			await conversationsStore.navigateToSibling(siblingId);
		},

		regenerateWithBranching: async (message: DatabaseMessage, modelOverride?: string) => {
			onUserAction?.();
			await chatStore.regenerateMessageWithBranching(message.id, modelOverride);
			refreshAllMessages();
		}
	};

	function refreshAllMessages() {
		const conversation = conversationsStore.activeConversation;

		if (conversation) {
			conversationsStore.getConversationMessages(conversation.id).then((messages) => {
				allConversationMessages = messages;
			});
		} else {
			allConversationMessages = [];
		}
	}

	// Refresh messages whenever the active conversation changes
	$effect(() => {
		if (conversationsStore.activeConversation) {
			refreshAllMessages();
		}
	});

	$effect(() => {
		void allConversationMessages;

		onMessagesReady?.(displayMessages.length);
	});

	let siblingInfoByMessageId = $derived(buildSiblingInfoMap(allConversationMessages));

	let displayMessages = $derived.by(() => {
		if (!messages.length) {
			return [];
		}

		const filteredMessages = currentConfig.showSystemMessage
			? messages
			: messages.filter((msg) => msg.type !== MessageRole.SYSTEM);
		// Build display entries, grouping agentic sessions into single entries.
		// An agentic session = assistant(with tool_calls) → tool → assistant → tool → ... → assistant(final)
		const result: Array<{
			message: DatabaseMessage;
			toolMessages: DatabaseMessage[];
			isLastAssistantMessage: boolean;
			isLastUserMessage: boolean;
			nextAssistantMessage: DatabaseMessage | null;
			siblingInfo: ChatMessageSiblingInfo;
		}> = [];

		for (let i = 0; i < filteredMessages.length; i++) {
			const msg = filteredMessages[i];

			// Skip tool messages - they're grouped with preceding assistant
			if (msg.role === MessageRole.TOOL) continue;

			const toolMessages: DatabaseMessage[] = [];

			if (msg.role === MessageRole.ASSISTANT && hasAgenticContent(msg)) {
				let j = i + 1;

				while (j < filteredMessages.length) {
					const next = filteredMessages[j];

					if (next.role === MessageRole.TOOL) {
						toolMessages.push(next);

						j++;
					} else if (next.role === MessageRole.ASSISTANT) {
						toolMessages.push(next);

						j++;
					} else {
						break;
					}
				}

				i = j - 1;
			} else if (msg.role === MessageRole.ASSISTANT) {
				let j = i + 1;

				while (j < filteredMessages.length && filteredMessages[j].role === MessageRole.TOOL) {
					toolMessages.push(filteredMessages[j]);
					j++;
				}
			}

			const siblingInfo = siblingInfoByMessageId.get(msg.id) ?? {
				currentIndex: 0,
				message: msg,
				siblingIds: [msg.id],
				totalSiblings: 1
			};

			result.push({
				isLastAssistantMessage: false,
				isLastUserMessage: false,
				message: msg,
				nextAssistantMessage: null,
				siblingInfo,
				toolMessages
			});
		}

		let lastAssistantIdx = -1;

		for (let i = result.length - 1; i >= 0; i--) {
			if (result[i].message.role === MessageRole.ASSISTANT) {
				result[i].isLastAssistantMessage = true;
				lastAssistantIdx = i;

				break;
			}
		}

		if (lastAssistantIdx > 0 && result[lastAssistantIdx - 1].message.role === MessageRole.USER) {
			result[lastAssistantIdx - 1].isLastUserMessage = true;
		}

		for (let i = 0; i < result.length; i++) {
			if (result[i].message.role !== MessageRole.USER) continue;

			for (let j = i + 1; j < result.length; j++) {
				if (result[j].message.role === MessageRole.ASSISTANT) {
					result[i].nextAssistantMessage = result[j].message;

					break;
				}
			}
		}

		return result;
	});
</script>

<div>
	{#each displayMessages as { isLastAssistantMessage, isLastUserMessage, message, nextAssistantMessage, siblingInfo, toolMessages } (message.id)}
		<ChatMessage
			{chatActions}
			class="mx-auto mt-12 w-full max-w-3xl"
			{isLastAssistantMessage}
			{isLastUserMessage}
			{message}
			{nextAssistantMessage}
			{siblingInfo}
			{toolMessages}
		/>
	{/each}

	{#if conversationsStore.activeConversation && agenticStore.getPendingSteeringMessageContent(conversationsStore.activeConversation!.id)}
		{@const convId = conversationsStore.activeConversation!.id}
		{@const pendingContent = agenticStore.getPendingSteeringMessageContent(convId)}

		{#if pendingContent}
			<ChatMessageUserPending
				class="mx-auto mt-12 w-full max-w-[48rem]"
				content={pendingContent}
				extras={agenticStore.getPendingSteeringMessageExtras(convId)}
				onDelete={() => agenticStore.clearSteeringMessage(convId)}
				onEdit={(newContent, extras) =>
					agenticStore.injectSteeringMessage(convId, newContent, extras)}
				onSendImmediately={() => chatStore.abortCurrentFlow(convId)}
			/>
		{/if}
	{:else if conversationsStore.activeConversation && chatStore.getPendingMessageContent(conversationsStore.activeConversation!.id)}
		{@const convId = conversationsStore.activeConversation!.id}
		{@const pendingContent = chatStore.getPendingMessageContent(convId)}

		{#if pendingContent}
			<ChatMessageUserPending
				class="mx-auto mt-12 w-full max-w-[48rem]"
				content={pendingContent}
				extras={chatStore.getPendingMessageExtras(convId)}
				onDelete={() => chatStore.clearPendingMessage(convId)}
				onEdit={(newContent, extras) => chatStore.injectPendingMessage(convId, newContent, extras)}
				onSendImmediately={() => chatStore.abortCurrentFlow(convId)}
			/>
		{/if}
	{/if}
</div>
