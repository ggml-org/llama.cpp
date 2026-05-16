<script lang="ts">
	import { onMount } from 'svelte';
	import { beforeNavigate, afterNavigate } from '$app/navigation';
	import { ChatMessage, ChatMessageUserPending } from '$lib/components/app';
	import { setChatActionsContext } from '$lib/contexts';
	import { MessageRole } from '$lib/enums';
	import { chatStore } from '$lib/stores/chat.svelte';
	import {
		chatPendingMessageContent,
		chatPendingMessageExtras,
		chatClearPendingMessage,
		chatInjectPendingMessage
	} from '$lib/stores/chat.svelte';
	import { conversationsStore, activeConversation } from '$lib/stores/conversations.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import {
		agenticPendingSteeringMessageContent,
		agenticPendingSteeringMessageExtras,
		agenticClearSteeringMessage,
		agenticInjectSteeringMessage
	} from '$lib/stores/agentic.svelte';
	import {
		copyToClipboard,
		formatMessageForClipboard,
		getMessageSiblings,
		hasAgenticContent
	} from '$lib/utils';

	interface Props {
		messages?: DatabaseMessage[];
		onUserAction?: () => void;
		onMessagesReady?: (messageCount: number) => void;
	}

	let { messages = [], onUserAction, onMessagesReady }: Props = $props();

	let allConversationMessages = $state<DatabaseMessage[]>([]);
	let isVisible = $state(false);
	let previousConversationId = $state<string | null>(null);

	const currentConfig = config();

	setChatActionsContext({
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

		navigateToSibling: async (siblingId: string) => {
			await conversationsStore.navigateToSibling(siblingId);
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

		editUserMessagePreserveResponses: async (
			message: DatabaseMessage,
			newContent: string,
			newExtras?: DatabaseMessageExtra[]
		) => {
			onUserAction?.();
			await chatStore.editUserMessagePreserveResponses(message.id, newContent, newExtras);
			refreshAllMessages();
		},

		regenerateWithBranching: async (message: DatabaseMessage, modelOverride?: string) => {
			onUserAction?.();
			await chatStore.regenerateMessageWithBranching(message.id, modelOverride);
			refreshAllMessages();
		},

		continueAssistantMessage: async (message: DatabaseMessage) => {
			onUserAction?.();
			await chatStore.continueAssistantMessage(message.id);
			refreshAllMessages();
		},

		forkConversation: async (
			message: DatabaseMessage,
			options: { name: string; includeAttachments: boolean }
		) => {
			await conversationsStore.forkConversation(message.id, options);
		}
	});

	function refreshAllMessages() {
		const conversation = activeConversation();

		if (conversation) {
			conversationsStore.getConversationMessages(conversation.id).then((messages) => {
				allConversationMessages = messages;
			});
		} else {
			allConversationMessages = [];
		}
	}

	// Track conversation changes to trigger transition even on same route
	$effect(() => {
		const conversation = activeConversation();
		const currentId = conversation?.id ?? null;

		if (currentId !== previousConversationId && previousConversationId !== null) {
			// Conversation changed - trigger fade out/in
			isVisible = false;
			requestAnimationFrame(() => {
				refreshAllMessages();
				previousConversationId = currentId;
				requestAnimationFrame(() => {
					isVisible = true;
				});
			});
		} else {
			previousConversationId = currentId;
			if (conversation) {
				refreshAllMessages();
			}
		}
	});

	$effect(() => {
		void allConversationMessages;

		onMessagesReady?.(displayMessages.length);
	});

	onMount(() => {
		requestAnimationFrame(() => {
			isVisible = true;
		});
	});

	beforeNavigate(() => {
		isVisible = false;
	});

	afterNavigate(() => {
		requestAnimationFrame(() => {
			isVisible = true;
		});
	});

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

			const siblingInfo = getMessageSiblings(allConversationMessages, msg.id);

			result.push({
				message: msg,
				toolMessages,
				isLastAssistantMessage: false,
				siblingInfo: siblingInfo || {
					message: msg,
					siblingIds: [msg.id],
					currentIndex: 0,
					totalSiblings: 1
				}
			});
		}

		// Mark the last assistant message
		for (let i = result.length - 1; i >= 0; i--) {
			if (result[i].message.role === MessageRole.ASSISTANT) {
				result[i].isLastAssistantMessage = true;
				break;
			}
		}

		return result;
	});
</script>

<div
	class="transition-opacity delay-300 duration-500 ease-out
		{isVisible ? 'opacity-100' : 'opacity-0'}"
>
	{#each displayMessages as { message, toolMessages, isLastAssistantMessage, siblingInfo } (message.id)}
		<ChatMessage
			class="mx-auto mt-12 w-full max-w-[48rem]"
			{message}
			{toolMessages}
			{isLastAssistantMessage}
			{siblingInfo}
		/>
	{/each}

	{#if activeConversation() && agenticPendingSteeringMessageContent(activeConversation()!.id)}
		{@const convId = activeConversation()!.id}
		{@const pendingContent = agenticPendingSteeringMessageContent(convId)}

		{#if pendingContent}
			<ChatMessageUserPending
				class="mx-auto mt-12 w-full max-w-[48rem]"
				content={pendingContent}
				extras={agenticPendingSteeringMessageExtras(convId)}
				onSendImmediately={() => chatStore.abortCurrentFlow(convId)}
				onEdit={(newContent, extras) => agenticInjectSteeringMessage(convId, newContent, extras)}
				onDelete={() => agenticClearSteeringMessage(convId)}
			/>
		{/if}
	{:else if activeConversation() && chatPendingMessageContent(activeConversation()!.id)}
		{@const convId = activeConversation()!.id}
		{@const pendingContent = chatPendingMessageContent(convId)}

		{#if pendingContent}
			<ChatMessageUserPending
				class="mx-auto mt-12 w-full max-w-[48rem]"
				content={pendingContent}
				extras={chatPendingMessageExtras(convId)}
				onSendImmediately={() => chatStore.abortCurrentFlow(convId)}
				onEdit={(newContent, extras) => chatInjectPendingMessage(convId, newContent, extras)}
				onDelete={() => chatClearPendingMessage(convId)}
			/>
		{/if}
	{/if}
</div>
