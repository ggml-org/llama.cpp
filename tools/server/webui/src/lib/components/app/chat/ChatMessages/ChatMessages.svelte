<script lang="ts">
	import { ChatMessage } from '$lib/components/app';
	import { setChatActionsContext } from '$lib/contexts';
	import { MessageRole } from '$lib/enums';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { conversationsStore, activeConversation } from '$lib/stores/conversations.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { copyToClipboard, formatMessageForClipboard, getMessageSiblings } from '$lib/utils';

	interface Props {
		class?: string;
		messages?: DatabaseMessage[];
		onUserAction?: () => void;
	}

	let { class: className, messages = [], onUserAction }: Props = $props();

	let allConversationMessages = $state<DatabaseMessage[]>([]);
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

	// Single effect that tracks both conversation and message changes
	$effect(() => {
		const conversation = activeConversation();

		if (conversation) {
			refreshAllMessages();
		}
	});

	let displayMessages = $derived.by(() => {
		if (!messages.length) {
			return [];
		}

		const filteredMessages = currentConfig.showSystemMessage
			? messages
			: messages.filter((msg) => msg.type !== MessageRole.SYSTEM);

		let lastAssistantIndex = -1;

		for (let i = filteredMessages.length - 1; i >= 0; i--) {
			if (filteredMessages[i].role === MessageRole.ASSISTANT) {
				lastAssistantIndex = i;

				break;
			}
		}

		return filteredMessages.map((message, index) => {
			const siblingInfo = getMessageSiblings(allConversationMessages, message.id);
			const isLastAssistantMessage =
				message.role === MessageRole.ASSISTANT && index === lastAssistantIndex;

			return {
				message,
				isLastAssistantMessage,
				siblingInfo: siblingInfo || {
					message,
					siblingIds: [message.id],
					currentIndex: 0,
					totalSiblings: 1
				}
			};
		});
	});
</script>

<div class="flex h-full flex-col space-y-10 pt-24 {className}" style="height: auto; ">
	{#each displayMessages as { message, isLastAssistantMessage, siblingInfo } (message.id)}
		<ChatMessage
			class="mx-auto w-full max-w-[48rem]"
			{message}
			{isLastAssistantMessage}
			{siblingInfo}
		/>
	{/each}
</div>
