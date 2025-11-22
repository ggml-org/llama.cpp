<script lang="ts">
	import { ChatMessage } from '$lib/components/app';
	import { DatabaseStore } from '$lib/stores/database';
	import {
		activeConversation,
		continueAssistantMessage,
		deleteMessage,
		editAssistantMessage,
		editMessageWithBranching,
		editUserMessagePreserveResponses,
		navigateToSibling,
		regenerateMessageWithBranching
	} from '$lib/stores/chat.svelte';
	import { getMessageSiblings } from '$lib/utils/branching';

	interface Props {
		class?: string;
		messages?: DatabaseMessage[];
		onUserAction?: () => void;
	}

	let { class: className, messages = [], onUserAction }: Props = $props();

	let allConversationMessages = $state<DatabaseMessage[]>([]);

	function refreshAllMessages() {
		const conversation = activeConversation();

		if (conversation) {
			DatabaseStore.getConversationMessages(conversation.id).then((messages) => {
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

		return messages.map((message) => {
			const siblingInfo = getMessageSiblings(allConversationMessages, message.id);

			return {
				message,
				siblingInfo: siblingInfo || {
					message,
					siblingIds: [message.id],
					currentIndex: 0,
					totalSiblings: 1
				}
			};
		});
	});

	async function handleNavigateToSibling(siblingId: string) {
		await navigateToSibling(siblingId);
	}

	async function handleEditWithBranching(message: DatabaseMessage, newContent: string) {
		onUserAction?.();

		await editMessageWithBranching(message.id, newContent);

		refreshAllMessages();
	}

	async function handleEditWithReplacement(
		message: DatabaseMessage,
		newContent: string,
		shouldBranch: boolean
	) {
		onUserAction?.();

		await editAssistantMessage(message.id, newContent, shouldBranch);

		refreshAllMessages();
	}

	async function handleRegenerateWithBranching(message: DatabaseMessage) {
		onUserAction?.();

		await regenerateMessageWithBranching(message.id);

		refreshAllMessages();
	}

	async function handleContinueAssistantMessage(message: DatabaseMessage) {
		onUserAction?.();

		await continueAssistantMessage(message.id);

		refreshAllMessages();
	}

	async function handleEditUserMessagePreserveResponses(
		message: DatabaseMessage,
		newContent: string
	) {
		onUserAction?.();

		await editUserMessagePreserveResponses(message.id, newContent);

		refreshAllMessages();
	}

	async function handleDeleteMessage(message: DatabaseMessage) {
		await deleteMessage(message.id);

		refreshAllMessages();
	}
</script>

<div class="flex h-full flex-col space-y-10 pt-16 md:pt-24 {className}" style="height: auto; ">
	{#each displayMessages as { message, siblingInfo } (message.id)}
		<ChatMessage
			class="mx-auto w-full max-w-[48rem]"
			{message}
			{siblingInfo}
			onDelete={handleDeleteMessage}
			onNavigateToSibling={handleNavigateToSibling}
			onEditWithBranching={handleEditWithBranching}
			onEditWithReplacement={handleEditWithReplacement}
			onEditUserMessagePreserveResponses={handleEditUserMessagePreserveResponses}
			onRegenerateWithBranching={handleRegenerateWithBranching}
			onContinueAssistantMessage={handleContinueAssistantMessage}
		/>
	{/each}
</div>
