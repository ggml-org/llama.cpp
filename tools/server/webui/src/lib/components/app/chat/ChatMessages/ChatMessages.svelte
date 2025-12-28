<script lang="ts">
	import { ChatMessage } from '$lib/components/app';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { conversationsStore, activeConversation } from '$lib/stores/conversations.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { getMessageSiblings } from '$lib/utils';

	interface Props {
		class?: string;
		messages?: DatabaseMessage[];
		onUserAction?: () => void;
	}

	let { class: className, messages = [], onUserAction }: Props = $props();

	let allConversationMessages = $state<DatabaseMessage[]>([]);
	const currentConfig = config();

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

		// Filter out system messages if showSystemMessage is false
		const filteredMessages = currentConfig.showSystemMessage
			? messages
			: messages.filter((msg) => msg.type !== 'system');

		return filteredMessages.map((message) => {
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
		await conversationsStore.navigateToSibling(siblingId);
	}

	async function handleEditWithBranching(
		message: DatabaseMessage,
		newContent: string,
		newExtras?: DatabaseMessageExtra[]
	) {
		onUserAction?.();

		await chatStore.editMessageWithBranching(message.id, newContent, newExtras);

		refreshAllMessages();
	}

	async function handleEditWithReplacement(
		message: DatabaseMessage,
		newContent: string,
		shouldBranch: boolean
	) {
		onUserAction?.();

		await chatStore.editAssistantMessage(message.id, newContent, shouldBranch);

		refreshAllMessages();
	}

	async function handleRegenerateWithBranching(message: DatabaseMessage, modelOverride?: string) {
		onUserAction?.();

		await chatStore.regenerateMessageWithBranching(message.id, modelOverride);

		refreshAllMessages();
	}

	async function handleContinueAssistantMessage(message: DatabaseMessage) {
		onUserAction?.();

		await chatStore.continueAssistantMessage(message.id);

		refreshAllMessages();
	}

	async function handleEditUserMessagePreserveResponses(
		message: DatabaseMessage,
		newContent: string,
		newExtras?: DatabaseMessageExtra[]
	) {
		onUserAction?.();

		await chatStore.editUserMessagePreserveResponses(message.id, newContent, newExtras);

		refreshAllMessages();
	}

	async function handleDeleteMessage(message: DatabaseMessage) {
		await chatStore.deleteMessage(message.id);

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
