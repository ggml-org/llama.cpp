<script lang="ts">
	import {
		deleteMessage,
		navigateToSibling,
		editMessageWithBranching,
		regenerateMessageWithBranching
	} from '$lib/stores/chat.svelte';
	import { activeConversation, activeMessages } from '$lib/stores/chat.svelte';
	import { ChatMessage } from '$lib/components/app';
	import { getMessageSiblings } from '$lib/utils/branching';
	import { DatabaseService } from '$lib/services/database';

	interface Props {
		class?: string;
		messages?: DatabaseMessage[];
	}

	let { class: className, messages = [] }: Props = $props();

	let allConversationMessages = $state<DatabaseMessage[]>([]);
	let lastUpdateTime = $state(0);
	
	function refreshAllMessages() {
		const conversation = activeConversation();
		if (conversation) {
			DatabaseService.getConversationMessages(conversation.id).then(messages => {
				allConversationMessages = messages;
				lastUpdateTime = Date.now();
			});
		} else {
			allConversationMessages = [];
		}
	}
	
	// Single effect that tracks both conversation and message changes
	$effect(() => {
		const conversation = activeConversation();
		const currentActiveMessages = activeMessages();
		
		// Track message count and timestamps to detect changes
		const messageCount = currentActiveMessages.length;
		const lastMessageTimestamp = currentActiveMessages.length > 0 ? 
			Math.max(...currentActiveMessages.map(m => m.timestamp || 0)) : 0;
		
		if (conversation) {
			refreshAllMessages();
		}
	});

	let displayMessages = $derived.by(() => {
		if (!messages.length) {
			return [];
		}

		// Force dependency on lastUpdateTime to ensure reactivity
		const _ = lastUpdateTime;

		return messages.map(message => {
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
		await editMessageWithBranching(message.id, newContent);
		// Refresh after editing to update sibling counts
		refreshAllMessages();
	}
	async function handleRegenerateWithBranching(message: DatabaseMessage) {
		await regenerateMessageWithBranching(message.id);
		// Refresh after regenerating to update sibling counts
		refreshAllMessages();
	}
	async function handleDeleteMessage(message: DatabaseMessage) {
		await deleteMessage(message.id);
		// Refresh after deleting to update sibling counts
		refreshAllMessages();
	}
</script>

<div class="flex h-full flex-col space-y-10 pt-16 md:pt-24 {className}" style="height: auto; ">
	{#each displayMessages as { message, siblingInfo }}
		<ChatMessage
			class="mx-auto w-full max-w-[48rem]"
			{message}
			{siblingInfo}
			onDelete={handleDeleteMessage}
			onNavigateToSibling={handleNavigateToSibling}
			onEditWithBranching={handleEditWithBranching}
			onRegenerateWithBranching={handleRegenerateWithBranching}
		/>
	{/each}
</div>
