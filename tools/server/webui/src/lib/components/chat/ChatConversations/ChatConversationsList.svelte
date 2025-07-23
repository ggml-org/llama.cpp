<script lang="ts">
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import ConversationItem from './ChatConversationsItem.svelte';
	import type { Conversation } from '$lib/types/conversation';

	interface Props {
		conversations: Conversation[];
		activeConversationId?: string;
		searchQuery?: string;
		onSelect?: (id: string) => void;
		onEdit?: (id: string) => void;
		onDelete?: (id: string) => void;
	}

	let {
		conversations,
		activeConversationId,
		searchQuery = '',
		onSelect,
		onEdit,
		onDelete
	}: Props = $props();
</script>

<ScrollArea class="flex-1">
	<div class="space-y-2">
		{#each conversations as conversation (conversation.id)}
			<ConversationItem
				{conversation}
				isActive={activeConversationId === conversation.id}
				{onSelect}
				{onEdit}
				{onDelete}
			/>
		{/each}

		{#if conversations.length === 0}
			<div class="text-muted-foreground py-8 text-center">
				{#if searchQuery}
					<p>No conversations found matching "{searchQuery}"</p>
				{:else}
					<p>No conversations yet</p>
					<p class="mt-1 text-sm">Start a new conversation to get started</p>
				{/if}
			</div>
		{/if}
	</div>
</ScrollArea>
