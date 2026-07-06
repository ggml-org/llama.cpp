<script lang="ts">
	import { buildConversationTree } from '$lib/stores/conversations.svelte';
	import SidebarNavigationConversationItem from './SidebarNavigationConversationItem.svelte';

	interface Props {
		class?: string;
		searchQuery: string;
		filteredConversations: DatabaseConversation[];
		currentChatId: string | undefined;
		onSelect: (id: string) => void;
		onEdit: (id: string) => void;
		onDelete: (id: string) => void;
		onStop: (id: string) => void;
	}

	let {
		class: className = '',
		searchQuery,
		filteredConversations,
		currentChatId,
		onSelect,
		onEdit,
		onDelete,
		onStop
	}: Props = $props();

	let tree = $derived(buildConversationTree(filteredConversations));

	const hasQuery = $derived(searchQuery.trim().length > 0);
	const showHeader = $derived(hasQuery && filteredConversations.length > 0);

	const emptyMessage = $derived(hasQuery ? 'No results found' : 'Start typing to see results');
</script>

<div class="flex min-h-0 flex-1 flex-col gap-2 whitespace-nowrap {className}">
	{#if showHeader}
		<div
			class="text-muted-foreground flex h-8 shrink-0 items-center rounded-md px-2 text-xs font-medium"
		>
			Search results
		</div>
	{/if}

	<div class="min-h-0 flex-1 overflow-y-auto">
		<ul class="flex w-full min-w-0 flex-col gap-1">
			{#each tree as { conversation, depth } (conversation.id)}
				<li class="group/item relative mb-1 p-0">
					<SidebarNavigationConversationItem
						conversation={{
							id: conversation.id,
							name: conversation.name,
							lastModified: conversation.lastModified,
							currNode: conversation.currNode,
							forkedFromConversationId: conversation.forkedFromConversationId,
							pinned: conversation.pinned
						}}
						{depth}
						isActive={currentChatId === conversation.id}
						{onSelect}
						{onEdit}
						{onDelete}
						{onStop}
					/>
				</li>
			{/each}

			{#if tree.length === 0}
				<li class="px-2 py-4 text-center">
					<p class="mb-4 p-4 text-sm text-muted-foreground">
						{emptyMessage}
					</p>
				</li>
			{/if}
		</ul>
	</div>
</div>
