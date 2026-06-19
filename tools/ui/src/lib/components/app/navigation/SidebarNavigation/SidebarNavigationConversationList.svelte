<script lang="ts">
	import { Pin } from '@lucide/svelte';
	import { buildConversationTree } from '$lib/stores/conversations.svelte';
	import SidebarNavigationConversationItem from './SidebarNavigationConversationItem.svelte';

	interface Props {
		class: string;
		filteredConversations: DatabaseConversation[];
		currentChatId: string | undefined;
		isSearchModeActive: boolean;
		searchQuery: string;
		onSelect: (id: string) => void;
		onEdit: (id: string) => void;
		onDelete: (id: string) => void;
		onStop: (id: string) => void;
	}

	let {
		class: className,
		filteredConversations,
		currentChatId,
		isSearchModeActive,
		searchQuery,
		onSelect,
		onEdit,
		onDelete,
		onStop
	}: Props = $props();

	let conversationTree = $derived(buildConversationTree(filteredConversations));

	let pinnedConversations = $derived(
		conversationTree.filter(({ conversation }) => conversation.pinned)
	);

	let unpinnedConversations = $derived(
		conversationTree.filter(({ conversation }) => !conversation.pinned)
	);

	// In search mode the tree is already filtered down to query matches, so
	// pinned/unpinned splits are not relevant and would just duplicate them.
	let visibleItems = $derived(isSearchModeActive ? conversationTree : unpinnedConversations);

	const emptyMessage = $derived.by(() => {
		if (searchQuery.length > 0) return 'No results found';
		if (isSearchModeActive) return 'Start typing to see results';
		return 'No conversations yet';
	});
</script>

{#if !isSearchModeActive && pinnedConversations.length > 0}
	<div class="py-2 flex whitespace-nowrap {className}">
		<div
			class="text-muted-foreground inline-flex h-8 shrink-0 items-center rounded-md px-2 text-xs font-medium gap-1"
		>
			<Pin class="h-3.5 w-3.5" />

			<span>Pinned</span>
		</div>
	</div>

	<ul class="flex w-full min-w-0 flex-col gap-1 {className}">
		{#each pinnedConversations as { conversation, depth } (conversation.id)}
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
	</ul>
{/if}

<div class="mt-2 flex min-h-0 flex-1 flex-col gap-2 whitespace-nowrap {className}">
	{#if (filteredConversations.length > 0 && isSearchModeActive) || !isSearchModeActive}
		<div
			class="text-muted-foreground flex h-8 shrink-0 items-center rounded-md px-2 text-xs font-medium"
		>
			{isSearchModeActive ? 'Search results' : 'Recent conversations'}
		</div>
	{/if}

	<div class="min-h-0 flex-1 overflow-y-auto">
		<ul class="flex w-full min-w-0 flex-col gap-1">
			{#each visibleItems as { conversation, depth } (conversation.id)}
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

			{#if visibleItems.length === 0}
				<div class="px-2 py-4 text-center">
					<p class="mb-4 p-4 text-sm text-muted-foreground">
						{emptyMessage}
					</p>
				</div>
			{/if}
		</ul>
	</div>
</div>
