<script lang="ts">
	import { Pin } from '@lucide/svelte';
	import { buildConversationTree } from '$lib/stores/conversations.svelte';
	import SidebarNavigationConversationItem from './SidebarNavigationConversationItem.svelte';
	import * as Sidebar from '$lib/components/ui/sidebar';

	interface Props {
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
	<Sidebar.Group class="p-0 px-4">
		<Sidebar.GroupLabel>
			<div class="flex items-center gap-1">
				<Pin class="h-3.5 w-3.5" />
				<span>Pinned</span>
			</div>
		</Sidebar.GroupLabel>
		<Sidebar.GroupContent>
			<Sidebar.Menu>
				{#each pinnedConversations as { conversation, depth } (conversation.id)}
					<Sidebar.MenuItem class="mb-1 p-0">
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
					</Sidebar.MenuItem>
				{/each}
			</Sidebar.Menu>
		</Sidebar.GroupContent>
	</Sidebar.Group>
{/if}

<Sidebar.Group class="mt-2 h-[calc(100vh-21rem)] space-y-2 p-0 px-3">
	{#if (filteredConversations.length > 0 && isSearchModeActive) || !isSearchModeActive}
		<Sidebar.GroupLabel>
			{isSearchModeActive ? 'Search results' : 'Recent conversations'}
		</Sidebar.GroupLabel>
	{/if}

	<Sidebar.GroupContent>
		<Sidebar.Menu>
			{#each visibleItems as { conversation, depth } (conversation.id)}
				<Sidebar.MenuItem class="mb-1 p-0">
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
				</Sidebar.MenuItem>
			{/each}

			{#if visibleItems.length === 0}
				<div class="px-2 py-4 text-center">
					<p class="mb-4 p-4 text-sm text-muted-foreground">
						{emptyMessage}
					</p>
				</div>
			{/if}
		</Sidebar.Menu>
	</Sidebar.GroupContent>
</Sidebar.Group>
