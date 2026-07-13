<script lang="ts">
	import { Pin } from '@lucide/svelte';
	import { buildConversationTree } from '$lib/stores/conversations.svelte';
	import SidebarNavigationConversationItem from './SidebarNavigationConversationItem.svelte';
	import SidebarNavigationSearchResults from './SidebarNavigationSearchResults.svelte';

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

	const recentEmptyMessage = $derived(
		searchQuery.length > 0 ? 'No results found' : 'No conversations yet'
	);
</script>

{#if isSearchModeActive}
	<SidebarNavigationSearchResults
		class={className}
		{searchQuery}
		{filteredConversations}
		{currentChatId}
		{onSelect}
		{onEdit}
		{onDelete}
		{onStop}
	/>
{:else}
	{#if pinnedConversations.length > 0}
		<div class="py-2 flex whitespace-nowrap {className}">
			<div
				class="text-muted-foreground inline-flex h-8 shrink-0 items-center rounded-md px-2 text-xs font-medium gap-1"
			>
				<Pin class="h-3.5 w-3.5" />

				<span>Pinned</span>
			</div>
		</div>

		<ul class="flex w-full min-w-0 flex-col gap-4 md:gap-1 {className}">
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

	<div class="mt-2 flex min-h-0 flex-1 flex-col gap-4 md:gap-2 whitespace-nowrap {className}">
		{#if filteredConversations.length > 0}
			<div
				class="text-muted-foreground flex h-8 shrink-0 items-center rounded-md px-2 text-xs font-medium"
			>
				Recent conversations
			</div>
		{/if}

		<div class="min-h-0 flex-1 md:overflow-y-auto">
			<ul class="flex w-full min-w-0 flex-col gap-4 md:gap-1">
				{#each unpinnedConversations as { conversation, depth } (conversation.id)}
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

				{#if unpinnedConversations.length === 0}
					<li class="px-2 py-4 text-center">
						<p class="mb-4 p-4 text-sm text-muted-foreground">
							{recentEmptyMessage}
						</p>
					</li>
				{/if}
			</ul>
		</div>
	</div>
{/if}
