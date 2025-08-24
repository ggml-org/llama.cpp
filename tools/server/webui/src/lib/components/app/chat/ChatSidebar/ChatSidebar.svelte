<script lang="ts">
	import * as Sidebar from '$lib/components/ui/sidebar/index.js';
	import { ChatSidebarConversationItem } from '$lib/components/app';
	import { conversations, deleteConversation, updateConversationName } from '$lib/stores/chat.svelte';
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { useSidebar } from '$lib/components/ui/sidebar';
	import ScrollArea from '$lib/components/ui/scroll-area/scroll-area.svelte';
	import ChatSidebarActions from './ChatSidebarActions.svelte';

	const sidebar = useSidebar();

	let currentChatId = $derived(page.params.id);
	let isSearchModeActive = $state(false);
	let searchQuery = $state('');

	let filteredConversations = $derived.by(() => {
		if (searchQuery.trim().length > 0) {
			return conversations().filter((conversation: { name: string }) =>
				conversation.name.toLowerCase().includes(searchQuery.toLowerCase())
			);
		}

		return conversations();
	});

	async function editConversation(id: string, name: string) {
		await updateConversationName(id, name);
	}

	async function handleDeleteConversation(id: string) {
		await deleteConversation(id);
	}

	export function handleMobileSidebarItemClick() {
		if (sidebar.isMobile) {
			sidebar.toggle();
		}
	}

	async function selectConversation(id: string) {
		if (isSearchModeActive) {
			isSearchModeActive = false;
			searchQuery = '';
		}

		await goto(`/chat/${id}`);
	}
</script>

<ScrollArea class="h-[100vh]">
	<Sidebar.Header class=" px-4 pb-2 pt-4 md:sticky top-0 bg-sidebar/50 backdrop-blur-lg z-10 gap-6">
		<a href="/" onclick={handleMobileSidebarItemClick}>
			<h1 class="inline-flex items-center gap-1 text-xl font-semibold px-2">llama.cpp</h1>
		</a>

		<ChatSidebarActions
			{handleMobileSidebarItemClick}
			bind:isSearchModeActive
			bind:searchQuery
		/>
	</Sidebar.Header>

	<Sidebar.Group class="space-y-2 mt-4 p-0 px-4">
		{#if (filteredConversations.length > 0 && isSearchModeActive) || !isSearchModeActive}
			<Sidebar.GroupLabel>
				{isSearchModeActive ? 'Search results' : 'Conversations'}
			</Sidebar.GroupLabel>
		{/if}

		<Sidebar.GroupContent>
			<Sidebar.Menu>
				{#each filteredConversations as conversation (conversation.id)}
					<Sidebar.MenuItem class="mb-1" onclick={handleMobileSidebarItemClick}>
						<ChatSidebarConversationItem
							conversation={{
								id: conversation.id,
								name: conversation.name,
								lastModified: conversation.lastModified,
								currNode: conversation.currNode
							}}
							isActive={currentChatId === conversation.id}
							onSelect={selectConversation}
							onEdit={editConversation}
							onDelete={handleDeleteConversation}
						/>
					</Sidebar.MenuItem>
				{/each}

				{#if filteredConversations.length === 0}
					<div class="px-2 py-4 text-center">
						<p class="text-muted-foreground text-sm p-4 mb-4">
							{searchQuery.length > 0
								? 'No results found'
								: isSearchModeActive
									? 'Start typing to see results'
									: 'No conversations yet'}
						</p>
					</div>
				{/if}

			</Sidebar.Menu>
		</Sidebar.GroupContent>
	</Sidebar.Group>
	
	<div class="md:sticky bottom-0 bg-sidebar z-10 px-4 py-4  bg-sidebar/50 backdrop-blur-lg">
		<p class="text-muted-foreground text-xs">
			Conversations are stored locally in your browser.
		</p>
	</div>
</ScrollArea>
