<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { ChatSidebarConversationItem } from '$lib/components/app';
	import ScrollArea from '$lib/components/ui/scroll-area/scroll-area.svelte';
	import * as Sidebar from '$lib/components/ui/sidebar';
	import {
		conversations,
		deleteConversation,
		updateConversationName
	} from '$lib/stores/chat.svelte';
	import ChatSidebarActions from './ChatSidebarActions.svelte';

	const sidebar = Sidebar.useSidebar();

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

	export function activateSearchMode() {
		isSearchModeActive = true;
	}

	export function editActiveConversation() {
		if (currentChatId) {
			const activeConversation = filteredConversations.find((conv) => conv.id === currentChatId);

			if (activeConversation) {
				const event = new CustomEvent('edit-active-conversation', {
					detail: { conversationId: currentChatId }
				});
				document.dispatchEvent(event);
			}
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
	<Sidebar.Header class=" top-0 z-10 gap-6 bg-sidebar/50 px-4 pt-4 pb-2 backdrop-blur-lg md:sticky">
		<a href="/" onclick={handleMobileSidebarItemClick}>
			<h1 class="inline-flex items-center gap-1 px-2 text-xl font-semibold">llama.cpp</h1>
		</a>

		<ChatSidebarActions {handleMobileSidebarItemClick} bind:isSearchModeActive bind:searchQuery />
	</Sidebar.Header>

	<Sidebar.Group class="mt-4 space-y-2 p-0 px-4">
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
						<p class="mb-4 p-4 text-sm text-muted-foreground">
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

	<div class="bottom-0 z-10 bg-sidebar bg-sidebar/50 px-4 py-4 backdrop-blur-lg md:sticky">
		<p class="text-xs text-muted-foreground">Conversations are stored locally in your browser.</p>
	</div>
</ScrollArea>
