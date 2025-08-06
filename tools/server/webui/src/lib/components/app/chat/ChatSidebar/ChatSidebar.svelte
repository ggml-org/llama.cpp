<script lang="ts">
	import * as Sidebar from '$lib/components/ui/sidebar/index.js';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Search, SquarePen, X } from '@lucide/svelte';
	import { ChatSidebarConversationItem } from '$lib/components/app';
	import { conversations, deleteConversation } from '$lib/stores/chat.svelte';
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { useSidebar } from '$lib/components/ui/sidebar';
	import ScrollArea from '$lib/components/ui/scroll-area/scroll-area.svelte';

	const sidebar = useSidebar();

	export function handleMobileSidebarItemClick() {
		if (sidebar.isMobile) {
			sidebar.toggle();
		}
	}

	let currentChatId = $derived(page.params.id);
	let searchQuery = $state('');

	let isSearchModeActive = $state(false);

	let filteredConversations = $derived.by(() => {
		if (isSearchModeActive && searchQuery.trim().length > 0) {
			return conversations().filter((conversation: { name: string }) =>
				conversation.name.toLowerCase().includes(searchQuery.toLowerCase())
			);
		}

		if (isSearchModeActive && searchQuery.trim().length === 0) {
			return [];
		}

		return conversations();
	});

	async function selectConversation(id: string) {
		await goto(`/chat/${id}`);
	}

	async function editConversation(id: string) {
		console.log('Editing conversation:', id);
	}

	async function handleDeleteConversation(id: string) {
		await deleteConversation(id);
	}
</script>

<Sidebar.Header class="px-1 pb-2 pt-4">
	<a href="/" onclick={handleMobileSidebarItemClick}>
		<h1 class="inline-flex items-center gap-1 text-xl font-semibold">llama.cpp</h1>
	</a>
</Sidebar.Header>

<div class="space-y-0.5 py-4">
	{#if isSearchModeActive}
		<div class="relative">
			<Search class="text-muted-foreground absolute left-2 top-2.5 h-4 w-4" />

			<Input bind:value={searchQuery} placeholder="Search conversations..." class="pl-8" />

			<X
				class="cursor-pointertext-muted-foreground absolute right-2 top-2.5 h-4 w-4"
				onclick={() => (isSearchModeActive = false)}
			/>
		</div>
	{:else}
		<Button
			class="w-full justify-start gap-2"
			href="/?new_chat=true"
			variant="ghost"
			onclick={handleMobileSidebarItemClick}
		>
			<SquarePen class="h-4 w-4" />

			New chat
		</Button>

		<Button
			class="w-full justify-start gap-2"
			variant="ghost"
			onclick={() => {
				isSearchModeActive = true;
			}}
		>
			<Search class="h-4 w-4" />

			Search conversations
		</Button>
	{/if}
</div>

<Sidebar.Group class="space-y-2 p-0">
	{#if (filteredConversations.length > 0 && isSearchModeActive) || !isSearchModeActive}
		<Sidebar.GroupLabel>
			{isSearchModeActive ? 'Search results' : 'Conversations'}
		</Sidebar.GroupLabel>
	{/if}

	<Sidebar.GroupContent>
		<Sidebar.Menu>
			<ScrollArea
				class={!isSearchModeActive
					? 'h-[calc(100vh-16.5rem)]'
					: 'h-[calc(100vh-11.625rem)]'}
			>
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
						<p class="text-muted-foreground text-sm">
							{searchQuery.length > 0
								? 'No results found'
								: isSearchModeActive
									? 'Start typing to see results'
									: 'No conversations yet'}
						</p>
					</div>
				{/if}
			</ScrollArea>
		</Sidebar.Menu>
	</Sidebar.GroupContent>
</Sidebar.Group>

<Sidebar.Footer>
	<div class="px-2 py-2">
		<p class="text-muted-foreground text-xs">
			Conversations are stored locally in your browser.
		</p>
	</div>
</Sidebar.Footer>
