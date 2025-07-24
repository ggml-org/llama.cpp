<script lang="ts">
	import * as Sidebar from '$lib/components/ui/sidebar/index.js';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Plus, Search } from '@lucide/svelte';
	import ChatConversationsItem from '$lib/components/chat/ChatConversations/ChatConversationsItem.svelte';
	import { chats, activeChat, createChat, updateChatName, deleteChat } from '$lib/stores/chat.svelte';
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';

	// Sidebar state
	let sidebarOpen = $state(true);
	let searchQuery = $state('');

	// Get current chat ID from URL
	const currentChatId = $derived($page.params.id);

	// Filter chats based on search query
	let filteredChats = $derived(
		chats().filter((chat) => chat.name.toLowerCase().includes(searchQuery.toLowerCase()))
	);

	async function createNewConversation() {
		await createChat();
	}

	async function selectConversation(id: string) {
		await goto(`/chat/${id}`);
	}

	async function editConversation(id: string) {
		// TODO: Implement inline editing
		console.log('Editing conversation:', id);
	}

	async function handleDeleteConversation(id: string) {
		await deleteChat(id);
	}
</script>

<!-- Header -->
<Sidebar.Header>
	<div class="px-2 py-2">
		<a href="/">
			<h1 class="text-xl font-semibold">llama.cpp</h1>
		</a>
	</div>
</Sidebar.Header>

<!-- Search -->
<div class="px-2 pb-2">
	<div class="relative">
		<Search class="text-muted-foreground absolute left-2 top-2.5 h-4 w-4" />
		<Input
			bind:value={searchQuery}
			placeholder="Search conversations..."
			class="pl-8"
		/>
	</div>
</div>

<!-- New Chat Button -->
<div class="px-2 pb-4">
	<Button 
		onclick={createNewConversation} 
		class="w-full justify-start gap-2 rounded-lg border-2 border-dashed border-muted-foreground/25 bg-transparent hover:bg-accent hover:border-accent-foreground/25 transition-colors"
		variant="ghost"
	>
		<Plus class="h-4 w-4" />
		New Chat
		<div class="ml-auto flex items-center gap-1 text-xs text-muted-foreground">
			<span class="rounded border px-1.5 py-0.5">⌘</span>
			<span class="rounded border px-1.5 py-0.5">K</span>
		</div>
	</Button>
</div>

<!-- Conversations -->
<Sidebar.Group class="space-y-2">
	<Sidebar.GroupLabel>Conversations</Sidebar.GroupLabel>
	<Sidebar.GroupContent>
		<Sidebar.Menu class="space-y-2">
			{#each filteredChats as chat (chat.id)}
				<Sidebar.MenuItem>
					<ChatConversationsItem
						conversation={{
							id: chat.id,
							name: chat.name,
							lastModified: chat.updatedAt,
							messageCount: chat.messageCount
						}}
						isActive={currentChatId === chat.id}
						onSelect={selectConversation}
						onEdit={editConversation}
						onDelete={handleDeleteConversation}
					/>
				</Sidebar.MenuItem>
			{/each}

			{#if filteredChats.length === 0}
				<div class="px-2 py-4 text-center">
					<p class="text-muted-foreground text-sm">
						{searchQuery
							? 'No conversations found'
							: 'No conversations yet'}
					</p>
			</div>
			{/if}
		</Sidebar.Menu>
	</Sidebar.GroupContent>
</Sidebar.Group>

<!-- Footer -->
<Sidebar.Footer>
	<div class="px-2 py-2">
		<p class="text-muted-foreground text-xs">
			Conversations are stored locally in your browser.
		</p>
	</div>
</Sidebar.Footer>
