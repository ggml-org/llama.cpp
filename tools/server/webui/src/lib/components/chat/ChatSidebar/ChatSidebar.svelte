<script lang="ts">
	import * as Sidebar from '$lib/components/ui/sidebar/index.js';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Plus, Search } from '@lucide/svelte';
	import ChatSidebarConversationItem from '$lib/components/chat/ChatSidebar/ChatSidebarConversationItem.svelte';
	import { conversations, deleteConversation } from '$lib/stores/chat.svelte';
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { useSidebar } from '$lib/components/ui/sidebar';

	const sidebar = useSidebar();

	export function handleMobileSidebarItemClick() {
		if (sidebar.isMobile) {
			sidebar.toggle();
		}
	}

	let currentChatId = $derived(page.params.id);
	let searchQuery = $state('');
	let filteredConversations = $derived(
		conversations().filter((conversation: { name: string }) =>
			conversation.name.toLowerCase().includes(searchQuery.toLowerCase())
		)
	);

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

<Sidebar.Header class="px-0 pb-4">
	<div class="py-2">
		<a href="/" onclick={handleMobileSidebarItemClick}>
			<h1 class="text-xl font-semibold">llama.cpp</h1>
		</a>
	</div>
</Sidebar.Header>

<div class="relative pb-4">
	<Search class="text-muted-foreground absolute left-2 top-2.5 h-4 w-4" />
	<Input bind:value={searchQuery} placeholder="Search conversations..." class="pl-8" />
</div>

<div class="pb-4">
	<Button
		href="/?new_chat=true"
		class="border-muted-foreground/25 hover:bg-accent hover:border-accent-foreground/25 w-full justify-start gap-2 rounded-lg border-2 border-dashed bg-transparent transition-colors"
		variant="ghost"
		onclick={handleMobileSidebarItemClick}
	>
		<Plus class="h-4 w-4" />

		New Chat
	</Button>
</div>

<Sidebar.Group class="space-y-2 p-0">
	<Sidebar.GroupLabel>Conversations</Sidebar.GroupLabel>

	<Sidebar.GroupContent>
		<Sidebar.Menu class="space-y-0.5">
			{#each filteredConversations as conversation (conversation.id)}
				<Sidebar.MenuItem onclick={handleMobileSidebarItemClick}>
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
						{searchQuery ? 'No conversations found' : 'No conversations yet'}
					</p>
				</div>
			{/if}
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
