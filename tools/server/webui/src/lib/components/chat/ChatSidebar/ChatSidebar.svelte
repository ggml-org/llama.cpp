<script lang="ts">
	import * as Sidebar from '$lib/components/ui/sidebar/index.js';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Plus, Search } from '@lucide/svelte';
	import ChatConversationsItem from '$lib/components/chat/ChatConversations/ChatConversationsItem.svelte';
	import type { Conversation } from '$lib/types/conversation';

	// Sidebar state
	let sidebarOpen = $state(true);

	let conversations: Conversation[] = [
		{
			id: '1',
			name: 'Hello World Chat',
			lastModified: Date.now() - 1000 * 60 * 5,
			messageCount: 12
		},
		{
			id: '2',
			name: 'Code Review Discussion',
			lastModified: Date.now() - 1000 * 60 * 30,
			messageCount: 8
		},
		{
			id: '3',
			name: 'Project Planning',
			lastModified: Date.now() - 1000 * 60 * 60 * 2,
			messageCount: 24
		}
	];

	let searchQuery = $state('');
	let activeConversationId = $state('1');

	let filteredConversations = $derived(
		conversations.filter((conv) => conv.name.toLowerCase().includes(searchQuery.toLowerCase()))
	);

	function createNewConversation() {
		console.log('Creating new conversation...');
		// TODO: Implement new conversation logic
	}

	function selectConversation(id: string) {
		activeConversationId = id;
		console.log('Selected conversation:', id);
		// TODO: Navigate to conversation
	}

	function editConversation(id: string) {
		console.log('Editing conversation:', id);
		// TODO: Implement edit logic
	}

	function deleteConversation(id: string) {
		console.log('Deleting conversation:', id);
		// TODO: Implement delete logic
	}
</script>

<Sidebar.Provider bind:open={sidebarOpen}>
	<Sidebar.Root>
		<Sidebar.Content>
			<!-- Header -->
			<Sidebar.Header>
				<div class="flex items-center justify-between px-2 py-2">
					<a href="/">
						<h1 class="text-xl font-semibold">llama.cpp</h1>
					</a>

					<Button size="sm" onclick={createNewConversation} class="h-8 w-8 p-0">
						<Plus class="h-4 w-4" />
					</Button>
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

			<!-- Conversations -->
			<Sidebar.Group class="space-y-2">
				<Sidebar.GroupLabel>Conversations</Sidebar.GroupLabel>
				<Sidebar.GroupContent>
					<Sidebar.Menu class="space-y-2">
						{#each filteredConversations as conversation (conversation.id)}
							<Sidebar.MenuItem>
								<ChatConversationsItem
									{conversation}
									isActive={activeConversationId === conversation.id}
									onSelect={selectConversation}
									onEdit={editConversation}
									onDelete={deleteConversation}
								/>
							</Sidebar.MenuItem>
						{/each}

						{#if filteredConversations.length === 0}
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
		</Sidebar.Content>
	</Sidebar.Root>
</Sidebar.Provider>
