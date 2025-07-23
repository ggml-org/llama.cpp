<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import { Badge } from '$lib/components/ui/badge';
	import { Plus, Search, MessageCircle, Trash2, Pencil } from '@lucide/svelte';

	// Mock data for now - will be replaced with real store data
	let conversations = [
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

	function deleteConversation(id: string) {
		console.log('Deleting conversation:', id);
		// TODO: Implement delete logic
	}

	function formatLastModified(timestamp: number) {
		const now = Date.now();
		const diff = now - timestamp;
		const minutes = Math.floor(diff / (1000 * 60));
		const hours = Math.floor(diff / (1000 * 60 * 60));
		const days = Math.floor(diff / (1000 * 60 * 60 * 24));

		if (minutes < 1) return 'Just now';
		if (minutes < 60) return `${minutes}m ago`;
		if (hours < 24) return `${hours}h ago`;
		return `${days}d ago`;
	}
</script>

<div class="flex h-full flex-col p-4">
	<!-- Header -->
	<div class="mb-4 flex items-center justify-between">
		<h2 class="text-lg font-semibold">Conversations</h2>
		<Button size="sm" onclick={createNewConversation} class="h-8 w-8 p-0">
			<Plus class="h-4 w-4" />
		</Button>
	</div>

	<!-- Search -->
	<div class="relative mb-4">
		<Search
			class="text-muted-foreground absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 transform"
		/>
		<Input bind:value={searchQuery} placeholder="Search conversations..." class="pl-10" />
	</div>

	<!-- Conversations List -->
	<ScrollArea class="flex-1">
		<div class="space-y-2">
			{#each filteredConversations as conversation (conversation.id)}
				<button
					class="hover:bg-accent group flex cursor-pointer items-center justify-between rounded-lg border p-3 transition-colors {activeConversationId ===
					conversation.id
						? 'bg-accent border-primary'
						: 'border-border'} w-full text-left"
					onclick={() => selectConversation(conversation.id)}
				>
					<div class="flex min-w-0 flex-1 items-center space-x-3">
						<MessageCircle class="text-muted-foreground h-5 w-5 flex-shrink-0" />
						<div class="min-w-0 flex-1">
							<p class="truncate text-sm font-medium">{conversation.name}</p>
							<div class="mt-1 flex items-center space-x-2">
								<Badge variant="secondary" class="text-xs">
									{conversation.messageCount} messages
								</Badge>
								<span class="text-muted-foreground text-xs">
									{formatLastModified(conversation.lastModified)}
								</span>
							</div>
						</div>
					</div>

					<!-- Actions (visible on hover) -->
					<div
						class="flex items-center space-x-1 opacity-0 transition-opacity group-hover:opacity-100"
					>
						<Button
							size="sm"
							variant="ghost"
							class="h-6 w-6 p-0"
							onclick={(e) => {
								e.stopPropagation(); /* TODO: Edit name */
							}}
						>
							<Pencil class="h-3 w-3" />
						</Button>
						<Button
							size="sm"
							variant="ghost"
							class="text-destructive hover:text-destructive h-6 w-6 p-0"
							onclick={(e) => {
								e.stopPropagation();
								deleteConversation(conversation.id);
							}}
						>
							<Trash2 class="h-3 w-3" />
						</Button>
					</div>
				</button>
			{/each}

			{#if filteredConversations.length === 0}
				<div class="text-muted-foreground py-8 text-center">
					{#if searchQuery}
						<p>No conversations found matching "{searchQuery}"</p>
					{:else}
						<p>No conversations yet</p>
						<p class="mt-1 text-sm">Start a new conversation to get started</p>
					{/if}
				</div>
			{/if}
		</div>
	</ScrollArea>

	<!-- Footer -->
	<div class="mt-4 border-t pt-4">
		<p class="text-muted-foreground text-center text-xs">
			Conversations are saved locally in your browser
		</p>
	</div>
</div>
