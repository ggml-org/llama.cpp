<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import { Badge } from '$lib/components/ui/badge';
	import { 
		PlusIcon, 
		MagnifyingGlassIcon, 
		ChatBubbleLeftIcon,
		EllipsisVerticalIcon,
		TrashIcon,
		PencilIcon
	} from '@heroicons/svelte/24/outline';

	// Mock data for now - will be replaced with real store data
	let conversations = [
		{ id: '1', name: 'Hello World Chat', lastModified: Date.now() - 1000 * 60 * 5, messageCount: 12 },
		{ id: '2', name: 'Code Review Discussion', lastModified: Date.now() - 1000 * 60 * 30, messageCount: 8 },
		{ id: '3', name: 'Project Planning', lastModified: Date.now() - 1000 * 60 * 60 * 2, messageCount: 24 }
	];

	let searchQuery = '';
	let activeConversationId = '1';

	$: filteredConversations = conversations.filter(conv => 
		conv.name.toLowerCase().includes(searchQuery.toLowerCase())
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

<div class="flex flex-col h-full p-4">
	<!-- Header -->
	<div class="flex items-center justify-between mb-4">
		<h2 class="text-lg font-semibold">Conversations</h2>
		<Button size="sm" onclick={createNewConversation} class="h-8 w-8 p-0">
			<PlusIcon class="h-4 w-4" />
		</Button>
	</div>

	<!-- Search -->
	<div class="relative mb-4">
		<MagnifyingGlassIcon class="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
		<Input 
			bind:value={searchQuery}
			placeholder="Search conversations..."
			class="pl-10"
		/>
	</div>

	<!-- Conversations List -->
	<ScrollArea class="flex-1">
		<div class="space-y-2">
			{#each filteredConversations as conversation (conversation.id)}
				<div 
					class="group flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-colors hover:bg-accent {activeConversationId === conversation.id ? 'bg-accent border-primary' : 'border-border'}"
					onclick={() => selectConversation(conversation.id)}
				>
					<div class="flex items-center space-x-3 flex-1 min-w-0">
						<ChatBubbleLeftIcon class="h-5 w-5 text-muted-foreground flex-shrink-0" />
						<div class="flex-1 min-w-0">
							<p class="text-sm font-medium truncate">{conversation.name}</p>
							<div class="flex items-center space-x-2 mt-1">
								<Badge variant="secondary" class="text-xs">
									{conversation.messageCount} messages
								</Badge>
								<span class="text-xs text-muted-foreground">
									{formatLastModified(conversation.lastModified)}
								</span>
							</div>
						</div>
					</div>
					
					<!-- Actions (visible on hover) -->
					<div class="opacity-0 group-hover:opacity-100 transition-opacity flex items-center space-x-1">
						<Button 
							size="sm" 
							variant="ghost" 
							class="h-6 w-6 p-0"
							onclick={(e) => { e.stopPropagation(); /* TODO: Edit name */ }}
						>
							<PencilIcon class="h-3 w-3" />
						</Button>
						<Button 
							size="sm" 
							variant="ghost" 
							class="h-6 w-6 p-0 text-destructive hover:text-destructive"
							onclick={(e) => { e.stopPropagation(); deleteConversation(conversation.id); }}
						>
							<TrashIcon class="h-3 w-3" />
						</Button>
					</div>
				</div>
			{/each}
			
			{#if filteredConversations.length === 0}
				<div class="text-center py-8 text-muted-foreground">
					{#if searchQuery}
						<p>No conversations found matching "{searchQuery}"</p>
					{:else}
						<p>No conversations yet</p>
						<p class="text-sm mt-1">Start a new conversation to get started</p>
					{/if}
				</div>
			{/if}
		</div>
	</ScrollArea>

	<!-- Footer -->
	<div class="mt-4 pt-4 border-t">
		<p class="text-xs text-muted-foreground text-center">
			Conversations are saved locally in your browser
		</p>
	</div>
</div>
