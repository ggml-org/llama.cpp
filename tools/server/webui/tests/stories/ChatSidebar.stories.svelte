<script module lang="ts">
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatSidebar from '$lib/components/app/chat/ChatSidebar/ChatSidebar.svelte';
	import { waitFor } from 'storybook/test';
	import { screen } from 'storybook/test';

	const { Story } = defineMeta({
		title: 'Components/ChatSidebar',
		component: ChatSidebar,
		parameters: {
			layout: 'centered'
		}
	});

	// Mock conversations for the sidebar
	const mockConversations: DatabaseConversation[] = [
		{
			id: 'conv-1',
			name: 'Getting Started with AI',
			lastModified: Date.now() - 1000 * 60 * 5, // 5 minutes ago
			currNode: 'msg-1'
		},
		{
			id: 'conv-2',
			name: 'Python Programming Help',
			lastModified: Date.now() - 1000 * 60 * 60 * 2, // 2 hours ago
			currNode: 'msg-2'
		},
		{
			id: 'conv-3',
			name: 'Creative Writing Ideas',
			lastModified: Date.now() - 1000 * 60 * 60 * 24, // 1 day ago
			currNode: 'msg-3'
		},
		{
			id: 'conv-4',
			name: 'This is a very long conversation title that should be truncated properly when displayed',
			lastModified: Date.now() - 1000 * 60 * 60 * 24 * 3, // 3 days ago
			currNode: 'msg-4'
		},
		{
			id: 'conv-5',
			name: 'Math Problem Solving',
			lastModified: Date.now() - 1000 * 60 * 60 * 24 * 7, // 1 week ago
			currNode: 'msg-5'
		}
	];
</script>

<Story
	asChild
	name="Default"
	play={async () => {
		const { conversationsStore } = await import('$lib/stores/conversations.svelte');
		
		waitFor(() => setTimeout(() => {
			conversationsStore.conversations = mockConversations;
		}, 0));
	}}
>
	<div class="flex-column h-full h-screen w-72 bg-background">
		<ChatSidebar />
	</div>
</Story>

<Story
	asChild
	name="SearchActive"
	play={async ({ userEvent }) => {
		const { conversationsStore } = await import('$lib/stores/conversations.svelte');
		
		waitFor(() => setTimeout(() => {
			conversationsStore.conversations = mockConversations;
		}, 0));
		
		const searchTrigger = screen.getByText('Search conversations');
		userEvent.click(searchTrigger);
	}}
>
	<div class="flex-column h-full h-screen w-72 bg-background">
		<ChatSidebar />
	</div>
</Story>

<Story
	asChild
	name="Empty"
	play={async () => {
		// Mock empty conversations store
		const { conversationsStore } = await import('$lib/stores/conversations.svelte');
		conversationsStore.conversations = [];
	}}
>
	<div class="flex-column h-full h-screen w-72 bg-background">
		<ChatSidebar />
	</div>
</Story>
