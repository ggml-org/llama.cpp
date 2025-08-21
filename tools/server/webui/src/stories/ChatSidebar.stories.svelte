<script module lang="ts">
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatSidebar from '$lib/components/app/chat/ChatSidebar/ChatSidebar.svelte';

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

<Story name="Default">
	<div class="bg-background h-screen w-80 border-r">
		<ChatSidebar />
	</div>
</Story>

<Story 
	name="SearchActive"
	play={async ({ canvasElement }) => {
		// Wait for component to mount
		await new Promise(resolve => setTimeout(resolve, 100));
		
		// Find and interact with search input
		const searchInput = canvasElement.querySelector('input[placeholder*="Search"]') as HTMLInputElement;
		if (searchInput) {
			searchInput.focus();
			searchInput.value = 'Python';
			searchInput.dispatchEvent(new Event('input', { bubbles: true }));
		}
	}}
>
	<div class="bg-background h-screen w-80 border-r">
		<ChatSidebar />
	</div>
</Story>
