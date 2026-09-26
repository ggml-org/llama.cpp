<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import SidebarNavigation from '$lib/components/app/navigation/SidebarNavigation/SidebarNavigation.svelte';
	import { waitFor } from 'storybook/test';
	import { screen } from 'storybook/test';

	const { Story } = defineMeta({
		component: SidebarNavigation,
		parameters: {
			layout: 'centered'
		},
		title: 'Components/SidebarNavigation'
	});
</script>

<script lang="ts">
	// Mock conversations for the sidebar
	const mockConversations: DatabaseConversation[] = [
		{
			currNode: 'msg-1',
			id: 'conv-1',
			lastModified: Date.now() - 1000 * 60 * 5, // 5 minutes ago
			name: 'Getting Started with AI'
		},
		{
			currNode: 'msg-2',
			id: 'conv-2',
			lastModified: Date.now() - 1000 * 60 * 60 * 2, // 2 hours ago
			name: 'Python Programming Help'
		},
		{
			currNode: 'msg-3',
			id: 'conv-3',
			lastModified: Date.now() - 1000 * 60 * 60 * 24, // 1 day ago
			name: 'Creative Writing Ideas'
		},
		{
			currNode: 'msg-4',
			id: 'conv-4',
			lastModified: Date.now() - 1000 * 60 * 60 * 24 * 3, // 3 days ago
			name: 'This is a very long conversation title that should be truncated properly when displayed'
		},
		{
			currNode: 'msg-5',
			id: 'conv-5',
			lastModified: Date.now() - 1000 * 60 * 60 * 24 * 7, // 1 week ago
			name: 'Math Problem Solving'
		}
	];
</script>

<Story
	asChild
	name="Default"
	play={async () => {
		const { conversationsStore } = await import('$lib/stores/conversations/index.svelte');

		waitFor(() =>
			setTimeout(() => {
				conversationsStore.conversations = mockConversations;
			}, 0)
		);
	}}
>
	<div class="flex-column h-screen w-72 bg-background">
		<SidebarNavigation />
	</div>
</Story>

<Story
	asChild
	name="SearchActive"
	play={async ({ userEvent }) => {
		const { conversationsStore } = await import('$lib/stores/conversations/index.svelte');

		waitFor(() =>
			setTimeout(() => {
				conversationsStore.conversations = mockConversations;
			}, 0)
		);

		// Expand sidebar first, then click Search in the expanded button list
		const logoTrigger = screen.getByRole('button', { name: /expand navigation/i });

		await userEvent.click(logoTrigger);
		const searchTrigger = screen.getByText('Search');

		userEvent.click(searchTrigger);
	}}
>
	<div class="flex-column h-screen w-72 bg-background">
		<SidebarNavigation />
	</div>
</Story>

<Story
	asChild
	name="Empty"
	play={async () => {
		// Mock empty conversations store
		const { conversationsStore } = await import('$lib/stores/conversations/index.svelte');

		conversationsStore.conversations = [];
	}}
>
	<div class="flex-column h-screen w-72 bg-background">
		<SidebarNavigation />
	</div>
</Story>
