<script module lang="ts">
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { ChatSidebarConversationItem } from '$lib/components/app';

	const { Story } = defineMeta({
		title: 'Components/ChatSidebar/ChatSidebarConversationItem',
		component: ChatSidebarConversationItem,
		parameters: {
			layout: 'centered'
		}
	});

	const sampleConversation: DatabaseConversation = {
		id: '1',
		name: 'Hello World Chat',
		lastModified: Date.now() - 1000 * 60 * 5, // 5 minutes ago
		currNode: null
	};

	const longNameConversation: DatabaseConversation = {
		id: '2',
		name: 'This is a very long conversation name that should be truncated when displayed in the sidebar',
		lastModified: Date.now() - 1000 * 60 * 60 * 2, // 2 hours ago
		currNode: null
	};

	const recentConversation: DatabaseConversation = {
		id: '3',
		name: 'Recent Chat',
		lastModified: Date.now() - 1000 * 30, // 30 seconds ago
		currNode: null
	};
</script>

<Story
	name="Default"
	args={{
		conversation: sampleConversation
	}}
/>

<Story
	name="Active"
	args={{
		conversation: sampleConversation,
		isActive: true
	}}
/>

<Story
	name="LongName"
	args={{
		conversation: longNameConversation
	}}
/>

<Story
	name="Recent"
	args={{
		conversation: recentConversation
	}}
/>

<Story
	name="WithActions"
	args={{
		conversation: sampleConversation,
		onSelect: (id: string) => console.log('Selected:', id),
		onEdit: (id: string) => console.log('Edit:', id),
		onDelete: (id: string) => console.log('Delete:', id)
	}}
/>
