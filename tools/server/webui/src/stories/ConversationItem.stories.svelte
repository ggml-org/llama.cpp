<script module lang="ts">
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { ChatConversationsItem } from '$lib/components';
	import type { Conversation } from '$lib/types/conversation';

	const { Story } = defineMeta({
		title: 'Components/ChatSidebar/ChatConversationItem',
		component: ChatConversationsItem,
		parameters: {
			layout: 'centered'
		}
	});

	const sampleConversation: Conversation = {
		id: '1',
		name: 'Hello World Chat',
		lastModified: Date.now() - 1000 * 60 * 5, // 5 minutes ago
		messageCount: 12
	};

	const longNameConversation: Conversation = {
		id: '2',
		name: 'This is a very long conversation name that should be truncated when displayed in the sidebar',
		lastModified: Date.now() - 1000 * 60 * 60 * 2, // 2 hours ago
		messageCount: 24
	};

	const recentConversation: Conversation = {
		id: '3',
		name: 'Recent Chat',
		lastModified: Date.now() - 1000 * 30, // 30 seconds ago
		messageCount: 3
	};
</script>

<Story
	name="Default"
	args={{
		conversation: sampleConversation,
		class: 'w-80'
	}}
/>

<Story
	name="Active"
	args={{
		conversation: sampleConversation,
		isActive: true,
		class: 'w-80'
	}}
/>

<Story
	name="LongName"
	args={{
		conversation: longNameConversation,
		class: 'w-80'
	}}
/>

<Story
	name="Recent"
	args={{
		conversation: recentConversation,
		class: 'w-80'
	}}
/>

<Story
	name="WithActions"
	args={{
		conversation: sampleConversation,
		class: 'w-80',
		onSelect: (id: string) => console.log('Selected:', id),
		onEdit: (id: string) => console.log('Edit:', id),
		onDelete: (id: string) => console.log('Delete:', id)
	}}
/>
