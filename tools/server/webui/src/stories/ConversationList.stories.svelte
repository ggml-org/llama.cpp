<script module lang="ts">
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ConversationList from '$lib/components/chat/ChatConversations/ChatConversationsList.svelte';
	import type { Conversation } from '$lib/types/conversation';

	const { Story } = defineMeta({
		title: 'Components/ChatSidebar/ConversationList',
		component: ConversationList,
		parameters: {
			layout: 'centered'
		}
	});

	const sampleConversations: Conversation[] = [
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
</script>

<Story
	name="WithConversations"
	args={{
		conversations: sampleConversations,
		activeConversationId: '1',
		class: 'w-80 h-96'
	}}
/>

<Story
	name="Empty"
	args={{
		conversations: [],
		class: 'w-80 h-96'
	}}
/>

<Story
	name="EmptyWithSearch"
	args={{
		conversations: [],
		searchQuery: 'nonexistent',
		class: 'w-80 h-96'
	}}
/>

<Story
	name="WithActions"
	args={{
		conversations: sampleConversations,
		activeConversationId: '2',
		class: 'w-80 h-96',
		onSelect: (id: string) => console.log('Selected:', id),
		onEdit: (id: string) => console.log('Edit:', id),
		onDelete: (id: string) => console.log('Delete:', id)
	}}
/>
