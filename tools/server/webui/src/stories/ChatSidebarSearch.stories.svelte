<script module lang="ts">
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatSidebarSearch from '$lib/components/chat/ChatSidebar/ChatSidebarSearch.svelte';

	const { Story } = defineMeta({
		title: 'Components/ChatSidebar/ChatSidebarSearch',
		component: ChatSidebarSearch,
		parameters: {
			layout: 'centered'
		}
	});
</script>

<Story name="Default" args={{ class: 'w-80' }} />

<Story
	name="WithValue"
	args={{
		class: 'w-80',
		value: 'Hello World'
	}}
/>
