<script module lang="ts">
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { ChatSidebarSearch } from '$lib/components/app';

	const { Story } = defineMeta({
		title: 'Components/ChatSidebar/ChatSidebarSearch',
		component: ChatSidebarSearch,
		parameters: {
			layout: 'centered'
		}
	});
</script>

<Story name="Default" args={{ class: 'w-80' }} />

<Story name="Focus" args={{ class: 'w-80' }} play={({ canvasElement }) => {
	const input = canvasElement.querySelector('input') as HTMLInputElement;
	if (input) {
		input.focus();
	}
}}/>

<Story
	name="Active"
	args={{
		class: 'w-80',
		value: 'Python programming'
	}}
/>
