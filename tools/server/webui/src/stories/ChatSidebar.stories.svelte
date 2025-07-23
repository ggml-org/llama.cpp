<script module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatSidebar from '$lib/components/ChatSidebar.svelte';

	const { Story } = defineMeta({
		title: 'Components/ChatSidebar',
		component: ChatSidebar,
		parameters: {
			layout: 'centered'
		}
	});
</script>

<Story name="Default">
	<div class="bg-background h-screen w-80 border-r">
		<ChatSidebar />
	</div>
</Story>
