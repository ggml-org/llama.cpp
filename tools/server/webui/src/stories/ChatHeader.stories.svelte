<script module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatHeader from '$lib/components/ChatHeader.svelte';

	const { Story } = defineMeta({
		title: 'Components/ChatHeader',
		component: ChatHeader,
		parameters: {
			layout: 'fullscreen'
		}
	});
</script>

<Story name="Default" />
