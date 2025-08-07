<script module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { ChatScreenHeader } from '$lib/components/app';

	const { Story } = defineMeta({
		title: 'Components/ChatScreenHeader',
		component: ChatScreenHeader,
		parameters: {
			layout: 'fullscreen'
		}
	});
</script>

<Story name="Default" />
