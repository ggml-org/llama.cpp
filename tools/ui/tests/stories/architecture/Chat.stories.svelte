<script lang="ts" module>
	import Page from './Chat.svx';
	import Frame from './Frame.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Architecture/Chat'
	});
</script>

<Story asChild name="Chat">
	<Frame>
		<Page />
	</Frame>
</Story>
