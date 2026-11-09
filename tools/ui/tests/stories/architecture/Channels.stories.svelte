<script lang="ts" module>
	import Page from './Channels.svx';
	import Frame from './Frame.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Architecture/Channels'
	});
</script>

<Story asChild name="Channels">
	<Frame>
		<Page />
	</Frame>
</Story>
