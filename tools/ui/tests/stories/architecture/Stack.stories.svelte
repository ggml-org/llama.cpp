<script lang="ts" module>
	import Frame from './Frame.svelte';
	import Page from './Stack.svx';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Architecture/Stack'
	});
</script>

<Story asChild name="Stack">
	<Frame>
		<Page />
	</Frame>
</Story>
