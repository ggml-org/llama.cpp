<script lang="ts" module>
	import Overview from './llama-sans/Overview.svx';
	import OverviewFrame from './llama-sans/overview/OverviewFrame.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Llama Sans at a glance, before its UI and Display versions: the name at full size, both versions, the character
	// set, the weights and the design notes
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Brand/Typography/Overview'
	});
</script>

<Story asChild name="Overview">
	<OverviewFrame>
		<Overview />
	</OverviewFrame>
</Story>
