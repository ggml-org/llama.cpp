<script lang="ts" module>
	import OverviewFrame from './llama-sans/overview/OverviewFrame.svelte';
	import TypographyOverview from './typography/TypographyOverview.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// The family at a glance: Llama Sans UI, Llama Sans Display and Llama Mono on one page, with a switcher at the top
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
		<TypographyOverview />
	</OverviewFrame>
</Story>
