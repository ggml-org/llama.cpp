<script lang="ts" module>
	import OverviewFrame from './llama-sans/overview/OverviewFrame.svelte';
	import TypeTester from './typography/TypeTester.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// The family's tester: one text in Llama Sans and Llama Mono together, with the size, weight, tracking and line
	// shared, Llama Sans's optical size auto or pinned, and Llama Mono's ligatures on or off
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Brand/Typography/Tester'
	});
</script>

<Story asChild name="Tester">
	<OverviewFrame>
		<TypeTester />
	</OverviewFrame>
</Story>
