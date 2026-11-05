<script lang="ts" module>
	import Metrics from './llama-sans/lib/components/Metrics.svelte';
	import Setting from './llama-sans/lib/components/Setting.svelte';
	import ShowcaseDisplay from './llama-sans/lib/components/ShowcaseDisplay.svelte';
	import Tester from './llama-sans/lib/components/Tester.svelte';
	import Waterfall from './llama-sans/lib/components/Waterfall.svelte';
	import { specimen } from './llama-sans/lib/specimen';
	import SpecimenFrame from './llama-sans/SpecimenFrame.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Llama Sans Display, from the brand specimen: headlines and large type at work, its weights and a tester
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Brand/Typography/Llama Sans/Display'
	});

	const { glyphCount } = specimen;
</script>

<Story asChild name="In use">
	<SpecimenFrame variant="Display">
		<ShowcaseDisplay />
	</SpecimenFrame>
</Story>

<Story asChild name="Weights">
	<SpecimenFrame variant="Display">
		<Waterfall />

		{#each specimen.styles as s (s.weight)}
			<Setting weight={s.weight} />
		{/each}
	</SpecimenFrame>
</Story>

<Story asChild name="Tester">
	<SpecimenFrame variant="Display">
		<Metrics />

		<Tester {glyphCount} size={64} tracking={0} />
	</SpecimenFrame>
</Story>
