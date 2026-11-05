<script lang="ts" module>
	import GlyphGrid from './llama-sans/lib/components/GlyphGrid.svelte';
	import Metrics from './llama-sans/lib/components/Metrics.svelte';
	import Setting from './llama-sans/lib/components/Setting.svelte';
	import ShowcaseProse from './llama-sans/lib/components/ShowcaseProse.svelte';
	import ShowcaseUI from './llama-sans/lib/components/ShowcaseUI.svelte';
	import Tester from './llama-sans/lib/components/Tester.svelte';
	import Waterfall from './llama-sans/lib/components/Waterfall.svelte';
	import { specimen } from './llama-sans/lib/specimen';
	import SpecimenFrame from './llama-sans/SpecimenFrame.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Llama Sans UI, from the brand specimen: in controls and in running text, its weights, a tester and the glyphs
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Brand/Typography/Llama Sans/UI'
	});

	const { glyphCount } = specimen;
</script>

<Story asChild name="Interface">
	<SpecimenFrame variant="UI">
		<ShowcaseUI {glyphCount} guides />
	</SpecimenFrame>
</Story>

<Story asChild name="Running text">
	<SpecimenFrame variant="UI">
		<ShowcaseProse />
	</SpecimenFrame>
</Story>

<Story asChild name="Weights">
	<SpecimenFrame variant="UI">
		<Waterfall />

		{#each specimen.styles as s (s.weight)}
			<Setting weight={s.weight} />
		{/each}
	</SpecimenFrame>
</Story>

<Story asChild name="Tester">
	<SpecimenFrame variant="UI">
		<Metrics />

		<Tester {glyphCount} size={32} tracking={0} />
	</SpecimenFrame>
</Story>

<Story asChild name="Glyphs">
	<SpecimenFrame variant="UI">
		<GlyphGrid comparing={false} weight={400} />
	</SpecimenFrame>
</Story>
