<script lang="ts" module>
	import GlyphGrid from './typography-showcase/lib/components/GlyphGrid.svelte';
	import InUse from './typography-showcase/lib/components/InUse.svelte';
	import Metrics from './typography-showcase/lib/components/Metrics.svelte';
	import Setting from './typography-showcase/lib/components/Setting.svelte';
	import Tester from './typography-showcase/lib/components/Tester.svelte';
	import Waterfall from './typography-showcase/lib/components/Waterfall.svelte';
	import { specimen } from './typography-showcase/lib/specimen';
	import SpecimenFrame from './typography-showcase/SpecimenFrame.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Llama Sans piece by piece, from the specimen (Typography/Showcase): Display at work, UI in controls and
	// running text, the weights, a tester and the glyphs
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Typography/Llama Sans'
	});

	const { glyphCount } = specimen;
</script>

<Story asChild name="Display">
	<SpecimenFrame variant="Display">
		<InUse {glyphCount} variant="Display" />
	</SpecimenFrame>
</Story>

<Story asChild name="UI">
	<SpecimenFrame variant="UI">
		<InUse {glyphCount} guides variant="UI" />
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

<Story asChild name="Glyphs">
	<SpecimenFrame variant={specimen.grid.variant}>
		<GlyphGrid comparing={false} weight={400} />
	</SpecimenFrame>
</Story>
