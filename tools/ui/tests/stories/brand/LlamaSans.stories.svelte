<script lang="ts" module>
	import GlyphGrid from './llama-sans/lib/components/GlyphGrid.svelte';
	import ShowcaseDisplay from './llama-sans/lib/components/ShowcaseDisplay.svelte';
	import ShowcaseProse from './llama-sans/lib/components/ShowcaseProse.svelte';
	import SpecimenFrame from './llama-sans/SpecimenFrame.svelte';
	import UserInterface from './typography/UserInterface.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Llama Sans in use, from its UI optical size (opsz 16) to its Display one (opsz 40): the design system's own
	// components, running text, headlines and large type, and every glyph, from the brand specimen
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Brand/Typography/Llama Sans'
	});
</script>

<Story asChild name="User Interface">
	<UserInterface />
</Story>

<Story asChild name="Running text">
	<SpecimenFrame variant="UI">
		<ShowcaseProse />
	</SpecimenFrame>
</Story>

<Story asChild name="Display">
	<SpecimenFrame variant="Display">
		<ShowcaseDisplay />
	</SpecimenFrame>
</Story>

<Story asChild name="Glyphs">
	<SpecimenFrame variant="UI">
		<GlyphGrid comparing={false} weight={400} />
	</SpecimenFrame>
</Story>
