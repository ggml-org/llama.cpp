<script lang="ts" module>
	import GlyphGrid from './llama-sans/lib/components/GlyphGrid.svelte';
	import SpecimenFrame from './llama-sans/SpecimenFrame.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Every glyph of Llama Sans, from the brand specimen: the outlines are the same at every optical size
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Brand/Typography/Llama Sans/Glyphs'
	});
</script>

<Story asChild name="Glyphs">
	<SpecimenFrame variant="UI">
		<GlyphGrid comparing={false} weight={400} />
	</SpecimenFrame>
</Story>
