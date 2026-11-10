<script lang="ts" module>
	import { FAMILIES } from './catalog';
	import Frames from './Frames.svelte';
	import HeaderVariant from './HeaderVariant.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// llama.app's model page header, in six layouts to choose from: where the maker's avatar goes against the name, and
	// where the maker and release go. Each at a desktop width and a phone's, for a short name (Gemma 4) and a longer
	// one (EmbeddingGemma 2)
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Scrapbook/UI/llama.app/Model page header'
	});

	const pick = (name: string) => FAMILIES.find((family) => family.name === name) ?? FAMILIES[0];
	const SHORT = pick('Gemma 4');
	const LONG = pick('EmbeddingGemma 2');
</script>

<!-- A: the avatar on the name's line, the facts under the name (llama.app today) -->
<Story asChild name="A Inline">
	<Frames><HeaderVariant family={SHORT} layout="inline" /><HeaderVariant family={LONG} layout="inline" /></Frames>
</Story>

<!-- B: the avatar on the name's line, the facts at the avatar's edge -->
<Story asChild name="B Flush">
	<Frames><HeaderVariant family={SHORT} layout="flush" /><HeaderVariant family={LONG} layout="flush" /></Frames>
</Story>

<!-- C: Llama Chat's Discover details, a larger avatar beside the name and the facts -->
<Story asChild name="C Discover">
	<Frames><HeaderVariant family={SHORT} layout="discover" /><HeaderVariant family={LONG} layout="discover" /></Frames>
</Story>

<!-- D: the avatar above the name, like an app's icon over its title -->
<Story asChild name="D Stacked">
	<Frames><HeaderVariant family={SHORT} layout="stacked" /><HeaderVariant family={LONG} layout="stacked" /></Frames>
</Story>

<!-- E: the maker and release as an eyebrow over the name -->
<Story asChild name="E Eyebrow">
	<Frames><HeaderVariant family={SHORT} layout="eyebrow" /><HeaderVariant family={LONG} layout="eyebrow" /></Frames>
</Story>

<!-- F: no line of facts, the maker and release join the chips -->
<Story asChild name="F Chips">
	<Frames><HeaderVariant family={SHORT} layout="chips" /><HeaderVariant family={LONG} layout="chips" /></Frames>
</Story>
