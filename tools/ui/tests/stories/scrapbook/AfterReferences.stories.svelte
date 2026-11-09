<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import FieldCards from './after-references/FieldCards.svelte';
	import IndexCard from './after-references/IndexCard.svelte';
	import Series from './after-references/Series.svelte';
	import SkyLabels from './after-references/SkyLabels.svelte';
	import SplitField from './after-references/SplitField.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// New key visuals learnt from four reference sets (project cards on soft fields, a poster triplet, grainy posters
	// with glass pills, and web cards with pixelated fields): one field of artwork per piece, crisp type on soft
	// texture, hierarchy by size and ink rather than weight, a hairline signature, numbered series and inset panels.
	// Each is drawn with the brand's grounds, fonts and motifs. Grain and glass go past BRAND.md's rules today (no
	// gradients but the skies, no glows), so choosing one of those means changing the rule too
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Scrapbook/Communications/After the references'
	});
</script>

<Story asChild name="A · Field cards">
	<StateGallery minItemWidth="22rem">
		<StateGalleryItem
			label="Landscape, light"
			note="The family as three cards, each one field: the night sky, a quantised sky for the engine, the day sky"
			span={2}
		>
			<FieldCards format="landscape" tone="light" />
		</StateGalleryItem>

		<StateGalleryItem label="Portrait, light" note="One card to a slide, as a carousel would run">
			<FieldCards format="portrait" tone="light" />
		</StateGalleryItem>

		<StateGalleryItem label="Landscape, dark" span={2}>
			<FieldCards format="landscape" tone="dark" />
		</StateGalleryItem>

		<StateGalleryItem label="Story, sand">
			<FieldCards format="story" tone="sand" />
		</StateGalleryItem>
	</StateGallery>
</Story>

<Story asChild name="B · Series of three">
	<StateGallery minItemWidth="22rem">
		<StateGalleryItem
			label="1 · Say it"
			note="Stepped words on ink, the place in the series as [1]"
		>
			<Series part={1} />
		</StateGalleryItem>

		<StateGalleryItem label="2 · The mark" note="The lockup alone on the night sky">
			<Series part={2} />
		</StateGalleryItem>

		<StateGalleryItem
			label="3 · Explain it"
			note="One paragraph on sand, signed with the icon over a hairline"
		>
			<Series part={3} />
		</StateGalleryItem>
	</StateGallery>
</Story>

<Story asChild name="C · Split field">
	<StateGallery minItemWidth="22rem">
		<StateGalleryItem label="Portrait, light" note="A quantised day sky over a plain panel">
			<SplitField format="portrait" tone="light" />
		</StateGalleryItem>

		<StateGalleryItem label="Portrait, dark" note="The night sky, quantised">
			<SplitField format="portrait" tone="dark" />
		</StateGalleryItem>

		<StateGalleryItem label="Story, light">
			<SplitField format="story" split={0.5} tone="light" />
		</StateGalleryItem>

		<StateGalleryItem label="Square, sand">
			<SplitField format="square" split={0.38} tone="sand" />
		</StateGalleryItem>
	</StateGallery>
</Story>

<Story asChild name="D · Labels on the sky">
	<StateGallery minItemWidth="22rem">
		<StateGalleryItem
			label="Portrait, night"
			note="Three glass labels stepped down the sky, the lockup small"
		>
			<SkyLabels format="portrait" sky="night" />
		</StateGalleryItem>

		<StateGalleryItem label="Portrait, day">
			<SkyLabels format="portrait" sky="day" />
		</StateGalleryItem>

		<StateGalleryItem label="Story, night">
			<SkyLabels format="story" sky="night" />
		</StateGalleryItem>

		<StateGalleryItem label="Square, day">
			<SkyLabels format="square" sky="day" />
		</StateGalleryItem>
	</StateGallery>
</Story>

<Story asChild name="E · Index card">
	<StateGallery minItemWidth="22rem">
		<StateGalleryItem
			label="Portrait, light"
			note="A site's header bar, the headline, the motif in an inset"
		>
			<IndexCard format="portrait" tone="light" />
		</StateGalleryItem>

		<StateGalleryItem label="Portrait, sand">
			<IndexCard format="portrait" tone="sand" />
		</StateGalleryItem>

		<StateGalleryItem label="Portrait, dark">
			<IndexCard format="portrait" tone="dark" />
		</StateGalleryItem>

		<StateGalleryItem label="Story, light">
			<IndexCard format="story" tone="light" />
		</StateGalleryItem>
	</StateGallery>
</Story>
