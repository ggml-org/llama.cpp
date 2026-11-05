<script lang="ts" module>
	import { seedChatStory } from '../chat/messages/fixtures';
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import Boards from './communications/Boards.svelte';
	import PaletteRow from './communications/PaletteRow.svelte';
	import { basePalette } from './communications/palettes';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Colour for communications (posters, the site, social): the neutral key visuals, the design system's greys on
	// every surface, field and accent with the real UI on it as it looks in the app; and the colour artwork, specimen
	// boards and posters in the brand palette (the art colours, beige and cinder)
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Brand/Colors/Communications'
	});
</script>

<Story asChild beforeEach={() => seedChatStory()} name="Key visuals">
	<StateGallery columns={1}>
		<StateGalleryItem label={basePalette.name} note={basePalette.note}>
			<PaletteRow palette={basePalette} />
		</StateGalleryItem>
	</StateGallery>
</Story>

<Story asChild name="Artwork">
	<StateGallery columns={1}>
		<StateGalleryItem
			label="Specimen boards and posters"
			note="After the classic Swiss specimen boards, in the brand palette: the art colours, beige and cinder"
		>
			<Boards />
		</StateGalleryItem>
	</StateGallery>
</Story>
