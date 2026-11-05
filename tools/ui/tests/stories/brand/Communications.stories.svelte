<script lang="ts" module>
	import { seedChatStory } from '../chat/messages/fixtures';
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import PaletteRow from './communications/PaletteRow.svelte';
	import { basePalette } from './communications/palettes';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Colour for communications (posters, the site, social): the neutral base, the design system's greys on every
	// surface, field and accent, with the real UI on it as it looks in the app
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Brand/Colors/Communications'
	});
</script>

<Story asChild beforeEach={() => seedChatStory()} name="Base">
	<StateGallery columns={1}>
		<StateGalleryItem label={basePalette.name} note={basePalette.note}>
			<PaletteRow palette={basePalette} />
		</StateGalleryItem>
	</StateGallery>
</Story>
