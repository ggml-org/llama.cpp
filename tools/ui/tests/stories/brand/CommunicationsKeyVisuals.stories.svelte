<script lang="ts" module>
	import { seedChatStory } from '../chat/messages/fixtures';
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import PaletteRow from './communications/PaletteRow.svelte';
	import { basePalette } from './communications/palettes';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Communications' neutral key visuals, with the real UI on them
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Communications/Key visuals'
	});
</script>

<Story asChild beforeEach={() => seedChatStory()} name="Key visuals">
	<StateGallery columns={1}>
		<StateGalleryItem label={basePalette.name} note={basePalette.note}>
			<PaletteRow palette={basePalette} />
		</StateGalleryItem>
	</StateGallery>
</Story>
