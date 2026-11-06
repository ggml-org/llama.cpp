<script lang="ts" module>
	import { seedChatStory } from '../chat/messages/fixtures';
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import Boards from './communications/Boards.svelte';
	import PaletteRow from './communications/PaletteRow.svelte';
	import { basePalette } from './communications/palettes';
	import Principles from './communications/Principles.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Communications' direction: what a piece should feel like, before any rule. The principles, the mood board (posters
	// with the real UI on them) and the specimen boards, as reference for everything built on Foundations
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Communications/Direction'
	});
</script>

<Story asChild name="Principles">
	<div class="p-6"><Principles /></div>
</Story>

<Story asChild beforeEach={() => seedChatStory()} name="Mood board">
	<StateGallery columns={1}>
		<StateGalleryItem
			label="Mood board"
			note="Posters in the neutral base, with the real UI on them"
		>
			<PaletteRow palette={basePalette} />
		</StateGalleryItem>
	</StateGallery>
</Story>

<Story asChild name="Specimen boards">
	<StateGallery columns={1}>
		<StateGalleryItem
			label="Specimen boards and posters"
			note="After the classic Swiss specimen boards, in the brand palette: the art colours, beige and cinder"
		>
			<Boards />
		</StateGalleryItem>
	</StateGallery>
</Story>
