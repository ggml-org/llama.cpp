<script lang="ts" module>
	import { seedChatStory } from '../../chat/messages/fixtures';
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import Boards from './palette/Boards.svelte';
	import PaletteRow from './palette/PaletteRow.svelte';
	import { basePalette, poudrePalette } from './palette/palettes';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Proposal: the communications palette, from the neutral base to the poudre palette. (The UI's status colours were
	// chosen and are in the tokens; see DESIGN.md.) Delete once these colours are in the tokens too (see DESIGN.md,
	// "Proposing changes").
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Design System/Proposals/Palette'
	});
</script>

<Story asChild beforeEach={() => seedChatStory()} name="Communications base">
	<StateGallery columns={1}>
		<StateGalleryItem label={basePalette.name} note={basePalette.note}>
			<PaletteRow palette={basePalette} />
		</StateGalleryItem>
	</StateGallery>
</Story>

<Story asChild beforeEach={() => seedChatStory()} name="Specimen boards">
	<StateGallery columns={1}>
		<StateGalleryItem label="Key visuals" note={poudrePalette.note}>
			<PaletteRow palette={poudrePalette} />
		</StateGalleryItem>

		<StateGalleryItem
			label="Specimen boards"
			note="After the classic Swiss specimen boards, in the same palette"
		>
			<Boards />
		</StateGalleryItem>
	</StateGallery>
</Story>
