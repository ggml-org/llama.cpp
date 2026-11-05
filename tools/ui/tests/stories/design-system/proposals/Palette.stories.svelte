<script lang="ts" module>
	import PaletteRow from '../../brand/communications/PaletteRow.svelte';
	import { llamaPalette } from '../../brand/communications/palettes';
	import { seedChatStory } from '../../chat/messages/fixtures';
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import Boards from './palette/Boards.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Proposal: the Llama palette for communications (the design system's own colours with beige), on top of the
	// neutral base in Communications/Base. Delete once it is chosen (see DESIGN.md, "Proposing changes").
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Brand/Colors/Palette proposal'
	});
</script>

<Story asChild beforeEach={() => seedChatStory()} name="Specimen boards">
	<StateGallery columns={1}>
		<StateGalleryItem label="Key visuals" note={llamaPalette.note}>
			<PaletteRow palette={llamaPalette} />
		</StateGalleryItem>

		<StateGalleryItem
			label="Specimen boards"
			note="After the classic Swiss specimen boards, in the same palette"
		>
			<Boards />
		</StateGalleryItem>
	</StateGallery>
</Story>
