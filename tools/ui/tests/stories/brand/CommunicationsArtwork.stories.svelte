<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import Boards from './communications/Boards.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Communications' colour artwork: specimen boards and posters in the brand palette
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Communications/Artwork'
	});
</script>

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
