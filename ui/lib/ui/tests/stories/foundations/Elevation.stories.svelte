<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// the elevation tokens from ui/lib/ui/src/lib/styles/tokens/semantic.css, each on a plain sample
	const { Story } = defineMeta({
		parameters: {
			layout: 'centered'
		},
		title: 'Llama UI/Foundations/Elevation'
	});

	const levels = [
		{ name: 'elevation-inset', note: 'fields, recessed' },
		{ name: 'elevation-control', note: 'buttons and tabs' },
		{ name: 'elevation-surface', note: 'chat form, cards, sidebar, tooltips' },
		{ name: 'elevation-raised', note: 'menus, popovers, dialogs' }
	];
</script>

<Story asChild name="Default">
	<StateGallery columns={5} minItemWidth="10rem">
		{#each levels as level (level.name)}
			<StateGalleryItem label={level.name} note={level.note}>
				<div class="p-4">
					<div
						style:box-shadow="var(--{level.name})"
						class="h-20 rounded-xl bg-card"
					></div>
				</div>
			</StateGalleryItem>
		{/each}
	</StateGallery>
</Story>
