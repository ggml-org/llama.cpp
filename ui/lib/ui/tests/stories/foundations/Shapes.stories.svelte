<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// the corner roles from ui/lib/ui/src/lib/styles/tokens/semantic.css
	const { Story } = defineMeta({
		parameters: {
			layout: 'centered'
		},
		title: 'Llama UI/Foundations/Shapes'
	});

	const radii = [
		{ name: 'rounded-sm', note: '6px, small marks and chips' },
		{ name: 'rounded-md', note: '8px, buttons, fields, badges' },
		{ name: 'rounded-lg', note: '10px, menus and dialogs' },
		{ name: 'rounded-xl', note: '14px, cards' },
		{ name: 'rounded-2xl', note: '16px, the sidebar, bubbles' },
		{ name: 'rounded-3xl', note: '24px, chat form' },
		{ name: 'rounded-4xl', note: '32px, chat form on mobile' },
		{ name: 'rounded-full', note: 'lamps, avatars, round icon buttons' }
	];
</script>

<Story asChild name="Default">
	<StateGallery columns={4}>
		{#each radii as radius (radius.name)}
			<StateGalleryItem label={radius.name} note={radius.note}>
				<div class="h-24 border border-border bg-muted {radius.name}"></div>
			</StateGalleryItem>
		{/each}
	</StateGallery>
</Story>
