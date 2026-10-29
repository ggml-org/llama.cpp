<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// the corner roles from src/lib/styles/tokens/semantic.css
	const { Story } = defineMeta({
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Foundations/Shapes'
	});

	const radii = [
		{ name: 'rounded-control', note: '8px, buttons, fields, badges' },
		{ name: 'rounded-panel', note: '10px, menus and dialogs' },
		{ name: 'rounded-card', note: '14px, cards' },
		{ name: 'rounded-container', note: '16px, the sidebar' },
		{ name: 'rounded-surface', note: '24px, chat form, bubbles' },
		{ name: 'rounded-surface-lg', note: '32px, chat form on mobile' },
		{ name: 'rounded-full', note: 'lamps, avatars, round icon buttons' }
	];
</script>

<Story asChild name="Default">
	<StateGallery columns={4}>
		{#each radii as radius (radius.name)}
			<StateGalleryItem label={radius.name} note={radius.note}>
				<div class="h-24 border border-line-default bg-surface-muted {radius.name}"></div>
			</StateGalleryItem>
		{/each}
	</StateGallery>
</Story>
