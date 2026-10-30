<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { Logo } from '$lib/components/app';

	// the llama.cpp mark; sized by --size (1rem by default) and filled with the text colour
	const { Story } = defineMeta({
		component: Logo,
		parameters: {
			layout: 'centered'
		},
		title: 'Brand/Logo'
	});

	const sizes = [
		{ label: '14px', note: 'model rows', size: '0.875rem' },
		{ label: '16px', note: 'sidebar', size: '1rem' },
		{ label: '32px', size: '2rem' },
		{ label: '64px', size: '4rem' }
	];
</script>

<Story asChild name="Sizes">
	<StateGallery columns={4} minItemWidth="8rem">
		{#each sizes as size (size.label)}
			<StateGalleryItem label={size.label} note={size.note}>
				<Logo style="--size: {size.size}" />
			</StateGalleryItem>
		{/each}
	</StateGallery>
</Story>

<Story asChild name="Colour">
	<StateGallery columns={3} minItemWidth="8rem">
		<StateGalleryItem label="Default" note="fg-default">
			<Logo class="text-fg-default" style="--size: 2rem" />
		</StateGalleryItem>

		<StateGalleryItem label="Muted" note="fg-muted">
			<Logo class="text-fg-muted" style="--size: 2rem" />
		</StateGalleryItem>

		<StateGalleryItem label="On primary" note="fg-on-action">
			<div class="flex size-14 items-center justify-center rounded-card bg-(--action-primary)">
				<Logo class="text-fg-on-action" style="--size: 2rem" />
			</div>
		</StateGalleryItem>
	</StateGallery>
</Story>
