<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import Lockups from './logo-showcase/lib/components/Lockups.svelte';
	import LogoGeometry from './logo-showcase/LogoGeometry.svx';
	import ShowcaseFrame from './logo-showcase/ShowcaseFrame.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { Logo } from '$lib/components/app';

	// the llama.cpp mark; sized by --size (1rem by default) and filled with the text colour
	const { Story } = defineMeta({
		component: Logo,
		parameters: {
			layout: 'centered'
		},
		title: 'Logo/Icon'
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

<!-- the rules that shape the icon: two blades, 60° edges, one diagonal, the split, steps of 120, soft corners -->
<Story asChild name="Geometry" parameters={{ controls: { disable: true }, layout: 'fullscreen' }}>
	<ShowcaseFrame>
		<LogoGeometry />
	</ShowcaseFrame>
</Story>

<!-- the icon as a glyph of Llama Sans (U+E000), sitting on the baseline at cap height -->
<Story asChild name="Usage" parameters={{ controls: { disable: true }, layout: 'fullscreen' }}>
	<ShowcaseFrame>
		<Lockups />
	</ShowcaseFrame>
</Story>
