<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { Logo } from '$lib/components/app';

	// the llama.cpp mark; it takes the current text colour
	const { Story } = defineMeta({
		component: Logo,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Patterns/Logo'
	});
</script>

<Story asChild name="Default">
	<StateGallery columns={3} minItemWidth="8rem">
		<StateGalleryItem label="Default">
			<Logo class="size-8" />
		</StateGalleryItem>

		<StateGalleryItem label="Muted">
			<Logo class="size-8 text-fg-muted" />
		</StateGalleryItem>

		<StateGalleryItem label="Large">
			<Logo class="size-16" />
		</StateGalleryItem>
	</StateGallery>
</Story>
