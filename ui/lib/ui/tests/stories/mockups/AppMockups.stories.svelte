<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { MacMenu, WindowsTray } from '$lib/components/mockups';

	// the native apps' UIs as the brand draws them, for pages and communications: Llama for Mac's menu and Llama for
	// Windows' tray flyout, each with its parts marked the way llama.app's hero numbers them
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'centered'
		},
		title: 'UI/Blocks/Mockups/App mockups'
	});
</script>

{#snippet number(part: string)}
	<span
		class="flex size-6 items-center justify-center rounded-full bg-foreground text-[13px] font-semibold text-background ring-2 ring-background"
		title={part}>{part[0].toUpperCase()}</span
	>
{/snippet}

<Story asChild name="Mac menu">
	<StateGallery columns={3} minItemWidth="20rem">
		<StateGalleryItem label="Default">
			<MacMenu />
		</StateGalleryItem>

		<StateGalleryItem label="First launch, a download in progress">
			<MacMenu downloading={0.42} installed={[]} />
		</StateGalleryItem>

		<StateGalleryItem label="With a page's marks">
			<MacMenu annotate={number} />
		</StateGalleryItem>
	</StateGallery>
</Story>

<Story asChild name="Windows tray">
	<StateGallery columns={2} minItemWidth="28rem">
		<StateGalleryItem label="Default">
			<WindowsTray />
		</StateGalleryItem>

		<StateGalleryItem label="With a page's marks">
			<WindowsTray annotate={number} />
		</StateGalleryItem>

		<StateGalleryItem label="A model's page">
			<WindowsTray view="details" />
		</StateGalleryItem>

		<StateGalleryItem label="The Models card alone">
			<WindowsTray card />
		</StateGalleryItem>
	</StateGallery>
</Story>
