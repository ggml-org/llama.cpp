<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { ScrollCarousel } from '#lib/components/app/index.ts';
	import { ScrollCarouselVariant } from '#lib/enums/index.ts';
	import StateGallery from '$llama-ui-stories/helpers/StateGallery.svelte';
	import StateGalleryItem from '$llama-ui-stories/helpers/StateGalleryItem.svelte';

	// the horizontal scroller behind attachment rows and chat tabs
	const { Story } = defineMeta({
		component: ScrollCarousel,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama Chat/Misc/ScrollCarousel'
	});

	const items = Array.from({ length: 12 }, (_, index) => index + 1);
</script>

<Story asChild name="Default">
	<StateGallery columns={1}>
		{#each [ScrollCarouselVariant.TOP, ScrollCarouselVariant.CENTER] as variant (variant)}
			<StateGalleryItem label={variant} note="arrows always shown">
				<ScrollCarousel alwaysShowArrows class="w-[min(32rem,calc(100vw-4rem))]" {variant}>
					{#each items as item (item)}
						<div
							class="flex h-16 w-24 shrink-0 items-center justify-center rounded-xl border border-border bg-muted text-label-sm text-muted-foreground"
						>
							{item}
						</div>
					{/each}
				</ScrollCarousel>
			</StateGalleryItem>
		{/each}
	</StateGallery>
</Story>
