<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { ScrollCarousel } from '$lib/components/app';
	import { ScrollCarouselVariant } from '$lib/enums';

	// the horizontal scroller behind attachment rows and chat tabs
	const { Story } = defineMeta({
		component: ScrollCarousel,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Patterns/ScrollCarousel'
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
							class="flex h-16 w-24 shrink-0 items-center justify-center rounded-card border border-line-default bg-surface-muted text-label-sm text-fg-muted"
						>
							{item}
						</div>
					{/each}
				</ScrollCarousel>
			</StateGalleryItem>
		{/each}
	</StateGallery>
</Story>
