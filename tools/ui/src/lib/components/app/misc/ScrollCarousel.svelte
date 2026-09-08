<script lang="ts">
	import { ChevronLeft, ChevronRight } from '@lucide/svelte';
	import { cn } from '$lib/components/ui/utils';
	import { ICON_CLASS_DEFAULT } from '$lib/constants';
	import { ScrollCarouselVariant } from '$lib/enums';
	import { useScrollCarousel } from '$lib/hooks/use-scroll-carousel.svelte';
	import type { Snippet } from 'svelte';

	interface Props {
		children: Snippet;
		/** External carousel hook for callers that need to drive it (e.g. scrollToCenter). */
		carousel?: ReturnType<typeof useScrollCarousel>;
		/** Classes for the outer relative wrapper. */
		class?: string;
		/** Classes for the scrollable overflow container. */
		containerClass?: string;
		/** Classes for the min-w-max content wrapper. */
		innerClass?: string;
		/** Tailwind gap class applied to the content wrapper. */
		gapSize?: string;
		/** Show the arrows whenever the content overflows, even without hover. */
		alwaysShowArrows?: boolean;
		/** Arrow placement and styling. */
		variant?: ScrollCarouselVariant;
	}

	let {
		alwaysShowArrows = false,
		carousel: externalCarousel,
		children,
		class: className = '',
		containerClass = '',
		gapSize = '3',
		innerClass = '',
		variant = ScrollCarouselVariant.TOP
	}: Props = $props();

	const internalCarousel = useScrollCarousel();
	const carousel = $derived(externalCarousel ?? internalCarousel);

	const isCenter = $derived(variant === ScrollCarouselVariant.CENTER);

	function scrollLeft(event?: MouseEvent) {
		event?.stopPropagation();
		event?.preventDefault();

		const container = carousel.scrollContainer;

		if (!container) return;

		container.scrollBy({ behavior: 'smooth', left: -(container.clientWidth * 0.67) });
	}

	function scrollRight(event?: MouseEvent) {
		event?.stopPropagation();
		event?.preventDefault();

		const container = carousel.scrollContainer;

		if (!container) return;

		container.scrollBy({ behavior: 'smooth', left: container.clientWidth * 0.67 });
	}

	export function resetScroll() {
		const container = carousel.scrollContainer;

		if (!container) return;

		container.scrollLeft = 0;
		setTimeout(() => carousel.updateScrollButtons(), 0);
	}
</script>

<div
	class={cn('group relative', !isCenter && 'flex items-center', className)}
	style={!isCenter ? 'scroll-padding: 1rem;' : undefined}
>
	<button
		class={cn(
			'absolute z-10 flex h-6 w-6 items-center justify-center rounded-full shadow-md transition-opacity',
			isCenter
				? 'top-1/2 left-4 -translate-y-1/2 bg-background/25 backdrop-blur-xs hover:bg-background/45 disabled:pointer-events-none disabled:opacity-0'
				: 'left-2 bg-muted backdrop-blur-sm hover:bg-accent',
			!isCenter &&
				(carousel.canScrollLeft
					? alwaysShowArrows
						? 'opacity-100'
						: 'opacity-0 group-hover:opacity-100'
					: 'pointer-events-none opacity-0')
		)}
		{...isCenter ? { disabled: !carousel.canScrollLeft } : {}}
		aria-label="Scroll left"
		onclick={scrollLeft}
	>
		<ChevronLeft class={ICON_CLASS_DEFAULT} />
	</button>

	<div
		bind:this={carousel.scrollContainer}
		class={cn('scrollbar-hide overflow-x-auto', containerClass)}
		onscroll={carousel.updateScrollButtons}
	>
		<div
			bind:this={carousel.contentContainer}
			class={cn('flex min-w-max', isCenter && 'items-start', `gap-${gapSize}`, innerClass)}
		>
			{@render children?.()}
		</div>
	</div>

	<button
		class={cn(
			'absolute z-10 flex h-6 w-6 items-center justify-center rounded-full shadow-md transition-opacity',
			isCenter
				? 'top-1/2 right-4 -translate-y-1/2 bg-background/25 backdrop-blur-xs hover:bg-background/45 disabled:pointer-events-none disabled:opacity-0'
				: 'right-2 bg-muted backdrop-blur-sm hover:bg-accent',
			!isCenter &&
				(carousel.canScrollRight
					? alwaysShowArrows
						? 'opacity-100'
						: 'opacity-0 group-hover:opacity-100'
					: 'pointer-events-none opacity-0')
		)}
		{...isCenter ? { disabled: !carousel.canScrollRight } : {}}
		aria-label="Scroll right"
		onclick={scrollRight}
	>
		<ChevronRight class={ICON_CLASS_DEFAULT} />
	</button>
</div>
