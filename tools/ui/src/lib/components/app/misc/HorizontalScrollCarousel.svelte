<script lang="ts">
	import { ICON_CLASS_DEFAULT } from '$lib/constants/css-classes';
	import { ChevronLeft, ChevronRight } from '@lucide/svelte';
	import type { Snippet } from 'svelte';

	interface Props {
		class?: string;
		children?: Snippet;
		gapSize?: string;
		onScrollableChange?: (isScrollable: boolean) => void;
	}

	let { class: className = '', children, gapSize = '3', onScrollableChange }: Props = $props();

	let canScrollLeft = $state(false);
	let canScrollRight = $state(false);
	let scrollContainer: HTMLDivElement | undefined = $state();

	function scrollLeft(event?: MouseEvent) {
		event?.stopPropagation();
		event?.preventDefault();

		if (!scrollContainer) return;

		scrollContainer.scrollBy({ left: scrollContainer.clientWidth * -0.67, behavior: 'smooth' });
	}

	function scrollRight(event?: MouseEvent) {
		event?.stopPropagation();
		event?.preventDefault();

		if (!scrollContainer) return;

		scrollContainer.scrollBy({ left: scrollContainer.clientWidth * 0.67, behavior: 'smooth' });
	}

	function updateScrollButtons() {
		if (!scrollContainer) return;

		const { scrollLeft, scrollWidth, clientWidth } = scrollContainer;

		canScrollLeft = scrollLeft > 0;
		canScrollRight = scrollLeft < scrollWidth - clientWidth - 1;

		const isScrollable = scrollWidth > clientWidth;
		onScrollableChange?.(isScrollable);
	}

	export function resetScroll() {
		if (scrollContainer) {
			scrollContainer.scrollLeft = 0;
			setTimeout(() => {
				updateScrollButtons();
			}, 0);
		}
	}

	$effect(() => {
		if (!scrollContainer) return;

		const observer = new ResizeObserver(() => updateScrollButtons());
		observer.observe(scrollContainer);

		return () => observer.disconnect();
	});
</script>

<div class="relative {className}">
	<button
		class="absolute top-1/2 left-4 z-10 flex h-6 w-6 -translate-y-1/2 items-center justify-center rounded-full bg-background/25 shadow-md backdrop-blur-xs transition-opacity hover:bg-background/45 disabled:pointer-events-none disabled:opacity-0"
		onclick={scrollLeft}
		disabled={!canScrollLeft}
		aria-label="Scroll left"
	>
		<ChevronLeft class={ICON_CLASS_DEFAULT} />
	</button>

	<div
		class="scrollbar-hide flex items-start gap-{gapSize} overflow-x-auto"
		bind:this={scrollContainer}
		onscroll={updateScrollButtons}
	>
		{@render children?.()}
	</div>

	<button
		class="absolute top-1/2 right-4 z-10 flex h-6 w-6 -translate-y-1/2 items-center justify-center rounded-full bg-background/25 shadow-md backdrop-blur-xs transition-opacity hover:bg-background/45 disabled:pointer-events-none disabled:opacity-0"
		onclick={scrollRight}
		disabled={!canScrollRight}
		aria-label="Scroll right"
	>
		<ChevronRight class={ICON_CLASS_DEFAULT} />
	</button>
</div>
