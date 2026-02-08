<script lang="ts">
	import { ChevronLeft, ChevronRight } from '@lucide/svelte';

	interface Props {
		class?: string;
		children?: import('svelte').Snippet;
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
		if (scrollContainer) {
			setTimeout(() => {
				updateScrollButtons();
			}, 0);
		}
	});
</script>

<div class="relative {className}">
	<button
		class="absolute top-1/2 left-4 z-10 flex h-6 w-6 -translate-y-1/2 items-center justify-center rounded-full bg-foreground/15 shadow-md backdrop-blur-xs transition-opacity hover:bg-foreground/35 {canScrollLeft
			? 'opacity-100'
			: 'pointer-events-none opacity-0'}"
		onclick={scrollLeft}
		aria-label="Scroll left"
	>
		<ChevronLeft class="h-4 w-4" />
	</button>

	<div
		class="scrollbar-hide flex items-start gap-{gapSize} overflow-x-auto"
		bind:this={scrollContainer}
		onscroll={updateScrollButtons}
	>
		{@render children?.()}
	</div>

	<button
		class="absolute top-1/2 right-4 z-10 flex h-6 w-6 -translate-y-1/2 items-center justify-center rounded-full bg-foreground/15 shadow-md backdrop-blur-xs transition-opacity hover:bg-foreground/35 {canScrollRight
			? 'opacity-100'
			: 'pointer-events-none opacity-0'}"
		onclick={scrollRight}
		aria-label="Scroll right"
	>
		<ChevronRight class="h-4 w-4" />
	</button>
</div>
