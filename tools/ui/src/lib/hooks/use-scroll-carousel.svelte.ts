export function useScrollCarousel() {
	let canScrollLeft = $state(false);
	let canScrollRight = $state(false);
	let scrollContainer = $state<HTMLDivElement | undefined>();

	function scrollToCenter(element: HTMLElement) {
		if (!scrollContainer) return;

		const containerRect = scrollContainer.getBoundingClientRect();
		const elementRect = element.getBoundingClientRect();

		const elementCenter = elementRect.left + elementRect.width / 2;
		const containerCenter = containerRect.left + containerRect.width / 2;
		const scrollOffset = elementCenter - containerCenter;

		scrollContainer.scrollBy({ left: scrollOffset, behavior: 'smooth' });
	}

	function scrollLeft() {
		if (!scrollContainer) return;
		scrollContainer.scrollBy({ left: -250, behavior: 'smooth' });
	}

	function scrollRight() {
		if (!scrollContainer) return;
		scrollContainer.scrollBy({ left: 250, behavior: 'smooth' });
	}

	function updateScrollButtons() {
		if (!scrollContainer) return;

		const { scrollLeft: sl, scrollWidth, clientWidth } = scrollContainer;
		canScrollLeft = sl > 0;
		canScrollRight = sl < scrollWidth - clientWidth - 1;
	}

	$effect(() => {
		if (scrollContainer) {
			updateScrollButtons();
		}
	});

	return {
		get canScrollLeft() {
			return canScrollLeft;
		},
		get canScrollRight() {
			return canScrollRight;
		},
		get scrollContainer() {
			return scrollContainer;
		},
		set scrollContainer(el: HTMLDivElement | undefined) {
			scrollContainer = el;
		},
		scrollToCenter,
		scrollLeft,
		scrollRight,
		updateScrollButtons
	};
}
