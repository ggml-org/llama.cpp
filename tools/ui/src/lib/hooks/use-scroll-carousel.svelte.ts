export function useScrollCarousel() {
	let canScrollLeft = $state(false);
	let canScrollRight = $state(false);
	let scrollContainer = $state<HTMLDivElement | undefined>();
	let contentContainer = $state<HTMLDivElement | undefined>();

	function scrollToCenter(element: HTMLElement) {
		if (!scrollContainer) return;

		const containerRect = scrollContainer.getBoundingClientRect();
		const elementRect = element.getBoundingClientRect();
		const elementCenter = elementRect.left + elementRect.width / 2;
		const containerCenter = containerRect.left + containerRect.width / 2;
		const scrollOffset = elementCenter - containerCenter;

		scrollContainer.scrollBy({ behavior: 'smooth', left: scrollOffset });
	}

	function updateScrollButtons() {
		if (!scrollContainer) return;

		const { clientWidth, scrollLeft: sl, scrollWidth } = scrollContainer;

		canScrollLeft = sl > 0;
		canScrollRight = sl < scrollWidth - clientWidth - 1;
	}

	// Re-evaluate arrow visibility whenever the container or its content resizes,
	// otherwise the arrows may not appear when overflowing items are added (e.g. new
	// tabs/attachments) and the user has not scrolled yet.
	$effect(() => {
		if (!scrollContainer) return;

		updateScrollButtons();

		const observer = new ResizeObserver(() => updateScrollButtons());

		observer.observe(scrollContainer);

		if (contentContainer) observer.observe(contentContainer);

		return () => observer.disconnect();
	});

	return {
		get canScrollLeft() {
			return canScrollLeft;
		},
		get canScrollRight() {
			return canScrollRight;
		},
		get contentContainer() {
			return contentContainer;
		},
		set contentContainer(el: HTMLDivElement | undefined) {
			contentContainer = el;
		},
		get scrollContainer() {
			return scrollContainer;
		},
		set scrollContainer(el: HTMLDivElement | undefined) {
			scrollContainer = el;
		},
		scrollToCenter,
		updateScrollButtons
	};
}
