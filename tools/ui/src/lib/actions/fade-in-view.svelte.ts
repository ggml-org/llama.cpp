import { isElementInViewport } from '$lib/utils/viewport';

/**
 * Svelte action that fades in an element when it enters the viewport.
 * Uses IntersectionObserver for efficient viewport detection.
 *
 * If skipIfVisible is set and the element is already visible in the viewport
 * when the action attaches (e.g. a markdown block promoted from unstable
 * during streaming), the fade is skipped entirely to avoid a flash.
 */
export function fadeInView(
	node: HTMLElement,
	options: { duration?: number; y?: number; skipIfVisible?: boolean } = {}
) {
	const { duration = 300, y = 0, skipIfVisible = false } = options;

	if (skipIfVisible && isElementInViewport(node)) {
		return;
	}

	node.style.opacity = '0';
	node.style.transform = `translateY(${y}px)`;
	node.style.transition = `opacity ${duration}ms ease-out, transform ${duration}ms ease-out`;

	$effect(() => {
		const observer = new IntersectionObserver(
			(entries) => {
				for (const entry of entries) {
					if (entry.isIntersecting) {
						requestAnimationFrame(() => {
							node.style.opacity = '1';
							node.style.transform = 'translateY(0)';
						});
						observer.disconnect();
					}
				}
			},
			{ threshold: 0.05 }
		);

		observer.observe(node);

		return () => {
			observer.disconnect();
		};
	});
}
