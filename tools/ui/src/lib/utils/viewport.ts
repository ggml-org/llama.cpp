/**
 * Check if an element is within the current viewport.
 */
export function isElementInViewport(node: HTMLElement): boolean {
	const rect = node.getBoundingClientRect();
	return (
		rect.top < window.innerHeight &&
		rect.bottom > 0 &&
		rect.left < window.innerWidth &&
		rect.right > 0
	);
}
