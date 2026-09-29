import type { ActionReturn } from 'svelte/action';

/** Margin around the viewport that already counts as "near". */
const NEAR_VIEWPORT_MARGIN = '200px';

/**
 * Calls back once the element comes near the viewport, so per-row work that costs
 * a request (avatars, Hub details) does not fire on mount.
 */
export function nearViewport(node: HTMLElement, onNear: () => void): ActionReturn {
	if (typeof IntersectionObserver === 'undefined') {
		onNear();

		return {};
	}

	const observer = new IntersectionObserver(
		(entries) => {
			if (entries.some((entry) => entry.isIntersecting)) {
				observer.disconnect();
				onNear();
			}
		},
		{ rootMargin: NEAR_VIEWPORT_MARGIN }
	);

	observer.observe(node);

	return {
		destroy() {
			observer.disconnect();
		}
	};
}
