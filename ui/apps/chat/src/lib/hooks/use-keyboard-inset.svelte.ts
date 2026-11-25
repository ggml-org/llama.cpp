/**
 * Publishes how much of the screen's bottom the on-screen keyboard covers, as
 * `--keyboard-inset` on the root, while `active()` holds; unset while no keyboard is up.
 *
 * iOS lays the keyboard over the page instead of resizing it, so what is pinned to the
 * screen's bottom stays under it: the visual viewport shrinks, the layout viewport does not.
 * Android, with `interactive-widget=resizes-content`, shrinks the page itself, so the two
 * viewports match and this stays unset there. A pinch zoom shrinks the visual viewport too,
 * so none is read while zoomed.
 */
export function useKeyboardInset(active: () => boolean) {
	$effect(() => {
		const viewport = window.visualViewport;

		if (!active() || !viewport) return;

		const root = document.documentElement;
		const update = () => {
			const inset =
				viewport.scale > 1 ? 0 : window.innerHeight - viewport.height - viewport.offsetTop;

			if (inset > 0) {
				root.style.setProperty('--keyboard-inset', `${Math.round(inset)}px`);
			} else {
				root.style.removeProperty('--keyboard-inset');
			}
		};

		update();
		viewport.addEventListener('resize', update);
		viewport.addEventListener('scroll', update);

		return () => {
			viewport.removeEventListener('resize', update);
			viewport.removeEventListener('scroll', update);
			root.style.removeProperty('--keyboard-inset');
		};
	});
}
