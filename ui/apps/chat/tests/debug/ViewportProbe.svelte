<script lang="ts">
	/**
	 * Debug readout of the viewport's heights, for checking layout on a real phone, where each
	 * browser and the installed app size the viewport units differently. Lists the elements
	 * that reach past the bottom of the window, which is what makes a page scroll.
	 * Mounted only in a dev build with VITE_DEBUG set.
	 */
	import { onMount } from 'svelte';

	const REFRESH_MS = 1000;
	const OVERFLOW_ROWS = 4;
	const CLASS_PREVIEW_LENGTH = 40;

	let lines = $state<string[]>([]);

	/** Resolves a CSS length to px by laying out a hidden box of that height. */
	function toPx(length: string): number {
		const el = document.createElement('div');

		el.style.cssText = `position: absolute; visibility: hidden; height: ${length}`;
		document.body.append(el);

		const height = el.getBoundingClientRect().height;

		el.remove();

		return Math.round(height);
	}

	/** The deepest elements in the flow that end below the window; fixed ones never scroll it. */
	function overflowingElements(): string[] {
		const limit = window.innerHeight;

		return [...document.querySelectorAll<HTMLElement>('body *')]
			.filter(
				(el) =>
					getComputedStyle(el).position !== 'fixed' &&
					el.getBoundingClientRect().bottom > limit + 0.5
			)
			.slice(-OVERFLOW_ROWS)
			.map((el) => {
				const { bottom, top } = el.getBoundingClientRect();
				const classes = (el.getAttribute('class') ?? '').slice(0, CLASS_PREVIEW_LENGTH);

				return `${el.tagName.toLowerCase()} ${Math.round(top)}-${Math.round(bottom)} ${classes}`;
			});
	}

	function measure() {
		const root = document.documentElement;

		lines = [
			`inner ${window.innerHeight} vv ${Math.round(window.visualViewport?.height ?? 0)} screen ${screen.height}`,
			`dvh ${toPx('100dvh')} svh ${toPx('100svh')} lvh ${toPx('100lvh')}`,
			`safeTop ${toPx('env(safe-area-inset-top, 0px)')} safeBottom ${toPx('env(safe-area-inset-bottom, 0px)')}`,
			`html ${root.clientHeight} scroll ${root.scrollHeight}`,
			...overflowingElements()
		];
	}

	onMount(() => {
		measure();

		const id = setInterval(measure, REFRESH_MS);

		return () => clearInterval(id);
	});
</script>

<pre
	aria-hidden="true"
	class="pointer-events-none fixed top-1/3 left-2 z-[99999] rounded bg-black/80 p-2 text-[10px] leading-tight whitespace-pre-wrap text-lime-300">{lines.join(
		'\n'
	)}</pre>
