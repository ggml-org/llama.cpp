<script lang="ts">
	import { browser } from '$app/environment';
	import { mode } from 'mode-watcher';

	import githubDarkCss from 'highlight.js/styles/github-dark.css?inline';
	import githubLightCss from 'highlight.js/styles/github.css?inline';
	import { ColorMode } from '$lib/enums';
	import { highlightCode } from '$lib/utils';

	interface Props {
		code: string;
		language?: string;
		class?: string;
		maxHeight?: string;
		maxWidth?: string;
		// When true the container auto-scrolls to the bottom as new chunks
		// arrive; scrolling up pauses the follow until the user returns to
		// the bottom. Same pattern as ChatMessageReasoningBlock.
		streaming?: boolean;
	}

	let {
		code,
		language = 'text',
		class: className = '',
		maxHeight = '60vh',
		maxWidth = '',
		streaming = false
	}: Props = $props();

	const highlightedHtml = $derived(highlightCode(code, language));

	let scrollEl = $state<HTMLDivElement>();
	let userScrolledUp = $state(false);
	let lastScrollTop = 0;
	// Any scroll position within this many pixels of the bottom counts as
	// "at the bottom" and continues to follow new content. Bigger than the
	// chat main view's 10px threshold because line wrap reflows while the
	// highlight.js pass settles can drift a few pixels off bottom.
	const SCROLL_BOTTOM_THRESHOLD_PX = 32;
	let pendingFrame: number | null = null;

	function loadHighlightTheme(isDark: boolean) {
		if (!browser) return;

		const existingThemes = document.querySelectorAll('style[data-highlight-theme-preview]');
		existingThemes.forEach((style) => style.remove());

		const style = document.createElement('style');
		style.setAttribute('data-highlight-theme-preview', 'true');
		style.textContent = isDark ? githubDarkCss : githubLightCss;

		document.head.appendChild(style);
	}

	function isAtBottom(): boolean {
		if (!scrollEl) return false;
		return (
			scrollEl.scrollHeight - scrollEl.clientHeight - scrollEl.scrollTop <=
			SCROLL_BOTTOM_THRESHOLD_PX
		);
	}

	function scrollToBottomOnFrame() {
		if (pendingFrame !== null || !scrollEl || userScrolledUp) return;
		pendingFrame = requestAnimationFrame(() => {
			pendingFrame = null;
			// Re-check `userScrolledUp` at paint time. Skip an `isAtBottom`
			// gate: it would falsely return false on the first chunk that
			// overflows the container (scrollTop is still at the top),
			// freezing autoscroll for the rest of the stream.
			if (scrollEl && !userScrolledUp) {
				scrollEl.scrollTop = scrollEl.scrollHeight;
			}
		});
	}

	function handleScrollEvent() {
		if (!scrollEl) return;
		const isScrollingUp = scrollEl.scrollTop < lastScrollTop;
		if (isScrollingUp && !isAtBottom()) {
			userScrolledUp = true;
		} else if (isAtBottom()) {
			userScrolledUp = false;
		}
		lastScrollTop = scrollEl.scrollTop;
	}

	$effect(() => {
		const currentMode = mode.current;
		const isDark = currentMode === ColorMode.DARK;

		loadHighlightTheme(isDark);
	});

	// Reset sticky state at the start of a streaming episode so the first
	// chunk pins to the bottom again. Only depends on `streaming`, so
	// post-stream code updates don't trigger this and preserve the user's
	// scroll position.
	$effect(() => {
		if (streaming) {
			userScrolledUp = false;
			lastScrollTop = 0;
		}
	});

	// Follow growing content while streaming. Tracks `code` so each chunk
	// schedules a scroll, but skips if the user has scrolled up.
	$effect(() => {
		void code;
		if (!streaming || userScrolledUp) return;
		scrollToBottomOnFrame();
	});

	// Catch DOM mutations that don't change `code` directly (e.g. layout
	// shifts after highlight.js re-tokenizes, line-wrap reflows).
	$effect(() => {
		if (!streaming || !scrollEl) return;

		const observer = new MutationObserver(() => scrollToBottomOnFrame());
		observer.observe(scrollEl, {
			childList: true,
			subtree: true,
			characterData: true
		});

		return () => observer.disconnect();
	});
</script>

<div
	bind:this={scrollEl}
	onscroll={handleScrollEvent}
	class="code-preview-wrapper min-w-0 max-w-full overflow-auto rounded-xl border shadow-[0_1px_2px_0_rgb(0_0_0_/_0.05)] {className}"
	style="border-color: color-mix(in oklch, var(--border) 30%, transparent); background: var(--code-background); max-height: {maxHeight}; {maxWidth
		? `max-width: ${maxWidth};`
		: ''}"
>
	<!-- Needs to be formatted as single line for proper rendering -->
	<pre class="m-0"><code class="hljs text-sm leading-relaxed">{@html highlightedHtml}</code></pre>
</div>

<style>
	.code-preview-wrapper {
		overscroll-behavior: contain;
	}

	.code-preview-wrapper pre {
		background: transparent;
		padding: 0;
	}

	.code-preview-wrapper code {
		background: transparent;
		display: block;
		padding: 0.5rem;
	}

	:global(.dark) .code-preview-wrapper {
		border-color: color-mix(in oklch, var(--border) 20%, transparent);
	}
</style>
