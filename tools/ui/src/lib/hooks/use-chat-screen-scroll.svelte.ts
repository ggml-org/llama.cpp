/**
 * Scroll container binding and navigation guard for the ChatScreen.
 *
 * Binds the `AutoScrollController` to `document.documentElement`, exposes
 * the container for programmatic scrolling, and flags an `isNavigating`
 * window during route changes so the controller can reset without its
 * scroll handler seeing spurious events from layout shifts.
 */

import type { AutoScrollController } from './use-auto-scroll.svelte';
import { afterNavigate, beforeNavigate, disableScrollHandling, onNavigate } from '$app/navigation';
import { LANDING_QUIET_MS } from '$lib/constants';
import { onDestroy } from 'svelte';

const LANDING_INTENT_EVENTS = ['keydown', 'pointerdown', 'touchstart', 'wheel'] as const;

export function useChatScreenScroll(autoScroll: AutoScrollController) {
	let chatScrollContainer: HTMLElement | undefined = $state();
	let isNavigating = $state(false);
	let stopLanding: (() => void) | null = null;

	function handleScroll(event: UIEvent) {
		// Ignore scroll events caused by navigation layout changes. Programmatic
		// and layout scrolls cannot stop auto-scroll: only the reader's own
		// input counts as scrolling up (see AutoScrollController.handleScroll).
		if (isNavigating || !event.isTrusted) return;

		autoScroll.handleScroll();
	}

	/**
	 * Pins the page to its bottom while a freshly opened conversation settles.
	 * ResizeObserver callbacks run after layout and before paint, so every
	 * height change (lazy rows mounting, content-visibility realizing a row's
	 * size, highlight passes) is corrected in the frame that caused it and no
	 * frame is ever painted off the bottom. Stops on the reader's first input
	 * or once the page has not resized for `LANDING_QUIET_MS`.
	 */
	function landAtBottom() {
		stopLanding?.();

		const container = chatScrollContainer;

		if (!container) return;

		let quietTimer: ReturnType<typeof setTimeout> | undefined;

		const pin = () => {
			container.scrollTop = container.scrollHeight;
			clearTimeout(quietTimer);
			quietTimer = setTimeout(stop, LANDING_QUIET_MS);
		};
		const observer = new ResizeObserver(pin);
		const stop = () => {
			observer.disconnect();
			clearTimeout(quietTimer);

			for (const type of LANDING_INTENT_EVENTS) {
				window.removeEventListener(type, stop, { capture: true });
			}

			if (stopLanding === stop) stopLanding = null;
		};

		stopLanding = stop;

		for (const type of LANDING_INTENT_EVENTS) {
			window.addEventListener(type, stop, { capture: true, passive: true });
		}

		pin();
		observer.observe(document.body);
	}

	beforeNavigate(() => {
		isNavigating = true;
		stopLanding?.();
		autoScroll.resetScrollState();
	});

	// Opening a conversation lands at its bottom once its messages load, so
	// SvelteKit's own reset (or restore, on back/forward) would only move the
	// previous conversation, still on screen, to its top for a few frames.
	onNavigate((navigation) => {
		if (navigation.to?.params?.id) disableScrollHandling();
	});

	afterNavigate(() => {
		setTimeout(() => {
			isNavigating = false;
			autoScroll.resetScrollState();
		}, 10);
	});

	$effect(() => {
		chatScrollContainer = document.documentElement;
		autoScroll.setContainer(chatScrollContainer);
	});

	onDestroy(() => stopLanding?.());

	return {
		get chatScrollContainer() {
			return chatScrollContainer;
		},
		handleScroll,
		landAtBottom
	};
}
