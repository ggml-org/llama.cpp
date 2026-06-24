/**
 * Scroll container binding and navigation guard for the ChatScreen.
 *
 * Binds the `AutoScrollController` to `document.documentElement`, exposes
 * the container for programmatic scrolling, and flags an `isNavigating`
 * window during route changes so the controller can reset without its
 * scroll handler seeing spurious events from layout shifts.
 */

import { afterNavigate, beforeNavigate } from '$app/navigation';
import type { AutoScrollController } from './use-auto-scroll.svelte';

export function useChatScreenScroll(autoScroll: AutoScrollController) {
	let chatScrollContainer: HTMLElement | undefined = $state();
	let isNavigating = $state(false);

	function handleScroll(event: UIEvent) {
		// Ignore scroll events caused by navigation layout changes or by our own
		// programmatic scrolls so they don't accidentally disable auto-scroll.
		if (isNavigating || !event.isTrusted) return;
		autoScroll.handleScroll();
	}

	beforeNavigate(() => {
		isNavigating = true;
		autoScroll.resetScrollState();
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

	return {
		get chatScrollContainer() {
			return chatScrollContainer;
		},
		handleScroll
	};
}
