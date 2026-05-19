import { browser } from '$app/environment';
import { DEFAULT_MOBILE_BREAKPOINT } from '$lib/constants/viewport';

export const viewport = $state({
	// Initialize with actual window width to avoid desktop-as-mobile flash
	width: browser ? window.innerWidth : 0
});

export function isMobile() {
	return viewport.width < DEFAULT_MOBILE_BREAKPOINT;
}
