import { browser } from '$app/environment';
import { DEFAULT_MOBILE_BREAKPOINT } from '$lib/constants/viewport';
import { MediaQuery } from 'svelte/reactivity';

export const viewport = $state({
	width: browser ? window.innerWidth : 0
});

export const isMobile = new MediaQuery(`max-width: ${DEFAULT_MOBILE_BREAKPOINT - 1}px`);
