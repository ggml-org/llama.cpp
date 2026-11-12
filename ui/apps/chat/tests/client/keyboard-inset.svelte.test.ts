// Guards useKeyboardInset: the part of the screen an on-screen keyboard covers is
// published as --keyboard-inset while active, and never while pinch-zoomed.

import { useKeyboardInset } from '$lib/hooks/use-keyboard-inset.svelte';
import { flushSync } from 'svelte';
import { afterEach, describe, expect, it } from 'vitest';

const realViewport = window.visualViewport;

function mockViewport(covered: number, scale = 1) {
	const viewport = Object.assign(new EventTarget(), {
		height: window.innerHeight - covered,
		offsetTop: 0,
		scale
	});

	Object.defineProperty(window, 'visualViewport', { configurable: true, value: viewport });

	return viewport;
}

function inset() {
	return document.documentElement.style.getPropertyValue('--keyboard-inset');
}

afterEach(() => {
	Object.defineProperty(window, 'visualViewport', { configurable: true, value: realViewport });
});

describe('useKeyboardInset', () => {
	it('publishes the covered height and clears it on cleanup', () => {
		const viewport = mockViewport(0);
		const stop = $effect.root(() => useKeyboardInset(() => true));

		flushSync();
		expect(inset()).toBe('');

		viewport.height = window.innerHeight - 300;
		viewport.dispatchEvent(new Event('resize'));
		expect(inset()).toBe('300px');

		stop();
		expect(inset()).toBe('');
	});

	it('reads nothing while zoomed or inactive', () => {
		mockViewport(300, 2);

		const stopZoomed = $effect.root(() => useKeyboardInset(() => true));

		flushSync();
		expect(inset()).toBe('');
		stopZoomed();

		mockViewport(300);

		const stopInactive = $effect.root(() => useKeyboardInset(() => false));

		flushSync();
		expect(inset()).toBe('');
		stopInactive();
	});
});
