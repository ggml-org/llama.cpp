// Guards hapticTap's three paths: nothing without touch, a short vibration where the browser
// vibrates (Android), and a toggled hidden switch where it does not (iOS), which leaves no
// element behind.

import { HAPTIC_TAP_MS } from '$lib/constants';
import { hapticTap } from '$lib/utils';
import { afterEach, describe, expect, it, vi } from 'vitest';

function setCoarsePointer(coarse: boolean) {
	vi.spyOn(window, 'matchMedia').mockImplementation(
		(query: string) => ({ matches: coarse, media: query }) as MediaQueryList
	);
}

afterEach(() => {
	vi.restoreAllMocks();
	vi.unstubAllGlobals();
});

describe('hapticTap', () => {
	it('does nothing without a touch screen', () => {
		setCoarsePointer(false);
		const vibrate = vi.fn();

		vi.stubGlobal('navigator', { ...navigator, vibrate });
		hapticTap();

		expect(vibrate).not.toHaveBeenCalled();
	});

	it('vibrates briefly where the browser can', () => {
		setCoarsePointer(true);
		const vibrate = vi.fn();

		vi.stubGlobal('navigator', { ...navigator, vibrate });
		hapticTap();

		expect(vibrate).toHaveBeenCalledWith(HAPTIC_TAP_MS);
	});

	it('toggles a hidden switch where it cannot, and removes it', () => {
		setCoarsePointer(true);
		vi.stubGlobal('navigator', { ...navigator, vibrate: undefined });
		const clicks = vi.spyOn(HTMLLabelElement.prototype, 'click');

		hapticTap();

		expect(clicks).toHaveBeenCalledTimes(1);
		expect(document.querySelector('input[switch]')).toBeNull();
	});
});
