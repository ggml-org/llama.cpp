import { HAPTIC_TAP_MS, MEDIA_QUERIES } from '$lib/constants';

/**
 * A light tap of the phone's haptic engine, the confirmation a native app gives by feel.
 *
 * Android browsers vibrate. iOS Safari has no vibration API, but toggling a native switch
 * control plays the system's switch haptic (iOS 18 and later), so a hidden one is toggled
 * instead. Either has to run inside the handler of the tap it confirms: called from a timer,
 * iOS stays silent. A screen without touch gets nothing.
 */
export function hapticTap(): void {
	if (typeof window === 'undefined' || !window.matchMedia(MEDIA_QUERIES.COARSE_POINTER).matches)
		return;

	if (typeof navigator.vibrate === 'function') {
		navigator.vibrate(HAPTIC_TAP_MS);

		return;
	}

	const label = document.createElement('label');
	const input = document.createElement('input');

	input.type = 'checkbox';
	input.setAttribute('switch', '');
	label.style.display = 'none';
	label.append(input);
	document.body.append(label);
	label.click();
	label.remove();
}
