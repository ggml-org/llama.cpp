/**
 * Creates a reactive throttle key that increments when `getValue()` changes
 * and the throttle window has elapsed since the last increment.
 *
 * Useful for throttling animations that should not fire on every rapid update.
 *
 * @param getValue - A reactive getter for the value to watch
 * @param ms - Throttle window in milliseconds
 * @returns A reactive number that increments when the throttled value changes
 */
export function useThrottle(getValue: () => string | undefined, ms: number) {
	let key = $state(0);
	let throttleEnd = $state(0);
	let lastValue: string | undefined = getValue();

	$effect(() => {
		const value = getValue();
		if (value === lastValue) return;
		const now = Date.now();
		if (now >= throttleEnd) {
			lastValue = value;
			key++;
			throttleEnd = now + ms;
		}
	});

	return {
		get key() {
			return key;
		}
	};
}
