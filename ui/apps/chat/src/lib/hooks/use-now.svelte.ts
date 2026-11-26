import { MS_PER_SECOND } from '#lib/constants/index.ts';

/**
 * The current time (ms since epoch), ticking once a second while `isTicking`
 * returns true. Call during component init.
 */
export function useNow(isTicking: () => boolean = () => true): { readonly current: number } {
	let now = $state(Date.now());

	$effect(() => {
		if (!isTicking()) return;

		now = Date.now();

		const timer = setInterval(() => (now = Date.now()), MS_PER_SECOND);

		return () => clearInterval(timer);
	});

	return {
		get current() {
			return now;
		}
	};
}
