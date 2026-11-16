import { PWA_FILE_PATHS, SW_CONFIG } from '#lib/constants/index.ts';
import { withBase } from '#lib/utils/base-path.ts';
import { dev } from '$app/env';
import { updated } from '$app/state';

/**
 * Hook for the service worker and for noticing a new build of the app.
 *
 * 1. A page the service worker controls updates through it: the worker is registered and polled,
 *    and a new one waiting to take over is a new build.
 * 2. Any other page relies on SvelteKit, which checks `_app/version.json` on navigation, on focus
 *    and every hour, and sets `updated.current` once the server has a newer build.
 */
export function usePwa() {
	let needRefresh = $state(false);
	let registration: ServiceWorkerRegistration | undefined;

	// read once: the worker serves its precached version.json, so `updated` only means something
	// for a page it doesn't control
	const controlled = !dev && !!globalThis.navigator?.serviceWorker?.controller;

	// an installed worker next to an active one is a new build waiting for the reload
	function watchInstalling(worker: ServiceWorker | null) {
		worker?.addEventListener('statechange', () => {
			if (worker.state === 'installed' && navigator.serviceWorker.controller) {
				needRefresh = true;
			}
		});
	}

	$effect(() => {
		// SvelteKit only builds the service worker for production
		if (dev || !('serviceWorker' in navigator)) return;

		const swUrl = withBase(PWA_FILE_PATHS.SERVICE_WORKER);

		let swCheckInterval: ReturnType<typeof setInterval> | undefined;

		navigator.serviceWorker
			.register(swUrl, { type: 'module' })
			.then((r) => {
				registration = r;

				if (r.waiting && navigator.serviceWorker.controller) {
					needRefresh = true;
				}

				r.addEventListener('updatefound', () => watchInstalling(r.installing));

				swCheckInterval = setInterval(async () => {
					if (r.installing || !navigator?.onLine) return;

					try {
						const resp = await fetch(swUrl, {
							cache: SW_CONFIG.UPDATE_FETCH_OPTIONS.CACHE,
							headers: {
								cache: SW_CONFIG.UPDATE_FETCH_OPTIONS.HEADERS.CACHE,
								'cache-control': SW_CONFIG.UPDATE_FETCH_OPTIONS.HEADERS.CACHE_CONTROL
							}
						});

						if (resp?.status === 200) {
							await r.update();
						}
					} catch (e) {
						console.error(e);
					}
				}, SW_CONFIG.CHECK_INTERVAL_MS);
			})
			.catch((error: unknown) => {
				console.error('[PWA] SW registration error:', error);
			});

		return () => clearInterval(swCheckInterval);
	});

	/** Settles once the worker reaches the state, gives up on it, or the wait runs out */
	function workerState(worker: ServiceWorker, state: ServiceWorkerState): Promise<void> {
		return new Promise((resolve) => {
			const done = () => {
				clearTimeout(timer);
				resolve();
			};
			const timer = setTimeout(done, SW_CONFIG.ACTIVATE_TIMEOUT_MS);

			if (worker.state === state || worker.state === 'redundant') return done();

			worker.addEventListener('statechange', () => {
				if (worker.state === state || worker.state === 'redundant') done();
			});
		});
	}

	/**
	 * Load the new build. A worker for it is fetched if the page has not seen one yet (as on a page
	 * the worker doesn't control), activated in place of the one serving the old build, and then
	 * the page reloads, so a single reload lands on the new build.
	 */
	async function updateServiceWorker() {
		if (registration) {
			await registration.update().catch(() => {});

			const next = registration.installing ?? registration.waiting;

			if (next) {
				await workerState(next, 'installed');
				next.postMessage({ type: 'SKIP_WAITING' });
				await workerState(next, 'activated');
			}
		}

		window.location.reload();
	}

	return {
		/** A new build is waiting: in the service worker, or on the server for an uncontrolled page */
		get needRefresh() {
			return needRefresh || (!controlled && updated.current);
		},
		updateServiceWorker
	};
}
