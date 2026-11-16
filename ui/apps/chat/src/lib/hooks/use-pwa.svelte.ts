import { BUILD_VERSION_LOCALSTORAGE_KEY, PWA_FILE_PATHS, SW_CONFIG } from '#lib/constants/index.ts';
import { versionStore } from '#lib/stores/index.ts';
import { withBase } from '#lib/utils/base-path.ts';
import { browser, dev } from '$app/env';
import { writable } from 'svelte/store';

/**
 * Hook for PWA service worker registration, update polling, and build version mismatch detection.
 *
 * Combines two concerns that always belong together:
 * 1. SW registration with periodic polling for updates
 * 2. localStorage-based version tracking for non-PWA users
 */
export function usePwa() {
	const needRefresh = writable(false);

	let needRefreshByStorage = $state(false);
	let registration: ServiceWorkerRegistration | undefined;

	// an installed worker next to an active one is a new build waiting for the reload
	function watchInstalling(worker: ServiceWorker | null) {
		worker?.addEventListener('statechange', () => {
			if (worker.state === 'installed' && navigator.serviceWorker.controller) {
				needRefresh.set(true);
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
					needRefresh.set(true);
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

	// Detect version mismatch via localStorage.
	// _app/version.json is SvelteKit's native version file for PWA cache invalidation.
	// This comparison detects server upgrades for non-PWA users.
	$effect(() => {
		if (!browser) return;

		// PWA pages update via the service worker path; the storage check is the non-PWA fallback only
		if (navigator.serviceWorker?.controller) return;

		const currentVersion = versionStore.frontend;

		if (!currentVersion) return;

		try {
			const storedVersion = localStorage.getItem(BUILD_VERSION_LOCALSTORAGE_KEY);

			needRefreshByStorage = !!storedVersion && storedVersion !== currentVersion;
			localStorage.setItem(BUILD_VERSION_LOCALSTORAGE_KEY, currentVersion);
		} catch {
			needRefreshByStorage = false;
		}
	});

	/** Activate the waiting worker, then reload once it controls the page */
	function updateServiceWorker() {
		const waiting = registration?.waiting;

		if (!waiting) return;

		navigator.serviceWorker.addEventListener('controllerchange', () => window.location.reload(), {
			once: true
		});
		waiting.postMessage({ type: 'SKIP_WAITING' });
	}

	return {
		/** Writable that is true when a PWA service worker update is available */
		get needRefresh() {
			return needRefresh;
		},
		/** Version mismatch detected via localStorage (non-PWA users) */
		get needRefreshByStorage() {
			return needRefreshByStorage;
		},
		updateServiceWorker
	};
}
