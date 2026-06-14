import { browser } from '$app/environment';
import { useRegisterSW } from 'virtual:pwa-register/svelte';
import { versionStore } from '$lib/stores/version.svelte';
import { BUILD_VERSION_LOCALSTORAGE_KEY } from '$lib/constants/storage';
import { SW_CONFIG } from '$lib/constants/pwa';

/**
 * Hook for PWA service worker registration, update polling, and build version mismatch detection.
 *
 * Combines two concerns that always belong together:
 * 1. SW registration with periodic polling for updates
 * 2. localStorage-based version tracking for non-PWA users
 */
export function usePwa() {
	let swCheckInterval: ReturnType<typeof setInterval> | null = null;
	let needRefreshByStorage = $state(false);

	const {
		// offlineReady, // to do - add installation banners for iOS
		needRefresh: pwaNeedRefresh,
		updateServiceWorker
	} = useRegisterSW({
		onRegisteredSW(swUrl: string, r: ServiceWorkerRegistration | undefined) {
			if (swCheckInterval) {
				clearInterval(swCheckInterval);
			}
			swCheckInterval = setInterval(async () => {
				if (!r || r.installing || !navigator?.onLine) return;

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
		},
		onRegisterError(error: unknown) {
			console.error('[PWA] SW registration error:', error);
		}
	});

	// Detect version mismatch via localStorage.
	// _app/version.json is SvelteKit's native version file for PWA cache invalidation.
	// This comparison detects server upgrades for non-PWA users.
	$effect(() => {
		if (!browser) return;

		const currentVersion = versionStore.value;
		if (!currentVersion) return;

		try {
			const storedVersion = localStorage.getItem(BUILD_VERSION_LOCALSTORAGE_KEY);
			needRefreshByStorage = !!storedVersion && storedVersion !== currentVersion;
			localStorage.setItem(BUILD_VERSION_LOCALSTORAGE_KEY, currentVersion);
		} catch {
			needRefreshByStorage = false;
		}
	});

	return {
		/** Writable that is true when a PWA service worker update is available */
		get needRefresh() {
			return pwaNeedRefresh;
		},
		updateServiceWorker,
		/** Version mismatch detected via localStorage (non-PWA users) */
		get needRefreshByStorage() {
			return needRefreshByStorage;
		}
	};
}
