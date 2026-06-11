/**
 * buildInfoStore - Build information
 *
 * Reads the build version from `build.json` — embedded at llama.cpp build time
 * with the llama.cpp build number (LLAMA_BUILD_NUMBER). This lets the UI detect
 * server upgrades regardless of how the app was deployed (PWA or not).
 *
 * In dev mode (via `npm run dev`), falls back to `import.meta.env.DEV`'s truthy
 * value as a placeholder since the artifact is not produced.
 *
 * Note: `_app/version.json` is SvelteKit's artifact for SW precache only.
 * Do not use it for frontend version detection — it changes with every npm build
 * but not necessarily with every llama.cpp server upgrade.
 */

import { browser } from '$app/environment';
import { base } from '$app/paths';
import { SvelteURLSearchParams } from 'svelte/reactivity';

let version = $state<string>('');

async function loadVersion() {
	if (!browser) return;

	// In dev mode the version.json artifact doesn't exist - use a simple fallback
	if (import.meta.env.DEV) {
		version = 'dev';
		return;
	}

	try {
		const res = await fetch(`${base}/build.json`, { cache: 'no-store' });
		if (res.ok) {
			const data = await res.json();
			version = data.version ?? '';
		}
	} catch {
		// build.json missing or unreachable - leave as empty string
	}
}

loadVersion();

/** Returns true when the app is running as an installed PWA (via ?pwa=1 in the URL).
 *  This allows the frontend to distinguish "running as installed app" vs "running in browser". */
export function isPwaMode(): boolean {
	return browser && new SvelteURLSearchParams(window.location.search).has('pwa');
}

export const buildInfoStore = {
	get value(): string {
		return version;
	}
};
