/**
 * buildInfoStore - llama.cpp build information
 *
 * Reads the build version from `build.json` — embedded at llama.cpp build time
 * with the llama.cpp build number (LLAMA_BUILD_NUMBER). Shown in the UI when
 * `showBuildVersion` is enabled.
 *
 * In dev mode (via `npm run dev`), falls back to `import.meta.env.DEV`'s truthy
 * value since the artifact is not produced.
 */

import { browser } from '$app/environment';
import { base } from '$app/paths';

let build = $state<string>('');

async function loadBuild() {
	if (!browser) return;

	if (import.meta.env.DEV) {
		build = 'dev';
		return;
	}

	try {
		const res = await fetch(`${base}/build.json`, { cache: 'no-store' });
		if (res.ok) {
			const data = await res.json();
			build = data.version ?? '';
		}
	} catch {
		// build.json missing or unreachable - leave as empty string
	}
}

loadBuild();

export const buildInfoStore = {
	get value(): string {
		return build;
	}
};
