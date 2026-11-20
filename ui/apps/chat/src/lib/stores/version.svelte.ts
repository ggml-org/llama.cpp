/**
 * versionStore - Build version information
 *
 * `build`: llama.cpp build number from `build.json`, embedded at llama.cpp
 * build time (LLAMA_BUILD_NUMBER). Shown in the UI when `showBuildVersion`
 * is enabled; `'dev'` in dev mode. A newer build of the app itself is noticed by
 * SvelteKit (`updated` from `$app/state`), see usePwa().
 */

import { withBase } from '#lib/utils/base-path.ts';
import { browser } from '$app/env';

class VersionStore {
	build = $state<string>('');

	/**
	 * Fetch the build number. Called by initStores(); order-independent,
	 * so it runs in the background.
	 */
	initialize(): void {
		if (!browser) return;

		if (import.meta.env.DEV) {
			this.build = 'dev';

			return;
		}

		void this.load();
	}

	private async load(): Promise<void> {
		try {
			const res = await fetch(withBase('/build.json'), { cache: 'no-store' });

			if (res.ok) {
				const data = await res.json();

				this.build = data.version ?? '';
			}
		} catch {
			// build.json missing or unreachable - leave as empty string
		}
	}
}

export const versionStore = new VersionStore();
