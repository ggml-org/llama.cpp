/**
 * backendsStore - API endpoints the UI can talk to.
 *
 * The built-in local backend is the llama-server serving this UI. External
 * backends are user-configured endpoints read from settings. The store
 * registers the resolved list with the api-base registry, which services use
 * to build request URLs.
 */

import { browser } from '$app/environment';
import { LOCAL_BACKEND_ID, SETTINGS_KEYS } from '$lib/constants';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import type { Backend } from '$lib/types';
import { setBackendsResolver } from '$lib/utils/api-base';
import { createLocalBackend, parseBackendsSettings } from '$lib/utils/backend';

class BackendsStore {
	activeId = $state<string>(LOCAL_BACKEND_ID);

	get active(): Backend {
		return this.list.find((backend) => backend.id === this.activeId) ?? this.local;
	}

	get enabled(): Backend[] {
		return this.list.filter((backend) => backend.enabled);
	}

	get external(): Backend[] {
		return parseBackendsSettings(settingsStore.config[SETTINGS_KEYS.BACKENDS]);
	}

	get list(): Backend[] {
		return [this.local, ...this.external];
	}

	get local(): Backend {
		return createLocalBackend();
	}

	initialize(): void {
		if (!browser) return;

		setBackendsResolver(() => ({ activeId: this.activeId, backends: this.list }));
	}

	setActive(backendId: string): void {
		this.activeId = this.list.some((backend) => backend.id === backendId)
			? backendId
			: LOCAL_BACKEND_ID;
	}
}

export const backendsStore = new BackendsStore();
