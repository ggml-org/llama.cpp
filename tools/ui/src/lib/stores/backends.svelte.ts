/**
 * backendsStore - API endpoints the UI can talk to.
 *
 * The built-in local backend is the llama-server serving this UI. External
 * backends are user-configured endpoints persisted in settings. The store
 * registers the resolved list with the api-base registry, which services use
 * to build request URLs.
 */

import { browser } from '$app/environment';
import { ACTIVE_BACKEND_LOCALSTORAGE_KEY, LOCAL_BACKEND_ID, SETTINGS_KEYS } from '$lib/constants';
import { serverStore } from '$lib/stores/server.svelte';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import type { Backend } from '$lib/types';
import { setBackendsResolver } from '$lib/utils/api-base';
import { createLocalBackend, parseBackendsSettings } from '$lib/utils/backend';

function loadActiveBackendId(): string {
	if (!browser) return LOCAL_BACKEND_ID;

	try {
		return localStorage.getItem(ACTIVE_BACKEND_LOCALSTORAGE_KEY) ?? LOCAL_BACKEND_ID;
	} catch {
		return LOCAL_BACKEND_ID;
	}
}

function persistActiveBackendId(backendId: string): void {
	if (!browser) return;

	try {
		localStorage.setItem(ACTIVE_BACKEND_LOCALSTORAGE_KEY, backendId);
	} catch {
		/* ignore */
	}
}

class BackendsStore {
	activeId = $state<string>(loadActiveBackendId());

	get active(): Backend {
		const active = this.enabled.find((backend) => backend.id === this.activeId);

		return active ?? this.enabled[0] ?? this.local;
	}

	get enabled(): Backend[] {
		return this.list.filter((backend) => backend.enabled && !this.isMissingLocal(backend));
	}

	get external(): Backend[] {
		return parseBackendsSettings(settingsStore.config[SETTINGS_KEYS.BACKENDS]);
	}

	get list(): Backend[] {
		return [this.local, ...this.external];
	}

	get local(): Backend {
		return createLocalBackend(
			settingsStore.config.apiKey?.toString().trim() || undefined,
			settingsStore.config[SETTINGS_KEYS.LOCAL_BACKEND_ENABLED] !== false
		);
	}

	addBackend(backend: Backend): void {
		this.saveExternal([...this.external, backend]);
	}

	initialize(): void {
		if (!browser) return;

		setBackendsResolver(() => ({ activeId: this.active.id, backends: this.list }));
	}

	removeBackend(backendId: string): void {
		this.saveExternal(this.external.filter((backend) => backend.id !== backendId));

		if (this.activeId === backendId) {
			this.setActive(LOCAL_BACKEND_ID);
		}
	}

	setActive(backendId: string): void {
		const id = this.list.some((backend) => backend.id === backendId) ? backendId : LOCAL_BACKEND_ID;

		this.activeId = id;
		persistActiveBackendId(id);
	}

	setLocalEnabled(enabled: boolean): void {
		settingsStore.updateConfig(SETTINGS_KEYS.LOCAL_BACKEND_ENABLED, enabled);
	}

	updateBackend(backendId: string, updates: Partial<Backend>): void {
		this.saveExternal(
			this.external.map((backend) =>
				backend.id === backendId ? { ...backend, ...updates } : backend
			)
		);
	}

	/** The built-in backend counts only when a local server answered this session. */
	private isMissingLocal(backend: Backend): boolean {
		return backend.id === LOCAL_BACKEND_ID && serverStore.localServerMissing;
	}

	private saveExternal(backends: Backend[]): void {
		settingsStore.updateConfig(SETTINGS_KEYS.BACKENDS, JSON.stringify(backends));
	}
}

export const backendsStore = new BackendsStore();
