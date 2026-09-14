/**
 * backendsStore - API endpoints the UI can talk to.
 *
 * The built-in local backend is the llama-server serving this UI. External
 * backends are user-configured endpoints persisted in settings. The store
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
		const active = this.enabled.find((backend) => backend.id === this.activeId);

		return active ?? this.enabled[0] ?? this.local;
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
			this.activeId = LOCAL_BACKEND_ID;
		}
	}

	setActive(backendId: string): void {
		this.activeId = this.list.some((backend) => backend.id === backendId)
			? backendId
			: LOCAL_BACKEND_ID;
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

	private saveExternal(backends: Backend[]): void {
		settingsStore.updateConfig(SETTINGS_KEYS.BACKENDS, JSON.stringify(backends));
	}
}

export const backendsStore = new BackendsStore();
