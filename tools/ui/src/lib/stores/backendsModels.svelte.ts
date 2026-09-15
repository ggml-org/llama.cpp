/**
 * backendsModelsStore - Per-backend model catalog cache.
 *
 * Prefetches every enabled backend's model list, so switching backends is
 * instant and the switcher can show load state. The active backend's list
 * still lives in modelsStore, which owns selection and chat wiring; this
 * cache is the prefetch layer the switches start from.
 */

import { BackendsService } from '$lib/services/backends.service';
import { backendsStore } from '$lib/stores/backends.svelte';
import type { ModelOption } from '$lib/types/models';

export interface BackendModelsState {
	error: string | null;
	loaded: boolean;
	loading: boolean;
	models: ModelOption[];
}

const EMPTY_STATE: BackendModelsState = { error: null, loaded: false, loading: false, models: [] };

class BackendsModelsStore {
	private states = $state<Record<string, BackendModelsState>>({});

	clear(backendId: string): void {
		delete this.states[backendId];
	}

	/**
	 * Load a backend's models once.
	 */
	async ensureLoaded(backendId: string): Promise<void> {
		const backend = backendsStore.enabled.find((candidate) => candidate.id === backendId);

		if (!backend) return;

		const state = this.states[backendId];

		if (state?.loaded || state?.loading) return;

		this.states[backendId] = { error: null, loaded: false, loading: true, models: [] };

		const result = await BackendsService.listModels(backend);

		this.states[backendId] = {
			error: result.error ?? null,
			loaded: result.ok,
			loading: false,
			models: result.models
		};
	}

	get(backendId: string): BackendModelsState {
		return this.states[backendId] ?? EMPTY_STATE;
	}

	/** Prefetch every enabled backend's model list. */
	async loadAll(): Promise<void> {
		const enabled = backendsStore.enabled;
		const ids = new Set(enabled.map((backend) => backend.id));

		for (const id of Object.keys(this.states)) {
			if (!ids.has(id)) {
				delete this.states[id];
			}
		}

		await Promise.all(enabled.map((backend) => this.ensureLoaded(backend.id)));
	}
}

export const backendsModelsStore = new BackendsModelsStore();
