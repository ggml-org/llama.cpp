import { browser } from '$app/environment';
import { DISABLED_SKILL_IDS_LOCALSTORAGE_KEY } from '$lib/constants';
import { SvelteSet } from 'svelte/reactivity';
import { persisted } from './persisted.svelte';

/** Persist disabled Skills by opaque server-owned catalog ID. */
export class SkillAvailabilityStore {
	private _disabledIds = $state(new SvelteSet<string>());
	private _storage = persisted<unknown>(DISABLED_SKILL_IDS_LOCALSTORAGE_KEY, []);

	constructor() {
		if (!browser) {
			return;
		}

		const stored = this._storage.value;

		if (Array.isArray(stored) && stored.every((item) => typeof item === 'string')) {
			this._disabledIds = new SvelteSet(stored);
		}
	}

	isDisabled(id: string): boolean {
		return this._disabledIds.has(id);
	}

	setEnabled(id: string, enabled: boolean): void {
		if (enabled) {
			if (!this._disabledIds.has(id)) {
				return;
			}
			this._disabledIds.delete(id);
		} else {
			if (this._disabledIds.has(id)) {
				return;
			}
			this._disabledIds.add(id);
		}

		this._storage.value = [...this._disabledIds];
	}

	get disabledIds(): ReadonlySet<string> {
		return this._disabledIds;
	}
}

/** Shared persisted Skills availability store. */
export const skillAvailabilityStore = new SkillAvailabilityStore();
