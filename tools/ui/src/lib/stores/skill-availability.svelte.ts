import { browser } from '$app/environment';
import { DISABLED_SKILL_IDS_LOCALSTORAGE_KEY } from '$lib/constants';
import type { SkillCatalogEntry } from '$lib/types';
import { SvelteSet } from 'svelte/reactivity';
import { persisted } from './persisted.svelte';

/**
 * Reactive set of opaque server-owned catalog entry IDs the user has disabled.
 * Only opaque IDs are ever stored — names/scope/provider/path are never used
 * as keys, and no name-based migration exists. Invalid stored shapes fall back
 * to an empty set; unknown stored IDs are preserved without rewriting storage.
 */
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

	enabledEntries(entries: readonly SkillCatalogEntry[]): readonly SkillCatalogEntry[] {
		if (this._disabledIds.size === 0) {
			return entries;
		}

		return entries.filter((entry) => !this._disabledIds.has(entry.id));
	}

	get disabledIds(): ReadonlySet<string> {
		return this._disabledIds;
	}
}

/** Production singleton, persisted across sessions via the DISABLED_SKILL_IDS key. */
export const skillAvailabilityStore = new SkillAvailabilityStore();