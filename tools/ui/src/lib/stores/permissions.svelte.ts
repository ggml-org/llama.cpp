/**
 * permissionsStore - Allowed tool permissions
 *
 * Owns the set of tools the user has permanently allowed, persisted to
 * localStorage. The agentic loop's permission gates consult it to run a
 * tool without prompting.
 */

import { browser } from '$app/environment';
import { ALWAYS_ALLOWED_TOOLS_LOCALSTORAGE_KEY } from '$lib/constants';
import { SvelteSet } from 'svelte/reactivity';

class PermissionsStore {
	private _tools = $state(new SvelteSet<string>());

	get tools(): ReadonlySet<string> {
		return this._tools;
	}

	allowTool(key: string): void {
		this._tools.add(key);
		this.persist();
	}

	allowTools(keys: string[]): void {
		for (const key of keys) this._tools.add(key);
		this.persist();
	}

	hasTool(key: string): boolean {
		return this._tools.has(key);
	}

	/**
	 * Load persisted permissions. Called by initStores() after migrations
	 * have run.
	 */
	initialize(): void {
		// browser-only init: skip on SSR to avoid localStorage side effects
		if (!browser) return;

		try {
			const stored = localStorage.getItem(ALWAYS_ALLOWED_TOOLS_LOCALSTORAGE_KEY);

			if (stored) {
				for (const name of JSON.parse(stored) as string[]) {
					if (typeof name === 'string') this._tools.add(name);
				}
			}
		} catch (err) {
			console.error(
				`Failed to load permissions from localStorage ("${ALWAYS_ALLOWED_TOOLS_LOCALSTORAGE_KEY}"):`,
				err
			);
		}
	}

	revokeTool(key: string): void {
		this._tools.delete(key);
		this.persist();
	}

	private persist(): void {
		try {
			localStorage.setItem(ALWAYS_ALLOWED_TOOLS_LOCALSTORAGE_KEY, JSON.stringify([...this._tools]));
		} catch (err) {
			console.error(
				`Failed to persist to localStorage ("${ALWAYS_ALLOWED_TOOLS_LOCALSTORAGE_KEY}"):`,
				err
			);
		}
	}
}

export const permissionsStore = new PermissionsStore();
