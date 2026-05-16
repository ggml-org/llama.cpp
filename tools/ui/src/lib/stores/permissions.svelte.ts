import { ALWAYS_ALLOWED_TOOLS_LOCALSTORAGE_KEY } from '$lib/constants';

import { SvelteSet } from 'svelte/reactivity';

class PermissionsStore {
	private _tools = $state(new SvelteSet<string>());

	constructor() {
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

	get tools(): ReadonlySet<string> {
		return this._tools;
	}

	hasTool(key: string): boolean {
		return this._tools.has(key);
	}

	allowTool(key: string): void {
		this._tools.add(key);
		this._persist();
	}

	allowTools(keys: string[]): void {
		for (const key of keys) this._tools.add(key);
		this._persist();
	}

	revokeTool(key: string): void {
		this._tools.delete(key);
		this._persist();
	}

	private _persist(): void {
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
