import { ALWAYS_ALLOWED_TOOLS_LOCALSTORAGE_KEY } from '$lib/constants';
import { SvelteSet } from 'svelte/reactivity';

class PermissionsStore {
	private _alwaysAllowedTools = $state(new SvelteSet<string>());

	constructor() {
		try {
			const storedTools = localStorage.getItem(ALWAYS_ALLOWED_TOOLS_LOCALSTORAGE_KEY);
			if (storedTools) {
				const parsedTools = JSON.parse(storedTools) as unknown;
				if (Array.isArray(parsedTools)) {
					for (const name of parsedTools) {
						if (typeof name === 'string') this._alwaysAllowedTools.add(name);
					}
				}
			}
		} catch {
			/* ignore */
		}
	}

	get alwaysAllowedTools(): ReadonlySet<string> {
		return this._alwaysAllowedTools;
	}

	isAlwaysAllowed(toolName: string): boolean {
		return this._alwaysAllowedTools.has(toolName);
	}

	alwaysAllow(toolName: string): void {
		this._alwaysAllowedTools.add(toolName);
		this.persist();
	}

	alwaysAllowServer(toolNames: string[]): void {
		for (const name of toolNames) {
			this._alwaysAllowedTools.add(name);
		}
		this.persist();
	}

	revokeAlwaysAllow(toolName: string): void {
		this._alwaysAllowedTools.delete(toolName);
		this.persist();
	}

	revokeAlwaysAllowServer(toolNames: string[]): void {
		for (const name of toolNames) {
			this._alwaysAllowedTools.delete(name);
		}
		this.persist();
	}

	private persist(): void {
		try {
			localStorage.setItem(
				ALWAYS_ALLOWED_TOOLS_LOCALSTORAGE_KEY,
				JSON.stringify([...this._alwaysAllowedTools])
			);
		} catch {
			/* ignore */
		}
	}
}

export const permissionsStore = new PermissionsStore();
