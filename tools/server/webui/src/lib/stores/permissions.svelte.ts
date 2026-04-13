import { ALWAYS_ALLOWED_TOOLS_LOCALSTORAGE_KEY } from '$lib/constants';
import { SvelteSet } from 'svelte/reactivity';

export type ToolPermissionDecision = 'always' | 'once' | 'deny';

class PermissionsStore {
	private _alwaysAllowedTools = $state(new SvelteSet<string>());

	constructor() {
		try {
			const stored = localStorage.getItem(ALWAYS_ALLOWED_TOOLS_LOCALSTORAGE_KEY);
			if (stored) {
				const parsed = JSON.parse(stored) as unknown;
				if (Array.isArray(parsed)) {
					for (const name of parsed) {
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

	revokeAlwaysAllow(toolName: string): void {
		this._alwaysAllowedTools.delete(toolName);
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
