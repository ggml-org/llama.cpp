/**
 * LocalStorage service — simple localStorage access with automatic migration.
 *
 * Provides a thin wrapper around localStorage.getItem/setItem that handles
 * the transition from the old `LlamaCppWebui` key prefix to the new `LlamaUi`
 * prefix automatically.
 *
 * **Architecture:**
 * - `localStorageGetItem(key)` — regular getter that also checks deprecated keys
 * - `migrateFromDeprecatedKey(key)` — migration helper, called internally
 * - `localStorageSetItem(key, value)` — regular setter (no migration needed on write)
 *
 * @see $lib/constants/storage.ts for all key constants
 */

import { NEW_TO_DEPRECATED_MAP } from '$lib/constants';

/**
 * Get a value from localStorage.
 * If the new key doesn't exist, falls back to the deprecated key and migrates
 * the value to the new key so future reads use the new key directly.
 */
export function localStorageGetItem(key: string): string | null {
	const value = localStorage.getItem(key);
	if (value !== null) return value;

	migrateFromDeprecatedKey(key);

	return localStorage.getItem(key);
}

/**
 * Set a value in localStorage. No migration needed — always write to the new key.
 */
export function localStorageSetItem(key: string, value: string): void {
	localStorage.setItem(key, value);
}

/**
 * Attempt to migrate a value from a deprecated localStorage key to its new key.
 * If the deprecated key exists, copies its value to the new key and removes the old key.
 */
function migrateFromDeprecatedKey(key: string): void {
	const deprecatedKey = NEW_TO_DEPRECATED_MAP[key];
	if (!deprecatedKey) return;

	const oldValue = localStorage.getItem(deprecatedKey);
	if (oldValue === null) return;

	// Migrate: write to new key, remove old key
	try {
		localStorage.setItem(key, oldValue);
		localStorage.removeItem(deprecatedKey);
	} catch {
		// Ignore storage errors (e.g., storage full)
	}
}
