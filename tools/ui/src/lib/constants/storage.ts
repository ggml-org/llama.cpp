/**
 * Storage-related constants (localStorage, IndexedDB).
 *
 * Centralized to ensure consistency across the app and simplify future
 * name changes.
 */

/** Name prefix for all localStorage keys */
export const STORAGE_APP_NAME = 'LlamaUi';

/** Deprecated localStorage key prefix (old app name) */
export const STORAGE_APP_NAME_DEPRECATED = 'LlamaCppWebui';

/** @deprecated Deprecated IndexedDB name — will be removed after all users have migrated */
export const DB_APP_NAME_DEPRECATED = 'LlamacppWebui';

export const ALWAYS_ALLOWED_TOOLS_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.alwaysAllowedTools`;
export const CONFIG_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.config`;
export const DISABLED_TOOLS_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.disabledTools`;
export const FAVORITE_MODELS_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.favoriteModels`;
export const MCP_DEFAULT_ENABLED_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.mcpDefaultEnabled`;
export const USER_OVERRIDES_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.userOverrides`;

// Deprecated old key names (kept for backward compat while users migrate)
/** @deprecated Use {@link ALWAYS_ALLOWED_TOOLS_LOCALSTORAGE_KEY} instead */
export const DEPRECATED_ALWAYS_ALLOWED_TOOLS_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME_DEPRECATED}.alwaysAllowedTools`;
/** @deprecated Use {@link CONFIG_LOCALSTORAGE_KEY} instead */
export const DEPRECATED_CONFIG_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME_DEPRECATED}.config`;
/** @deprecated Use {@link DISABLED_TOOLS_LOCALSTORAGE_KEY} instead */
export const DEPRECATED_DISABLED_TOOLS_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME_DEPRECATED}.disabledTools`;
/** @deprecated Use {@link FAVORITE_MODELS_LOCALSTORAGE_KEY} instead */
export const DEPRECATED_FAVORITE_MODELS_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME_DEPRECATED}.favoriteModels`;
/** @deprecated Use {@link MCP_DEFAULT_ENABLED_LOCALSTORAGE_KEY} instead */
export const DEPRECATED_MCP_DEFAULT_ENABLED_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME_DEPRECATED}.mcpDefaultEnabled`;
/** @deprecated Use {@link USER_OVERRIDES_LOCALSTORAGE_KEY} instead */
export const DEPRECATED_USER_OVERRIDES_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME_DEPRECATED}.userOverrides`;

// Maps new keys to their deprecated fallback keys for migration
const NEW_TO_DEPRECATED_MAP: Record<string, string> = {
	[ALWAYS_ALLOWED_TOOLS_LOCALSTORAGE_KEY]: DEPRECATED_ALWAYS_ALLOWED_TOOLS_LOCALSTORAGE_KEY,
	[CONFIG_LOCALSTORAGE_KEY]: DEPRECATED_CONFIG_LOCALSTORAGE_KEY,
	[DISABLED_TOOLS_LOCALSTORAGE_KEY]: DEPRECATED_DISABLED_TOOLS_LOCALSTORAGE_KEY,
	[FAVORITE_MODELS_LOCALSTORAGE_KEY]: DEPRECATED_FAVORITE_MODELS_LOCALSTORAGE_KEY,
	[MCP_DEFAULT_ENABLED_LOCALSTORAGE_KEY]: DEPRECATED_MCP_DEFAULT_ENABLED_LOCALSTORAGE_KEY,
	[USER_OVERRIDES_LOCALSTORAGE_KEY]: DEPRECATED_USER_OVERRIDES_LOCALSTORAGE_KEY
};

/**
 * Reads a localStorage value using the new key, falling back to the deprecated key.
 * This ensures existing user data is not lost after the rename.
 */
export function readLocalStorageWithFallback(key: string): string | null {
	const value = localStorage.getItem(key);
	if (value !== null) return value;

	const deprecatedKey = NEW_TO_DEPRECATED_MAP[key];
	if (deprecatedKey) {
		const oldValue = localStorage.getItem(deprecatedKey);
		if (oldValue !== null) {
			// Migrate to new key and clean up old one
			try {
				localStorage.setItem(key, oldValue);
				localStorage.removeItem(deprecatedKey);
			} catch {
				// Ignore storage errors
			}
			return oldValue;
		}
	}

	return null;
}
