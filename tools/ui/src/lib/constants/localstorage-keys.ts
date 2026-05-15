// Key prefix for all localStorage keys
const LS_PREFIX = 'LlamaUi';

// New key names (preferred)
export const ALWAYS_ALLOWED_TOOLS_LOCALSTORAGE_KEY = `${LS_PREFIX}.alwaysAllowedTools`;
export const CONFIG_LOCALSTORAGE_KEY = `${LS_PREFIX}.config`;
export const DISABLED_TOOLS_LOCALSTORAGE_KEY = `${LS_PREFIX}.disabledTools`;
export const FAVORITE_MODELS_LOCALSTORAGE_KEY = `${LS_PREFIX}.favoriteModels`;
export const MCP_DEFAULT_ENABLED_LOCALSTORAGE_KEY = `${LS_PREFIX}.mcpDefaultEnabled`;
export const USER_OVERRIDES_LOCALSTORAGE_KEY = `${LS_PREFIX}.userOverrides`;

// Deprecated old key names (kept for backward compat while users migrate)
export const DEPRECATED_ALWAYS_ALLOWED_TOOLS_LOCALSTORAGE_KEY = 'LlamaCppWebui.alwaysAllowedTools';
export const DEPRECATED_CONFIG_LOCALSTORAGE_KEY = 'LlamaCppWebui.config';
export const DEPRECATED_DISABLED_TOOLS_LOCALSTORAGE_KEY = 'LlamaCppWebui.disabledTools';
export const DEPRECATED_FAVORITE_MODELS_LOCALSTORAGE_KEY = 'LlamaCppWebui.favoriteModels';
export const DEPRECATED_MCP_DEFAULT_ENABLED_LOCALSTORAGE_KEY = 'LlamaCppWebui.mcpDefaultEnabled';
export const DEPRECATED_USER_OVERRIDES_LOCALSTORAGE_KEY = 'LlamaCppWebui.userOverrides';

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
