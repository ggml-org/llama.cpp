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

/** Disabled tools keyed by stable selection identity, no migration from the name based key */
export const DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.disabledToolKeys`;
export const FAVORITE_MODELS_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.favoriteModels`;
export const MCP_DEFAULT_ENABLED_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.mcpDefaultEnabled`;
export const THINKING_ENABLED_DEFAULT_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.thinkingEnabledDefault`;
export const REASONING_EFFORT_DEFAULT_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.reasoningEffortDefault`;
export const USER_OVERRIDES_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.userOverrides`;

/** Key prefix for per-conversation resumable stream state, conversationId is appended */
export const STREAM_RESUME_LOCALSTORAGE_KEY_PREFIX = `${STORAGE_APP_NAME}.streamResume.`;

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

/** Build version stored in localStorage for non-PWA update detection */
export const BUILD_VERSION_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.buildVersion`;

/** Maps new keys to their deprecated fallback keys */
export const NEW_TO_DEPRECATED_MAP: Record<string, string> = {
	[ALWAYS_ALLOWED_TOOLS_LOCALSTORAGE_KEY]: DEPRECATED_ALWAYS_ALLOWED_TOOLS_LOCALSTORAGE_KEY,
	[CONFIG_LOCALSTORAGE_KEY]: DEPRECATED_CONFIG_LOCALSTORAGE_KEY,
	[DISABLED_TOOLS_LOCALSTORAGE_KEY]: DEPRECATED_DISABLED_TOOLS_LOCALSTORAGE_KEY,
	[FAVORITE_MODELS_LOCALSTORAGE_KEY]: DEPRECATED_FAVORITE_MODELS_LOCALSTORAGE_KEY,
	[MCP_DEFAULT_ENABLED_LOCALSTORAGE_KEY]: DEPRECATED_MCP_DEFAULT_ENABLED_LOCALSTORAGE_KEY,
	[USER_OVERRIDES_LOCALSTORAGE_KEY]: DEPRECATED_USER_OVERRIDES_LOCALSTORAGE_KEY
};
