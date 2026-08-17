/** LocalStorage and IndexedDB names. */

/** Prefix for application localStorage keys. */
export const STORAGE_APP_NAME = 'LlamaUi';

/** Legacy localStorage prefix retained for migration. */
export const STORAGE_APP_NAME_DEPRECATED = 'LlamaCppWebui';

/** Legacy IndexedDB name retained for migration. */
export const DB_APP_NAME_DEPRECATED = 'LlamacppWebui';

export const ALWAYS_ALLOWED_TOOLS_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.alwaysAllowedTools`;
export const CONFIG_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.config`;
export const DISABLED_TOOLS_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.disabledTools`;
export const DISABLED_SKILL_IDS_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.disabledSkillIds`;
export const SKILLS_PANE_SIZES_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.skillsPaneSizes`;

/** Disabled tools keyed by stable selection identity. */
export const DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.disabledToolKeys`;
export const FAVORITE_MODELS_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.favoriteModels`;
export const REASONING_EFFORT_DEFAULT_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.reasoningEffortDefault`;
export const USER_OVERRIDES_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.userOverrides`;
export const DISMISSED_RECOMMENDED_MCP_SERVERS_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.dismissedRecommendedMcpServers`;
export const STREAM_RESUME_LOCALSTORAGE_KEY_PREFIX = `${STORAGE_APP_NAME}.streamResume.`;


// Legacy key names retained during migration.
/** @deprecated Use {@link ALWAYS_ALLOWED_TOOLS_LOCALSTORAGE_KEY} instead */
export const DEPRECATED_ALWAYS_ALLOWED_TOOLS_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME_DEPRECATED}.alwaysAllowedTools`;
/** @deprecated Use {@link CONFIG_LOCALSTORAGE_KEY} instead */
export const DEPRECATED_CONFIG_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME_DEPRECATED}.config`;
/** @deprecated Use {@link DISABLED_TOOLS_LOCALSTORAGE_KEY} instead */
export const DEPRECATED_DISABLED_TOOLS_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME_DEPRECATED}.disabledTools`;
/** @deprecated Use {@link FAVORITE_MODELS_LOCALSTORAGE_KEY} instead */
export const DEPRECATED_FAVORITE_MODELS_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME_DEPRECATED}.favoriteModels`;
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
	[USER_OVERRIDES_LOCALSTORAGE_KEY]: DEPRECATED_USER_OVERRIDES_LOCALSTORAGE_KEY
};
