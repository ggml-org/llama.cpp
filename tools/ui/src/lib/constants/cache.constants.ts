/**
 * Cache configuration constants
 */

/**
 * Default cache limits when no per-cache overrides are given.
 */
export const CACHE = {
	/** Default maximum number of entries in a cache */
	DEFAULT_MAX_ENTRIES: 100,
	/** Default TTL (Time-To-Live) for cache entries in milliseconds (5 minutes) */
	DEFAULT_TTL_MS: 5 * 60 * 1000
} as const;

/**
 * TTL and size for the model props cache.
 * Props don't change frequently, so we can cache them longer.
 */
export const MODEL_PROPS_CACHE = {
	/** Maximum number of model props to cache */
	MAX_ENTRIES: 50,
	/** TTL for model props cache entries in milliseconds (10 minutes) */
	TTL_MS: 10 * 60 * 1000
} as const;

/**
 * TTL and size for the MCP resource cache.
 */
export const MCP_RESOURCE_CACHE = {
	/** Maximum number of MCP resources to cache */
	MAX_ENTRIES: 50,
	/** TTL for MCP resource cache entries in milliseconds (5 minutes) */
	TTL_MS: 5 * 60 * 1000
} as const;

/**
 * Limits for pruning inactive conversation states held in memory.
 */
export const INACTIVE_CONVERSATION = {
	/** Maximum age (in ms) for inactive conversation states before cleanup (30 minutes) */
	MAX_AGE_MS: 30 * 60 * 1000,
	/** Maximum number of inactive conversation states to keep in memory */
	MAX_STATES: 10
} as const;
