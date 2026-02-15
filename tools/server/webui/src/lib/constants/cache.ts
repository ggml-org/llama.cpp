/**
 * Cache configuration constants
 */

/**
 * Default TTL (Time-To-Live) for cache entries in milliseconds.
 */
export const DEFAULT_CACHE_TTL_MS = 5 * 60 * 1000;

/**
 * Default maximum number of entries in a cache.
 */
export const DEFAULT_CACHE_MAX_ENTRIES = 100;

/**
 * TTL for model props cache in milliseconds.
 */
export const MODEL_PROPS_CACHE_TTL_MS = 10 * 60 * 1000;

/**
 * Maximum number of model props to cache.
 */
export const MODEL_PROPS_CACHE_MAX_ENTRIES = 50;

/**
 * Maximum number of inactive conversation states to keep in memory.
 */
export const MAX_INACTIVE_CONVERSATION_STATES = 10;

/**
 * Maximum age (in ms) for inactive conversation states before cleanup.
 */
export const INACTIVE_CONVERSATION_STATE_MAX_AGE_MS = 30 * 60 * 1000;
