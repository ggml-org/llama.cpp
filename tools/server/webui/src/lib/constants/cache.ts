/**
 * Cache configuration constants
 */

/**
 * Default TTL (Time-To-Live) for cache entries in milliseconds
 * @default 5 minutes
 */
export const DEFAULT_CACHE_TTL_MS = 5 * 60 * 1000;

/**
 * Default maximum number of entries in a cache
 * @default 100
 */
export const DEFAULT_CACHE_MAX_ENTRIES = 100;

/**
 * TTL for model props cache in milliseconds
 * Props don't change frequently, so we can cache them longer
 * @default 10 minutes
 */
export const MODEL_PROPS_CACHE_TTL_MS = 10 * 60 * 1000;

/**
 * Maximum number of model props to cache
 * @default 50
 */
export const MODEL_PROPS_CACHE_MAX_ENTRIES = 50;

/**
 * Maximum number of inactive conversation states to keep in memory
 * States for conversations beyond this limit will be cleaned up
 * @default 10
 */
export const MAX_INACTIVE_CONVERSATION_STATES = 10;

/**
 * Maximum age (in ms) for inactive conversation states before cleanup
 * States older than this will be removed during cleanup
 * @default 30 minutes
 */
export const INACTIVE_CONVERSATION_STATE_MAX_AGE_MS = 30 * 60 * 1000;
