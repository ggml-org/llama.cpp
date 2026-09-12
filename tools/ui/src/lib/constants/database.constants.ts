/**
 * Database-related constants (IndexedDB, Dexie).
 *
 * Centralized to ensure consistency across the app and simplify future
 * naming changes.
 */

import { STORAGE_APP_NAME } from './storage.constants';

/** IndexedDB database name */
export const DB_NAME = STORAGE_APP_NAME;

/** IndexedDB store / table names */
export const IDXDB_TABLES = {
	conversations: 'conversations',
	memoryEntries: 'memoryEntries',
	messages: 'messages'
} as const;

/** IndexedDB store schemas */
export const IDXDB_STORE_SCHEMAS = {
	conversations: 'id, lastModified, currNode, name',
	memoryEntries: 'name, updated',
	messages: 'id, convId, type, role, timestamp, parent, children'
} as const;

/** Dexie stores of schema version 1 — keys are table names, values are schemas */
export const IDXDB_STORES = {
	[IDXDB_TABLES.conversations]: IDXDB_STORE_SCHEMAS.conversations,
	[IDXDB_TABLES.messages]: IDXDB_STORE_SCHEMAS.messages
} as const;

/** Dexie stores added by schema version 2 — the memory entries table */
export const IDXDB_STORES_MEMORY = {
	[IDXDB_TABLES.memoryEntries]: IDXDB_STORE_SCHEMAS.memoryEntries
} as const;
