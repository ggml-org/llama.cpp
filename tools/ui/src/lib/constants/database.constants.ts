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
	groups: 'groups',
	messages: 'messages',
	projects: 'projects'
} as const;

/** IndexedDB store schemas */
export const IDXDB_STORE_SCHEMAS = {
	conversations: 'id, lastModified, currNode, name',
	groups: 'id, updatedAt, name',
	messages: 'id, convId, type, role, timestamp, parent, children',
	projects: 'id, updatedAt, name'
} as const;

/** Combined Dexie stores definition — keys are table names, values are schemas */
export const IDXDB_STORES = {
	[IDXDB_TABLES.conversations]: IDXDB_STORE_SCHEMAS.conversations,
	[IDXDB_TABLES.groups]: IDXDB_STORE_SCHEMAS.groups,
	[IDXDB_TABLES.messages]: IDXDB_STORE_SCHEMAS.messages,
	[IDXDB_TABLES.projects]: IDXDB_STORE_SCHEMAS.projects
} as const;

/** Stores of schema version 1, before the groups table existed. */
export const IDXDB_STORES_V1 = {
	[IDXDB_TABLES.conversations]: IDXDB_STORE_SCHEMAS.conversations,
	[IDXDB_TABLES.messages]: IDXDB_STORE_SCHEMAS.messages
} as const;

/** Schema version that added the groups table. */
export const IDXDB_VERSION_GROUPS = 2;

/** Schema version new Dexie instances open the database at; bump with a new version block. */
export const IDXDB_VERSION = 3;
