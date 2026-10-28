/**
 * Storage adapters - where conversations and user settings can be kept besides this browser
 *
 * llama-server's own JSON document store is the built-in adapter. To keep data
 * elsewhere (Postgres via Supabase or Neon, PocketBase, ...), implement
 * StorageAdapter in a file next to this one and add its definition below.
 *
 * @see tools/ui/docs/storage-adapters.md - the contract and example schemas
 */

import { llamaServerStorageAdapter } from './llama-server.adapter';
import type { StorageAdapterDefinition } from '$lib/types';

export { LlamaServerStorageAdapter } from './llama-server.adapter';
export { StorageConflictError } from './storage-conflict-error';

/** Adapters offered in Settings, in this order. */
export const STORAGE_ADAPTERS: StorageAdapterDefinition[] = [llamaServerStorageAdapter];

export function findStorageAdapter(id: string | null): StorageAdapterDefinition | undefined {
	return STORAGE_ADAPTERS.find((adapter) => adapter.id === id);
}
