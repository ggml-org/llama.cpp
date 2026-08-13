/**
 * Explicit store initialization, called once from the root layout.
 *
 * Order matters: migrations run first because they rename and rewrite
 * localStorage keys, so every store that reads localStorage initializes
 * only after they complete. Constructors and module-level side effects
 * stay empty so import order can no longer change startup behavior.
 */

// direct imports, not via the barrel, to avoid circular deps
import { conversationsStore } from './conversations.svelte';
import { permissionsStore } from './permissions.svelte';
import { settingsStore } from './settings.svelte';
import { toolsStore } from './tools.svelte';
import { versionStore } from './version.svelte';
import { browser } from '$app/environment';
import { MigrationService } from '$lib/services/migration.service';

let started = false;

export async function initStores(): Promise<void> {
	if (!browser || started) return;

	started = true;

	await MigrationService.runAllMigrations();

	settingsStore.initialize();
	permissionsStore.initialize();
	toolsStore.initialize();
	void versionStore.initialize();

	await conversationsStore.init();
}
