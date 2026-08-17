// direct imports, not via the barrel, to avoid circular deps
import { conversationsStore } from './conversations/index.svelte';
import { permissionsStore } from './permissions.svelte';
import { settingsStore } from './settings/index.svelte';
import { tabsStore } from './tabs.svelte';
import { toolsStore } from './tools.svelte';
import { versionStore } from './version.svelte';
import { browser } from '$app/environment';
import { MigrationService } from '$lib/services/migration.service';

let startup: Promise<void> | null = null;

export function initStores(): Promise<void> {
	if (!browser) return Promise.resolve();

	startup ??= (async () => {
		await MigrationService.runAllMigrations();

		settingsStore.initialize();
		permissionsStore.initialize();
		toolsStore.initialize();
		void versionStore.initialize();
		await conversationsStore.initialize();

		// prune persisted tabs against the loaded conversation list
		tabsStore.init(conversationsStore.conversations.map((c) => c.id));
	})();

	return startup;
}
