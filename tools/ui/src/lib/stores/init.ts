// direct imports, not via the barrel, to avoid circular deps
import { backendsStore } from './backends.svelte';
import { backendsModelsStore } from './backendsModels.svelte';
import { conversationsStore } from './conversations/index.svelte';
import { permissionsStore } from './permissions.svelte';
import { serverStore } from './server.svelte';
import { settingsStore } from './settings/index.svelte';
import { tabsStore } from './tabs.svelte';
import { toolsStore } from './tools.svelte';
import { versionStore } from './version.svelte';
import { browser } from '$app/environment';
import { LOCAL_BACKEND_ID } from '$lib/constants';
import { MigrationService } from '$lib/services/migration.service';

let startup: Promise<void> | null = null;

export function initStores(): Promise<void> {
	if (!browser) return Promise.resolve();

	startup ??= (async () => {
		await MigrationService.runAllMigrations();

		settingsStore.initialize();
		backendsStore.initialize();

		// prefetch every backend's model list in the background; failures are
		// per-backend and never block startup
		void backendsModelsStore.loadAll();

		// the local server state is needed once its tab is opened; loading it here
		// keeps the tab switch free of /props requests
		if (backendsStore.local.enabled && backendsStore.active.id !== LOCAL_BACKEND_ID) {
			void serverStore.prefetchLocalState();
		}

		permissionsStore.initialize();
		toolsStore.initialize();
		void versionStore.initialize();

		// the full conversation list loads in the background; once it is back,
		// prune persisted tabs against the conversations that still exist
		void conversationsStore.initialize().then(() => {
			tabsStore.init(conversationsStore.conversations.map((c) => c.id));
		});
	})();

	return startup;
}
