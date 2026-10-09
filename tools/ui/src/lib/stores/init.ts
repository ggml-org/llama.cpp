// direct imports, not via the barrel, to avoid circular deps
import { backendsStore } from './backends.svelte';
import { backendsModelsStore } from './backendsModels.svelte';
import { conversationsStore } from './conversations/index.svelte';
import { modelsStore } from './models/index.svelte';
import { permissionsStore } from './permissions.svelte';
import { serverStore } from './server.svelte';
import { settingsStore } from './settings/index.svelte';
import { tabsStore } from './tabs.svelte';
import { toolsStore } from './tools.svelte';
import { versionStore } from './version.svelte';
import { browser } from '$app/environment';
import { MigrationService } from '$lib/services/migration.service';

let hydration: Promise<void> | null = null;
let startup: Promise<void> | null = null;

/**
 * Read the stored state the first request needs: migrations, the settings that
 * carry the API key, and the backend that request goes to.
 *
 * Separate from {@link initStores} so a `load` can await it. The requests
 * `initStores` starts belong after a load, where plain `window.fetch` is the
 * intended one and SvelteKit raises no warning.
 */
export function hydrateStores(): Promise<void> {
	if (!browser) return Promise.resolve();

	hydration ??= (async () => {
		await MigrationService.runAllMigrations();

		settingsStore.initialize();
		backendsStore.initialize();
	})();

	return hydration;
}

export function initStores(): Promise<void> {
	if (!browser) return Promise.resolve();

	startup ??= (async () => {
		await hydrateStores();

		// prefetch every backend's model list in the background; failures are
		// per-backend and never block startup
		void backendsModelsStore.loadAll().then(() => modelsStore.warmHubDetails());

		permissionsStore.initialize();
		toolsStore.initialize();

		// the local server state backs the installation facts and decides whether
		// /tools exists at all, so probe it first and only then list the tools;
		// otherwise they stay empty until the tools menu is opened
		void serverStore.prefetchLocalState().then(() => toolsStore.fetchServerTools());
		void versionStore.initialize();

		// the full conversation list loads in the background; once it is back,
		// prune persisted tabs against the conversations that still exist
		void conversationsStore.initialize().then(() => {
			tabsStore.init(conversationsStore.conversations.map((c) => c.id));
		});
	})();

	return startup;
}
