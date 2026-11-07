// direct imports, not via the barrel, to avoid circular deps
import { conversationsStore } from './conversations/index.svelte';
import { modelsStore } from './models/index.svelte';
import { permissionsStore } from './permissions.svelte';
import { settingsStore } from './settings/index.svelte';
import { tabsStore } from './tabs.svelte';
import { toolsStore } from './tools.svelte';
import { versionStore } from './version.svelte';
import { browser } from '$app/environment';
import { MigrationService } from '$lib/services/migration.service';
import { StorageModeService } from '$lib/services/storage-mode.service';
import { validateApiKey } from '$lib/utils/api-key-validation';
import { toast } from 'svelte-sonner';

let startup: Promise<void> | null = null;

/** Ask the browser not to evict conversations and settings under storage pressure, best effort. */
async function requestPersistentStorage(): Promise<void> {
	try {
		if (!navigator.storage?.persist || (await navigator.storage.persisted())) return;

		await navigator.storage.persist();
	} catch {
		// unsupported or denied, storage stays best-effort
	}
}

export function initStores(): Promise<void> {
	if (!browser) return Promise.resolve();

	startup ??= (async () => {
		await MigrationService.runAllMigrations();

		void requestPersistentStorage();

		const chosen = StorageModeService.chosenAdapter;
		const active = await StorageModeService.activate(() => toast.error('Failed to save settings'));

		if (active) {
			// these read user data while being constructed, before the mode was known
			modelsStore.reloadPersisted();
			conversationsStore.preferences.resetPending();
		} else if (chosen) {
			toast.warning(`${chosen.name} storage is unavailable, using this browser's storage`);
		}

		settingsStore.initialize();
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

/** Load shared by the app's pages: start the stores, then check the API key. */
export async function loadAppRoute(fetch: typeof globalThis.fetch): Promise<void> {
	// loads run before the root layout script, so the stored API key reaches
	// the probe only once the settings store has read localStorage
	await initStores();
	await validateApiKey(fetch);
}
