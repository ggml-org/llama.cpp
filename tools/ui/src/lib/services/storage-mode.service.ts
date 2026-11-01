import { DatabaseService } from './database.service';
import { DocumentStoreBackend } from './document-store-backend';
import { PreferencesService } from './preferences.service';
import {
	API_KEY_LOCALSTORAGE_KEY,
	CONFIG_LOCALSTORAGE_KEY,
	SETTINGS_KEYS,
	STORAGE_ADAPTER_LOCALSTORAGE_KEY,
	USER_PREFERENCE_KEYS
} from '$lib/constants';
import { findStorageAdapter, StorageConflictError } from '$lib/storage-adapters';
import type { StorageAdapterDefinition } from '$lib/types';

/**
 * StorageModeService - chooses where conversations and user settings live
 *
 * This browser (IndexedDB + localStorage) by default, or one of the storage
 * adapters in $lib/storage-adapters. The choice is kept per browser. When the
 * chosen adapter cannot be reached at startup the app falls back to this
 * browser for that session and keeps the choice for the next one.
 */
export class StorageModeService {
	private static active: StorageAdapterDefinition | null = null;

	/** Adapter this session runs with, null for this browser, which can differ from the chosen one after a fallback. */
	static get activeAdapter(): StorageAdapterDefinition | null {
		return this.active;
	}

	/** Adapter chosen in this browser, null for this browser. */
	static get chosenAdapter(): StorageAdapterDefinition | null {
		try {
			return (
				findStorageAdapter(PreferencesService.getItem(STORAGE_ADAPTER_LOCALSTORAGE_KEY)) ?? null
			);
		} catch {
			return null;
		}
	}

	/**
	 * Connect to the chosen storage adapter, if any. Runs once at startup,
	 * before any store reads conversations or settings.
	 *
	 * @param onPushError - Called when a settings change could not be saved
	 * @returns The adapter the session runs with, null for this browser
	 */
	static async activate(
		onPushError: (error: unknown) => void
	): Promise<StorageAdapterDefinition | null> {
		const definition = this.chosenAdapter;

		if (!definition) return this.active;

		try {
			const adapter = definition.create();
			const settings = await adapter.getSettings();

			PreferencesService.useRemote(settings, async (set, remove) => {
				try {
					await adapter.updateSettings(set, remove);
				} catch (error) {
					onPushError(error);
				}
			});
			DatabaseService.useBackend(new DocumentStoreBackend(adapter));
			this.active = definition;
		} catch (error) {
			console.error(`${definition.name} storage is unavailable, using browser storage:`, error);
		}

		return this.active;
	}

	/**
	 * Copy this browser's conversations and settings through `definition`,
	 * then use it from the next load on. Conversations and settings the store
	 * already holds are kept as they are.
	 *
	 * @returns How many conversations were copied and how many the store already had
	 */
	static async switchTo(
		definition: StorageAdapterDefinition
	): Promise<{ copied: number; skipped: number }> {
		const adapter = definition.create();
		const conversations = await DatabaseService.getAllConversations();
		const documents = await DatabaseService.getConversationsWithMessages(
			conversations.map((conv) => conv.id)
		);

		let copied = 0;
		let skipped = 0;

		for (const { conv, messages } of documents.values()) {
			try {
				await adapter.putConversation({ conv, messages, rev: 0 });
				copied++;
			} catch (error) {
				if (!(error instanceof StorageConflictError)) throw error;

				skipped++;
			}
		}

		const remote = await adapter.getSettings();
		const localConfigRaw = PreferencesService.getItem(CONFIG_LOCALSTORAGE_KEY);
		const localConfig = JSON.parse(localConfigRaw || '{}') as Record<string, unknown>;
		// the API key stays in this browser, see SettingsService
		const { [SETTINGS_KEYS.API_KEY]: apiKey, ...sharedConfig } = localConfig;
		const set: Record<string, string> = {};

		for (const key of USER_PREFERENCE_KEYS) {
			const value =
				key === CONFIG_LOCALSTORAGE_KEY
					? localConfigRaw && JSON.stringify(sharedConfig)
					: PreferencesService.getItem(key);

			if (value === null || key in remote) continue;

			set[key] = value;
		}

		await adapter.updateSettings(set, []);

		if (typeof apiKey === 'string' && apiKey) {
			PreferencesService.setItem(API_KEY_LOCALSTORAGE_KEY, apiKey);
		}

		PreferencesService.setItem(STORAGE_ADAPTER_LOCALSTORAGE_KEY, definition.id);

		return { copied, skipped };
	}

	/**
	 * Use this browser's storage from the next load on. Data in the adapter stays there.
	 */
	static switchToBrowser(): void {
		PreferencesService.removeItem(STORAGE_ADAPTER_LOCALSTORAGE_KEY);
	}
}
