import { PreferencesService } from './preferences.service';
import { browser } from '$app/environment';
import {
	API_KEY_LOCALSTORAGE_KEY,
	CONFIG_LOCALSTORAGE_KEY,
	SETTINGS_KEYS,
	USER_OVERRIDES_LOCALSTORAGE_KEY
} from '$lib/constants';

/**
 * SettingsService - localStorage persistence layer for settings
 *
 * Stateless read/write of the settings config and user-override keys. Business
 * logic (default merging, mobile defaults, theme migration) stays in the store.
 *
 * **Architecture & Relationships:**
 * - **settingsStore**: Primary consumer - loads config on init and persists on change
 *
 * @see settingsStore in stores/settings/index.svelte.ts - reactive state + business logic
 */
export class SettingsService {
	/**
	 * Read the raw config and user overrides from localStorage.
	 * @returns Parsed values, or empty defaults when nothing is stored or parsing fails.
	 */
	static loadConfig(): {
		config: Record<string, unknown>;
		userOverrides: string[];
		isFirstVisit: boolean;
	} {
		if (!browser) {
			return { config: {}, isFirstVisit: false, userOverrides: [] };
		}

		try {
			const storedConfigRaw = PreferencesService.getItem(CONFIG_LOCALSTORAGE_KEY);
			const isFirstVisit = storedConfigRaw === null;
			const config = JSON.parse(storedConfigRaw || '{}') as Record<string, unknown>;

			// with the config on the server, the API key stays in this browser
			if (PreferencesService.isRemote) {
				const apiKey = PreferencesService.getItem(API_KEY_LOCALSTORAGE_KEY);

				if (apiKey) config[SETTINGS_KEYS.API_KEY] = apiKey;
			}

			const userOverrides = JSON.parse(
				PreferencesService.getItem(USER_OVERRIDES_LOCALSTORAGE_KEY) || '[]'
			) as string[];

			return { config, isFirstVisit, userOverrides };
		} catch (error) {
			console.warn('Failed to parse config from localStorage, using defaults:', error);

			return { config: {}, isFirstVisit: false, userOverrides: [] };
		}
	}

	/**
	 * Migrate the legacy un-namespaced "theme" localStorage key.
	 * Returns the legacy theme value (and removes the key) when present, else null.
	 */
	static migrateLegacyTheme(): string | null {
		if (!browser) return null;

		const legacyTheme = PreferencesService.getItem('theme');

		if (legacyTheme) {
			PreferencesService.removeItem('theme');

			return legacyTheme;
		}

		return null;
	}

	/**
	 * Persist the config and user overrides to localStorage.
	 */
	static saveConfig(config: Record<string, unknown>, userOverrides: string[]): void {
		if (!browser) return;

		try {
			if (PreferencesService.isRemote) {
				const { [SETTINGS_KEYS.API_KEY]: apiKey, ...shared } = config;

				if (typeof apiKey === 'string' && apiKey) {
					PreferencesService.setItem(API_KEY_LOCALSTORAGE_KEY, apiKey);
				} else {
					PreferencesService.removeItem(API_KEY_LOCALSTORAGE_KEY);
				}

				config = shared;
			}

			PreferencesService.setItem(CONFIG_LOCALSTORAGE_KEY, JSON.stringify(config));
			PreferencesService.setItem(USER_OVERRIDES_LOCALSTORAGE_KEY, JSON.stringify(userOverrides));
		} catch (error) {
			console.error('Failed to save config to localStorage:', error);
		}
	}
}
