/**
 * settingsStore - Application configuration and theme management
 *
 * Owns generation parameters, UI preferences and theme, persisted to
 * localStorage with Svelte 5 runes. Applies the admin's server ui_settings
 * as defaults on first visit; sampling parameters sync with the server via
 * ParameterSyncService.
 */

import { browser } from '$app/environment';
import { SETTING_CONFIG_DEFAULT, SETTINGS_KEYS } from '$lib/constants';
import { ColorMode } from '$lib/enums';
import { ParameterSyncService } from '$lib/services/parameter-sync.service';
import { SettingsService } from '$lib/services/settings.service';
import { deviceStore } from '$lib/stores/device.svelte';
// direct imports between stores, not via the barrel, to avoid circular deps
import { serverStore } from '$lib/stores/server.svelte';
import type { SettingsExportType } from '$lib/types';
import {
	configToParameterRecord,
	getConfigValue,
	normalizeFloatingPoint,
	setConfigValue
} from '$lib/utils';
import { setMode } from 'mode-watcher';

class SettingsStore {
	config = $state<SettingsConfigType>({ ...SETTING_CONFIG_DEFAULT });
	isInitialized = $state(false);
	userOverrides = $state<Set<string>>(new Set());

	// True until a config exists in localStorage; gates the one-time
	// application of server ui_settings defaults for new users.
	private isFirstVisit = false;

	canSyncParameter(key: string): boolean {
		return ParameterSyncService.canSyncParameter(key);
	}
	/**
	 * Clear all user overrides (for debugging)
	 */
	clearAllUserOverrides(): void {
		this.userOverrides.clear();
		this.saveConfig();
		console.log('Cleared all user overrides');
	}

	/**
	 * Export all settings as a versioned JSON-compatible object.
	 * The export captures the full config (excluding sensitive values like API key)
	 * and user overrides. Sensitive fields are filtered out for security by default.
	 * @param includeSensitiveData - If true, include sensitive fields (apiKey, MCP server headers) in export
	 */
	exportSettings(includeSensitiveData: boolean = false): SettingsExportType {
		// Build config excluding sensitive data unless user opts in
		const configToExport: Record<string, string | number | boolean | undefined> =
			includeSensitiveData
				? { ...this.config }
				: Object.fromEntries(Object.entries(this.config).filter(([key]) => key !== 'apiKey'));

		// Handle MCP servers: exclude custom headers unless user opts in
		if ('mcpServers' in configToExport && !includeSensitiveData) {
			try {
				const mcpServers = JSON.parse(configToExport.mcpServers as string) as Array<
					Record<string, unknown>
				>;
				const safeServers = mcpServers.map((server) => {
					delete server.headers;

					return server;
				});

				configToExport.mcpServers = JSON.stringify(safeServers);
			} catch {
				// If parsing fails, just exclude the entire mcpServers field
				delete (configToExport as Record<string, unknown>).mcpServers;
			}
		}

		return {
			config: configToExport,
			timestamp: Date.now(),
			userOverrides: Array.from(this.userOverrides),
			version: 1
		};
	}

	/**
	 * Reset all parameters to their default values (from props)
	 * This is used by the "Reset to Default" functionality
	 * Prioritizes Server defaults from /props, falls back to UI defaults
	 */
	forceSyncWithServerDefaults(): void {
		const propsDefaults = this.getServerDefaults();
		const uiSettings = serverStore.uiSettings;

		for (const key of ParameterSyncService.getSyncableParameterKeys()) {
			if (uiSettings && key in uiSettings) {
				// UI setting from admin config: write actual value
				setConfigValue(this.config, key, uiSettings[key]);
			} else if (propsDefaults[key] !== undefined) {
				// sampling param: clear it, let server decide
				setConfigValue(this.config, key, '');
			} else if (key in SETTING_CONFIG_DEFAULT) {
				setConfigValue(this.config, key, getConfigValue(SETTING_CONFIG_DEFAULT, key));
			}

			this.userOverrides.delete(key);
		}

		// Non-syncable keys: reset is a full return to the instance state, the
		// admin baseline value when defined, the factory default otherwise.
		for (const key of Object.keys(SETTING_CONFIG_DEFAULT)) {
			if (ParameterSyncService.canSyncParameter(key)) {
				continue;
			}

			const value =
				uiSettings && key in uiSettings && uiSettings[key] !== undefined
					? uiSettings[key]
					: getConfigValue(SETTING_CONFIG_DEFAULT, key);

			setConfigValue(this.config, key, value);

			if (key === SETTINGS_KEYS.THEME) {
				setMode(value as ColorMode);
			}

			this.userOverrides.delete(key);
		}

		this.saveConfig();
	}

	/**
	 * Get the entire configuration object
	 * @returns The complete configuration object
	 */
	getAllConfig(): SettingsConfigType {
		return { ...this.config };
	}

	/**
	 * Get a specific configuration value
	 * @param key - The configuration key to get
	 * @returns The configuration value
	 */
	getConfig<K extends keyof SettingsConfigType>(key: K): SettingsConfigType[K] {
		return this.config[key];
	}

	/**
	 * Get diff between current settings and server defaults
	 */
	getParameterDiff() {
		const serverDefaults = this.getServerDefaults();

		if (Object.keys(serverDefaults).length === 0) return {};

		const configAsRecord = configToParameterRecord(
			this.config,
			ParameterSyncService.getSyncableParameterKeys()
		);

		return ParameterSyncService.createParameterDiff(configAsRecord, serverDefaults);
	}

	/**
	 * Get parameter information including source for a specific parameter
	 */
	getParameterInfo(key: string) {
		const propsDefaults = this.getServerDefaults();
		const currentValue = getConfigValue(this.config, key);

		return ParameterSyncService.getParameterInfo(
			key,
			currentValue ?? '',
			propsDefaults,
			this.userOverrides
		);
	}

	/**
	 * Import settings from a previously exported object.
	 * Restores config (including theme) and user overrides.
	 * @param data - The exported settings object
	 */
	importSettings(data: SettingsExportType): void {
		if (!browser) return;

		if (!data || !data.config) {
			throw new Error('Invalid settings data: missing config');
		}

		// Restore config (theme is included in config)
		this.config = {
			...SETTING_CONFIG_DEFAULT,
			...data.config
		};

		// Restore user overrides (derived state — may be stale if server defaults differ)
		this.userOverrides = new Set(data.userOverrides ?? []);

		// Persist to localStorage
		this.saveConfig();

		// Apply theme for immediate visual feedback
		setMode(this.config[SETTINGS_KEYS.THEME] as ColorMode);

		console.log('Settings imported successfully');
	}

	/**
	 * Initialize the settings store by loading from localStorage.
	 * Called by initStores() after migrations have run.
	 */
	initialize() {
		if (!browser) return;

		try {
			this.loadConfig();
			this.migrateLegacyTheme();
			// Apply the persisted theme from config on initial load
			setMode(this.config[SETTINGS_KEYS.THEME] as ColorMode);
			this.isInitialized = true;
		} catch (error) {
			console.error('Failed to initialize settings store:', error);
		}
	}

	/**
	 * Reset all settings to defaults.
	 */
	resetAll() {
		this.resetConfig();

		this.resetTheme();
	}

	/**
	 * Reset configuration to defaults
	 */
	resetConfig() {
		this.config = { ...SETTING_CONFIG_DEFAULT };

		this.saveConfig();
	}

	/**
	 * Reset a parameter to Server default (or UI default if no Server default)
	 */
	resetParameterToServerDefault(key: string): void {
		const serverDefaults = this.getServerDefaults();
		const uiSettings = serverStore.uiSettings;

		if (uiSettings && key in uiSettings) {
			// UI setting from admin config: write actual value
			setConfigValue(this.config, key, uiSettings[key]);
		} else if (serverDefaults[key] !== undefined) {
			// sampling param known by server: clear it, let server decide
			setConfigValue(this.config, key, '');
		} else if (key in SETTING_CONFIG_DEFAULT) {
			setConfigValue(this.config, key, getConfigValue(SETTING_CONFIG_DEFAULT, key));
		}

		this.userOverrides.delete(key);
		this.saveConfig();
	}

	/**
	 * Reset theme to default value.
	 * Theme is now stored inside the config object.
	 */
	resetTheme() {
		this.updateConfig(SETTINGS_KEYS.THEME, SETTING_CONFIG_DEFAULT[SETTINGS_KEYS.THEME]);

		setMode(SETTING_CONFIG_DEFAULT[SETTINGS_KEYS.THEME] as ColorMode);
	}

	/**
	 * Initialize settings with props defaults when server properties are first loaded
	 * This sets up the default values from /props endpoint
	 */
	syncWithServerDefaults(): void {
		const propsDefaults = this.getServerDefaults();

		if (Object.keys(propsDefaults).length === 0) return;

		const uiSettings = serverStore.uiSettings;
		const uiSettingsKeys = new Set(uiSettings ? Object.keys(uiSettings) : []);

		for (const [key, propsValue] of Object.entries(propsDefaults)) {
			const currentValue = getConfigValue(this.config, key);
			const normalizedCurrent = normalizeFloatingPoint(currentValue);
			const normalizedDefault = normalizeFloatingPoint(propsValue);

			// if user value matches server, it's not a real override
			if (normalizedCurrent === normalizedDefault) {
				this.userOverrides.delete(key);

				if (!uiSettingsKeys.has(key) && getConfigValue(SETTING_CONFIG_DEFAULT, key) === undefined) {
					setConfigValue(this.config, key, undefined);
				}
			}
		}

		// UI settings are the admin's defaults for new users: applied once on
		// the first visit, never on later loads, so the user's config can
		// diverge. "Reset to Default" is the explicit way back to the baseline.
		// A first visit config carries factory values only, so a key that
		// already diverges here was set by the user before the baseline could
		// be reached, through the API key splash, and stays theirs.
		if (uiSettings && this.isFirstVisit) {
			this.isFirstVisit = false;

			for (const [key, value] of Object.entries(uiSettings)) {
				if (value === undefined || this.userOverrides.has(key)) continue;

				if (getConfigValue(this.config, key) !== getConfigValue(SETTING_CONFIG_DEFAULT, key)) {
					continue;
				}

				setConfigValue(this.config, key, value);

				// theme lives in mode-watcher, not just in config -> propagate
				if (key === SETTINGS_KEYS.THEME) {
					setMode(value as ColorMode);
				}
			}
		}

		this.saveConfig();
		console.log('User overrides after sync:', Array.from(this.userOverrides));
	}

	/**
	 * Update a specific configuration setting
	 * @param key - The configuration key to update
	 * @param value - The new value for the configuration key
	 */
	updateConfig<K extends keyof SettingsConfigType>(key: K, value: SettingsConfigType[K]): void {
		this.config[key] = value;

		if (ParameterSyncService.canSyncParameter(key as string)) {
			const propsDefaults = this.getServerDefaults();
			const propsDefault = propsDefaults[key as string];

			if (propsDefault !== undefined) {
				const normalizedValue = normalizeFloatingPoint(value);
				const normalizedDefault = normalizeFloatingPoint(propsDefault);

				if (normalizedValue === normalizedDefault) {
					this.userOverrides.delete(key as string);
				} else {
					this.userOverrides.add(key as string);
				}
			}
		}

		this.saveConfig();
	}

	/**
	 *
	 *
	 * Import / Export
	 *
	 *
	 */

	/**
	 * Update multiple configuration settings at once
	 * @param updates - Object containing the configuration updates
	 */
	updateMultipleConfig(updates: Partial<SettingsConfigType>) {
		Object.assign(this.config, updates);

		const propsDefaults = this.getServerDefaults();

		for (const [key, value] of Object.entries(updates)) {
			if (ParameterSyncService.canSyncParameter(key)) {
				const propsDefault = propsDefaults[key];

				if (propsDefault !== undefined) {
					const normalizedValue = normalizeFloatingPoint(value);
					const normalizedDefault = normalizeFloatingPoint(propsDefault);

					if (normalizedValue === normalizedDefault) {
						this.userOverrides.delete(key);
					} else {
						this.userOverrides.add(key);
					}
				}
			}
		}

		this.saveConfig();
	}

	/**
	 * Update the theme setting.
	 * @param newTheme - The new theme value
	 */
	updateTheme(newTheme: string) {
		this.updateConfig(SETTINGS_KEYS.THEME, newTheme);

		setMode(newTheme as ColorMode);
	}

	/**
	 *
	 *
	 * Utilities (private helpers)
	 *
	 *
	 */

	/**
	 * Helper method to get server defaults with null safety
	 * Centralizes the pattern of getting and extracting server defaults
	 */
	private getServerDefaults(): Record<string, string | number | boolean> {
		return ParameterSyncService.extractServerDefaults(serverStore.defaultParams);
	}

	/**
	 * Load configuration from localStorage via the persistence service.
	 * Returns default values for missing keys to prevent breaking changes.
	 */
	private loadConfig() {
		if (!browser) return;

		const {
			config: savedVal,
			isFirstVisit,
			userOverrides: savedOverrides
		} = SettingsService.loadConfig();

		// First visit: no stored config yet. Server ui_settings apply once in
		// this state, then the user's config diverges freely.
		this.isFirstVisit = isFirstVisit;

		// Merge with defaults to prevent breaking changes
		this.config = {
			...SETTING_CONFIG_DEFAULT,
			...savedVal
		};

		// Default sendOnEnter to false on mobile when the user has no saved preference
		if (!(SETTINGS_KEYS.SEND_ON_ENTER in savedVal)) {
			if (deviceStore.isMobile) {
				this.config[SETTINGS_KEYS.SEND_ON_ENTER] = false;
			}
		}

		// Load user overrides
		this.userOverrides = new Set(savedOverrides);
	}

	/**
	 * Migrate the legacy un-namespaced "theme" localStorage key into config.
	 * Previously theme was stored separately in localStorage("theme") — now it lives
	 * inside the config object alongside all other settings.
	 * After migration the legacy key is removed.
	 */
	private migrateLegacyTheme() {
		if (!browser) return;

		const legacyTheme = SettingsService.migrateLegacyTheme();

		if (legacyTheme) {
			this.config[SETTINGS_KEYS.THEME] = legacyTheme;
			this.saveConfig();
			setMode(legacyTheme as ColorMode);
		}
	}

	/**
	 * Save the current configuration to localStorage via the persistence service.
	 */
	private saveConfig() {
		if (!browser) return;

		SettingsService.saveConfig(this.config, Array.from(this.userOverrides));
	}
}

export const settingsStore = new SettingsStore();
