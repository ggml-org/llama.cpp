/**
 * settingsStore - Application configuration and theme management
 *
 * This store manages all application settings including AI model parameters, UI preferences,
 * and theme configuration. It provides persistent storage through localStorage with reactive
 * state management using Svelte 5 runes.
 *
 * **Architecture & Relationships:**
 * - **settingsStore** (this class): Configuration state management
 *   - Manages AI model parameters (temperature, max tokens, etc.)
 *   - Handles theme switching and persistence
 *   - Provides localStorage synchronization
 *   - Offers reactive configuration access
 *
 * - **ChatService**: Reads model parameters for API requests
 * - **UI Components**: Subscribe to theme and configuration changes
 *
 * **Key Features:**
 * - **Model Parameters**: Temperature, max tokens, top-p, top-k, repeat penalty
 * - **Theme Management**: Auto, light, dark theme switching
 * - **Persistence**: Automatic localStorage synchronization
 * - **Reactive State**: Svelte 5 runes for automatic UI updates
 * - **Default Handling**: Graceful fallback to defaults for missing settings
 * - **Batch Updates**: Efficient multi-setting updates
 * - **Reset Functionality**: Restore defaults for individual or all settings
 *
 * **Configuration Categories:**
 * - Generation parameters (temperature, tokens, sampling)
 * - UI preferences (theme, display options)
 * - System settings (model selection, prompts)
 * - Advanced options (seed, penalties, context handling)
 */

import { browser } from '$app/environment';
import { ColorMode } from '$lib/enums';
import type { SettingsExportType } from '$lib/types';
import { setMode } from 'mode-watcher';
import {
	CONFIG_LOCALSTORAGE_KEY,
	SETTING_CONFIG_DEFAULT,
	SETTINGS_KEYS,
	USER_OVERRIDES_LOCALSTORAGE_KEY
} from '$lib/constants';

import { IsMobile } from '$lib/hooks/is-mobile.svelte';
import { ParameterSyncService } from '$lib/services/parameter-sync.service';
import { serverStore } from '$lib/stores/server.svelte';
import {
	configToParameterRecord,
	normalizeFloatingPoint,
	getConfigValue,
	setConfigValue
} from '$lib/utils';

class SettingsStore {
	/**
	 *
	 *
	 * State
	 *
	 *
	 */

	config = $state<SettingsConfigType>({ ...SETTING_CONFIG_DEFAULT });
	isInitialized = $state(false);
	userOverrides = $state<Set<string>>(new Set());

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
		const serverParams = serverStore.defaultParams;
		const uiSettings = serverStore.uiSettings;

		return ParameterSyncService.extractServerDefaults(serverParams, uiSettings);
	}

	constructor() {
		if (browser) {
			this.initialize();
		}
	}

	/**
	 *
	 *
	 * Lifecycle
	 *
	 *
	 */

	/**
	 * Initialize the settings store by loading from localStorage
	 */
	initialize() {
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
	 * Load configuration from localStorage
	 * Returns default values for missing keys to prevent breaking changes
	 */
	private loadConfig() {
		if (!browser) return;

		try {
			const storedConfigRaw = localStorage.getItem(CONFIG_LOCALSTORAGE_KEY);
			const savedVal = JSON.parse(storedConfigRaw || '{}');

			// Merge with defaults to prevent breaking changes
			this.config = {
				...SETTING_CONFIG_DEFAULT,
				...savedVal
			};

			// Default sendOnEnter to false on mobile when the user has no saved preference
			if (!(SETTINGS_KEYS.SEND_ON_ENTER in savedVal)) {
				if (new IsMobile().current) {
					this.config[SETTINGS_KEYS.SEND_ON_ENTER] = false;
				}
			}

			// Load user overrides
			const savedOverrides = JSON.parse(
				localStorage.getItem(USER_OVERRIDES_LOCALSTORAGE_KEY) || '[]'
			);
			this.userOverrides = new Set(savedOverrides);
		} catch (error) {
			console.warn('Failed to parse config from localStorage, using defaults:', error);
			this.config = { ...SETTING_CONFIG_DEFAULT };
			this.userOverrides = new Set();
		}
	}

	/**
	 * Migrate the legacy un-namespaced "theme" localStorage key into config.
	 * Previously theme was stored separately in localStorage("theme") — now it lives
	 * inside the config object alongside all other settings.
	 * After migration the legacy key is removed.
	 */
	private migrateLegacyTheme() {
		if (!browser) return;

		const legacyTheme = localStorage.getItem('theme');
		if (legacyTheme) {
			this.config[SETTINGS_KEYS.THEME] = legacyTheme;
			localStorage.removeItem('theme');
			this.saveConfig();
			setMode(legacyTheme as ColorMode);
		}
	}
	/**
	 *
	 *
	 * Config Updates
	 *
	 *
	 */

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
	 * Save the current configuration to localStorage
	 */
	private saveConfig() {
		if (!browser) return;

		try {
			localStorage.setItem(CONFIG_LOCALSTORAGE_KEY, JSON.stringify(this.config));

			localStorage.setItem(
				USER_OVERRIDES_LOCALSTORAGE_KEY,
				JSON.stringify(Array.from(this.userOverrides))
			);
		} catch (error) {
			console.error('Failed to save config to localStorage:', error);
		}
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
	 * Reset
	 *
	 *
	 */

	/**
	 * Reset configuration to defaults
	 */
	resetConfig() {
		this.config = { ...SETTING_CONFIG_DEFAULT };

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
	 * Reset all settings to defaults.
	 */
	resetAll() {
		this.resetConfig();

		this.resetTheme();
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
	 *
	 *
	 * Server Sync
	 *
	 *
	 */

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

		// UI settings need actual values in config (no placeholder mechanism),
		// so write them for non-overridden keys
		if (uiSettings) {
			for (const [key, value] of Object.entries(uiSettings)) {
				if (!this.userOverrides.has(key) && value !== undefined) {
					setConfigValue(this.config, key, value);

					// theme lives in mode-watcher, not just in config -> propagate
					if (key === SETTINGS_KEYS.THEME) {
						setMode(value as ColorMode);
					}
				}
			}
		}

		this.saveConfig();
		console.log('User overrides after sync:', Array.from(this.userOverrides));
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

		this.saveConfig();
	}

	/**
	 *
	 *
	 * Utilities
	 *
	 *
	 */

	/**
	 * Get a specific configuration value
	 * @param key - The configuration key to get
	 * @returns The configuration value
	 */
	getConfig<K extends keyof SettingsConfigType>(key: K): SettingsConfigType[K] {
		return this.config[key];
	}

	/**
	 * Get the entire configuration object
	 * @returns The complete configuration object
	 */
	getAllConfig(): SettingsConfigType {
		return { ...this.config };
	}

	canSyncParameter(key: string): boolean {
		return ParameterSyncService.canSyncParameter(key);
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
	 * Clear all user overrides (for debugging)
	 */
	clearAllUserOverrides(): void {
		this.userOverrides.clear();
		this.saveConfig();
		console.log('Cleared all user overrides');
	}

	/**
	 *
	 *
	 * Import / Export
	 *
	 *
	 */

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
			version: 1,
			timestamp: Date.now(),
			config: configToExport,
			userOverrides: Array.from(this.userOverrides)
		};
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
}

export const settingsStore = new SettingsStore();

export const config = () => settingsStore.config;
export const theme = () => settingsStore.config[SETTINGS_KEYS.THEME];
export const isInitialized = () => settingsStore.isInitialized;
