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
import { SETTING_CONFIG_DEFAULT } from '$lib/constants/settings-config';
import { ParameterSyncService } from '$lib/services/parameter-sync.service';
import { serverStore } from '$lib/stores/server.svelte';
import {
	configToParameterRecord,
	normalizeFloatingPoint,
	getConfigValue,
	setConfigValue
} from '$lib/utils';
import {
	CONFIG_LOCALSTORAGE_KEY,
	USER_OVERRIDES_LOCALSTORAGE_KEY
} from '$lib/constants/localstorage-keys';

class SettingsStore {
	/**
	 *
	 *
	 * State
	 *
	 *
	 */

	config = $state<SettingsConfigType>({ ...SETTING_CONFIG_DEFAULT });
	theme = $state<string>('auto');
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
		const webuiSettings = serverStore.webuiSettings;
		return ParameterSyncService.extractServerDefaults(serverParams, webuiSettings);
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
			this.loadTheme();
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
	 * Load theme from localStorage
	 */
	private loadTheme() {
		if (!browser) return;

		this.theme = localStorage.getItem('theme') || 'auto';
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
	 * Update the theme setting
	 * @param newTheme - The new theme value
	 */
	updateTheme(newTheme: string) {
		this.theme = newTheme;
		this.saveTheme();
	}

	/**
	 * Save the current theme to localStorage
	 */
	private saveTheme() {
		if (!browser) return;

		try {
			if (this.theme === 'auto') {
				localStorage.removeItem('theme');
			} else {
				localStorage.setItem('theme', this.theme);
			}
		} catch (error) {
			console.error('Failed to save theme to localStorage:', error);
		}
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
	 * Reset theme to auto
	 */
	resetTheme() {
		this.theme = 'auto';
		this.saveTheme();
	}

	/**
	 * Reset all settings to defaults
	 */
	resetAll() {
		this.resetConfig();
		this.resetTheme();
	}

	/**
	 * Reset a parameter to server default (or webui default if no server default)
	 */
	resetParameterToServerDefault(key: string): void {
		const serverDefaults = this.getServerDefaults();

		if (serverDefaults[key] !== undefined) {
			const value = normalizeFloatingPoint(serverDefaults[key]);

			this.config[key as keyof SettingsConfigType] =
				value as SettingsConfigType[keyof SettingsConfigType];
		} else {
			if (key in SETTING_CONFIG_DEFAULT) {
				const defaultValue = getConfigValue(SETTING_CONFIG_DEFAULT, key);

				setConfigValue(this.config, key, defaultValue);
			}
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

		if (Object.keys(propsDefaults).length === 0) {
			console.warn('No server defaults available for initialization');

			return;
		}

		for (const [key, propsValue] of Object.entries(propsDefaults)) {
			const currentValue = getConfigValue(this.config, key);

			const normalizedCurrent = normalizeFloatingPoint(currentValue);
			const normalizedDefault = normalizeFloatingPoint(propsValue);

			if (normalizedCurrent === normalizedDefault) {
				this.userOverrides.delete(key);
				setConfigValue(this.config, key, propsValue);
			} else if (!this.userOverrides.has(key)) {
				setConfigValue(this.config, key, propsValue);
			}
		}

		this.saveConfig();
		console.log('Settings initialized with props defaults:', propsDefaults);
		console.log('Current user overrides after sync:', Array.from(this.userOverrides));
	}

	/**
	 * Reset all parameters to their default values (from props)
	 * This is used by the "Reset to Default" functionality
	 * Prioritizes server defaults from /props, falls back to webui defaults
	 */
	forceSyncWithServerDefaults(): void {
		const propsDefaults = this.getServerDefaults();
		const syncableKeys = ParameterSyncService.getSyncableParameterKeys();

		for (const key of syncableKeys) {
			if (propsDefaults[key] !== undefined) {
				const normalizedValue = normalizeFloatingPoint(propsDefaults[key]);

				setConfigValue(this.config, key, normalizedValue);
			} else {
				if (key in SETTING_CONFIG_DEFAULT) {
					const defaultValue = getConfigValue(SETTING_CONFIG_DEFAULT, key);

					setConfigValue(this.config, key, defaultValue);
				}
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
}

export const settingsStore = new SettingsStore();

export const config = () => settingsStore.config;
export const theme = () => settingsStore.theme;
export const isInitialized = () => settingsStore.isInitialized;
