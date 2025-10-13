/**
 * SettingsStore - Application configuration and theme management
 *
 * This store manages all application settings including AI model parameters, UI preferences,
 * and theme configuration. It provides persistent storage through localStorage with reactive
 * state management using Svelte 5 runes.
 *
 * **Architecture & Relationships:**
 * - **SettingsStore** (this class): Configuration state management
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
import { normalizeFloatingPoint } from '$lib/utils/precision';
import { ParameterSyncService } from '$lib/services/parameter-sync';
import { serverStore } from '$lib/stores/server.svelte';
import { setConfigValue, getConfigValue, configToParameterRecord } from '$lib/utils/config-helpers';

class SettingsStore {
	config = $state<SettingsConfigType>({ ...SETTING_CONFIG_DEFAULT });
	theme = $state<string>('auto');
	isInitialized = $state(false);
	userOverrides = $state<Set<string>>(new Set());

	/**
	 * Helper method to get server defaults with null safety
	 * Centralizes the pattern of getting and extracting server defaults
	 */
	private getServerDefaults(): Record<string, string | number | boolean> {
		const serverParams = serverStore.serverDefaultParams;
		return serverParams ? ParameterSyncService.extractServerDefaults(serverParams) : {};
	}

	constructor() {
		if (browser) {
			this.initialize();
		}
	}

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
			const savedVal = JSON.parse(localStorage.getItem('config') || '{}');

			// Merge with defaults to prevent breaking changes
			this.config = {
				...SETTING_CONFIG_DEFAULT,
				...savedVal
			};

			// Load user overrides
			const savedOverrides = JSON.parse(localStorage.getItem('userOverrides') || '[]');
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
			localStorage.setItem('config', JSON.stringify(this.config));

			localStorage.setItem('userOverrides', JSON.stringify(Array.from(this.userOverrides)));
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

	/**
	 * Initialize settings with props defaults when server properties are first loaded
	 * This sets up the default values from /props endpoint
	 */
	syncWithServerDefaults(): void {
		const serverParams = serverStore.serverDefaultParams;
		if (!serverParams) {
			console.warn('No server parameters available for initialization');

			return;
		}

		const propsDefaults = this.getServerDefaults();

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
	 * Clear all user overrides (for debugging)
	 */
	clearAllUserOverrides(): void {
		this.userOverrides.clear();
		this.saveConfig();
		console.log('Cleared all user overrides');
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
}

// Create and export the settings store instance
export const settingsStore = new SettingsStore();

// Export reactive getters for easy access in components
export const config = () => settingsStore.config;
export const theme = () => settingsStore.theme;
export const isInitialized = () => settingsStore.isInitialized;

// Export bound methods for easy access
export const updateConfig = settingsStore.updateConfig.bind(settingsStore);
export const updateMultipleConfig = settingsStore.updateMultipleConfig.bind(settingsStore);
export const updateTheme = settingsStore.updateTheme.bind(settingsStore);
export const resetConfig = settingsStore.resetConfig.bind(settingsStore);
export const resetTheme = settingsStore.resetTheme.bind(settingsStore);
export const resetAll = settingsStore.resetAll.bind(settingsStore);
export const getConfig = settingsStore.getConfig.bind(settingsStore);
export const getAllConfig = settingsStore.getAllConfig.bind(settingsStore);
export const syncWithServerDefaults = settingsStore.syncWithServerDefaults.bind(settingsStore);
export const forceSyncWithServerDefaults =
	settingsStore.forceSyncWithServerDefaults.bind(settingsStore);
export const getParameterInfo = settingsStore.getParameterInfo.bind(settingsStore);
export const resetParameterToServerDefault =
	settingsStore.resetParameterToServerDefault.bind(settingsStore);
export const getParameterDiff = settingsStore.getParameterDiff.bind(settingsStore);
export const clearAllUserOverrides = settingsStore.clearAllUserOverrides.bind(settingsStore);
