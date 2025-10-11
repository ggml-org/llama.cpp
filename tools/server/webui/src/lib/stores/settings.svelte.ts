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
import { ParameterSyncService, type ParameterRecord } from '$lib/services/parameter-sync';
import { getServerDefaultParams } from '$lib/services/settings-sync';

class SettingsStore {
	config = $state<SettingsConfigType>({ ...SETTING_CONFIG_DEFAULT });
	theme = $state<string>('auto');
	isInitialized = $state(false);
	userOverrides = $state<Set<string>>(new Set());

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

		// Only mark as user override if this is a syncable parameter AND differs from props default
		if (ParameterSyncService.canSyncParameter(key as string)) {
			const serverParams = getServerDefaultParams();
			const propsDefaults = serverParams
				? ParameterSyncService.extractServerDefaults(serverParams)
				: {};

			const propsDefault = propsDefaults[key as string];

			// Compare with props default - apply rounding for numbers to handle precision issues
			if (propsDefault !== undefined) {
				const normalizedValue =
					typeof value === 'number' ? Math.round(value * 1000000) / 1000000 : value;
				const normalizedDefault =
					typeof propsDefault === 'number'
						? Math.round(propsDefault * 1000000) / 1000000
						: propsDefault;

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

		// Get props defaults once for efficiency
		const serverParams = getServerDefaultParams();
		const propsDefaults = serverParams
			? ParameterSyncService.extractServerDefaults(serverParams)
			: {};

		// Check each updated parameter against props defaults
		for (const [key, value] of Object.entries(updates)) {
			if (ParameterSyncService.canSyncParameter(key)) {
				const propsDefault = propsDefaults[key];

				// Compare with props default - apply rounding for numbers to handle precision issues
				if (propsDefault !== undefined) {
					const normalizedValue =
						typeof value === 'number' ? Math.round(value * 1000000) / 1000000 : value;
					const normalizedDefault =
						typeof propsDefault === 'number'
							? Math.round(propsDefault * 1000000) / 1000000
							: propsDefault;

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
			// Also save user overrides to track which parameters are custom
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
		const serverParams = getServerDefaultParams();
		if (!serverParams) {
			console.warn('No server parameters available for initialization');
			return;
		}

		const propsDefaults = ParameterSyncService.extractServerDefaults(serverParams);

		// Clean up user overrides by comparing current values with props defaults
		// This fixes cases where localStorage has stale override flags
		for (const [key, propsValue] of Object.entries(propsDefaults)) {
			const currentValue = (this.config as ParameterRecord)[key];

			// Apply same rounding logic for comparison
			const normalizedCurrent =
				typeof currentValue === 'number'
					? Math.round(currentValue * 1000000) / 1000000
					: currentValue;
			const normalizedDefault =
				typeof propsValue === 'number' ? Math.round(propsValue * 1000000) / 1000000 : propsValue;

			if (normalizedCurrent === normalizedDefault) {
				// Values match - remove from overrides and ensure we use props value
				this.userOverrides.delete(key);
				(this.config as ParameterRecord)[key] = propsValue;
			} else if (!this.userOverrides.has(key)) {
				// Values differ but not marked as override - use props default
				(this.config as ParameterRecord)[key] = propsValue;
			}
			// If values differ and it's marked as override, keep the current value
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
	 */
	forceSyncWithServerDefaults(): void {
		const serverParams = getServerDefaultParams();
		if (!serverParams) {
			console.warn('No server parameters available for reset');
			return;
		}

		const propsDefaults = ParameterSyncService.extractServerDefaults(serverParams);

		// Reset all syncable parameters and clear user overrides
		for (const [key, propsValue] of Object.entries(propsDefaults)) {
			(this.config as ParameterRecord)[key] = propsValue;
			this.userOverrides.delete(key); // Clear user override since we're resetting
		}

		this.saveConfig();
		console.log('Reset all settings to props defaults:', propsDefaults);
	}

	/**
	 * Get parameter information including source for a specific parameter
	 */
	getParameterInfo(key: string) {
		const serverParams = getServerDefaultParams();
		const propsDefaults = serverParams
			? ParameterSyncService.extractServerDefaults(serverParams)
			: {};

		// Get the current value for this specific parameter
		const currentValue = (this.config as ParameterRecord)[key];

		return ParameterSyncService.getParameterInfo(
			key,
			currentValue,
			propsDefaults,
			this.userOverrides
		);
	}

	/**
	 * Reset a parameter to server default (or webui default if no server default)
	 */
	resetParameterToServerDefault(key: string): void {
		const serverParams = getServerDefaultParams();
		const serverDefaults = serverParams
			? ParameterSyncService.extractServerDefaults(serverParams)
			: {};

		if (serverDefaults[key] !== undefined) {
			// Apply the same rounding to ensure consistency
			const value =
				typeof serverDefaults[key] === 'number'
					? Math.round((serverDefaults[key] as number) * 1000000) / 1000000
					: serverDefaults[key];
			this.config[key as keyof SettingsConfigType] =
				value as SettingsConfigType[keyof SettingsConfigType];
		} else {
			// Fallback to webui default
			if (key in SETTING_CONFIG_DEFAULT) {
				(this.config as ParameterRecord)[key] = (SETTING_CONFIG_DEFAULT as ParameterRecord)[key];
			}
		}

		// Remove from user overrides
		this.userOverrides.delete(key);
		this.saveConfig();
	}

	/**
	 * Get diff between current settings and server defaults
	 */
	getParameterDiff() {
		const serverParams = getServerDefaultParams();
		if (!serverParams) return {};

		const serverDefaults = ParameterSyncService.extractServerDefaults(serverParams);
		return ParameterSyncService.createParameterDiff(this.config as ParameterRecord, serverDefaults);
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
