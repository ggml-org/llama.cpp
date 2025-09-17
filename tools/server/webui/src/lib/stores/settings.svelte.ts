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

class SettingsStore {
	config = $state<SettingsConfigType>({ ...SETTING_CONFIG_DEFAULT });
	theme = $state<string>('auto');
	isInitialized = $state(false);

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
		} catch (error) {
			console.warn('Failed to parse config from localStorage, using defaults:', error);
			this.config = { ...SETTING_CONFIG_DEFAULT };
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
	updateConfig<K extends keyof SettingsConfigType>(key: K, value: SettingsConfigType[K]) {
		this.config[key] = value;
		this.saveConfig();
	}

	/**
	 * Update multiple configuration settings at once
	 * @param updates - Object containing the configuration updates
	 */
	updateMultipleConfig(updates: Partial<SettingsConfigType>) {
		Object.assign(this.config, updates);
		this.saveConfig();
	}

	/**
	 * Save the current configuration to localStorage
	 */
	private saveConfig() {
		if (!browser) return;

		try {
			localStorage.setItem('config', JSON.stringify(this.config));
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
