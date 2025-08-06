// Settings management using localStorage, compatible with webui-old structure

import { browser } from "$app/environment";

export const CONFIG_DEFAULT: Record<string, string | number | boolean> = {
	// Note: in order not to introduce breaking changes, please keep the same data type (number, string, etc) if you want to change the default value. Do not use null or undefined for default value.
	// Do not use nested objects, keep it single level. Prefix the key if you need to group them.
	apiKey: '',
	systemMessage: '',
	showTokensPerSecond: false,
	showThoughtInProgress: true,
	excludeThoughtOnReq: false,
	pasteLongTextToFileLen: 2000,
	pdfAsImage: false,
	// make sure these default values are in sync with `common.h`
	samplers: 'top_k;tfs_z;typical_p;top_p;min_p;temperature',
	temperature: 0.8,
	dynatemp_range: 0.0,
	dynatemp_exponent: 1.0,
	top_k: 40,
	top_p: 0.95,
	min_p: 0.05,
	xtc_probability: 0.0,
	xtc_threshold: 0.1,
	typical_p: 1.0,
	repeat_last_n: 64,
	repeat_penalty: 1.0,
	presence_penalty: 0.0,
	frequency_penalty: 0.0,
	dry_multiplier: 0.0,
	dry_base: 1.75,
	dry_allowed_length: 2,
	dry_penalty_last_n: -1,
	max_tokens: 2048,
	custom: '', // custom json-stringified object
	// experimental features
	pyInterpreterEnabled: false
};

export type ConfigType = typeof CONFIG_DEFAULT & {
	[key: string]: string | number | boolean;
};

/**
 * Get configuration from localStorage
 * Returns default values for missing keys to prevent breaking changes
 */
export function getConfig(): ConfigType {
	if (!browser) return { ...CONFIG_DEFAULT };
	try {
		const savedVal = JSON.parse(localStorage.getItem('config') || '{}');
		// to prevent breaking changes in the future, we always provide default value for missing keys
		return {
			...CONFIG_DEFAULT,
			...savedVal,
		};
	} catch (error) {
		console.warn('Failed to parse config from localStorage, using defaults:', error);
		return { ...CONFIG_DEFAULT };
	}
}

/**
 * Save configuration to localStorage
 */
export function setConfig(config: ConfigType): void {
	try {
		localStorage.setItem('config', JSON.stringify(config));
	} catch (error) {
		console.error('Failed to save config to localStorage:', error);
	}
}

/**
 * Get theme from localStorage
 */
export function getTheme(): string {
	return localStorage.getItem('theme') || 'auto';
}

/**
 * Set theme in localStorage
 */
export function setTheme(theme: string): void {
	if (theme === 'auto') {
		localStorage.removeItem('theme');
	} else {
		localStorage.setItem('theme', theme);
	}
}

/**
 * Reset configuration to defaults
 */
export function resetConfig(): ConfigType {
	const defaultConfig = { ...CONFIG_DEFAULT };
	setConfig(defaultConfig);
	return defaultConfig;
}
