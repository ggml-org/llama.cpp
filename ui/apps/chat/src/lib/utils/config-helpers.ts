/**
 * Type-safe configuration helpers
 *
 * Provides utilities for safely accessing and modifying configuration objects
 * with dynamic keys while maintaining TypeScript type safety.
 */

import { SettingsPlatform } from '#lib/enums/settings.enums.ts';

/**
 * Type-safe helper to access config properties dynamically
 * Provides better type safety than direct casting to Record
 */
export function setConfigValue<T extends SettingsConfigType>(
	config: T,
	key: string,
	value: unknown
): void {
	if (key in config) {
		(config as Record<string, unknown>)[key] = value;
	}
}

/**
 * Type-safe helper to get config values dynamically
 */
export function getConfigValue<T extends SettingsConfigType>(
	config: T,
	key: string
): string | number | boolean | undefined {
	const value = (config as Record<string, unknown>)[key];

	return value as string | number | boolean | undefined;
}

/**
 * Convert a SettingsConfigType to a ParameterRecord for specific keys
 * Useful for parameter synchronization operations
 */
export function configToParameterRecord<T extends SettingsConfigType>(
	config: T,
	keys: string[]
): Record<string, string | number | boolean> {
	const record: Record<string, string | number | boolean> = {};

	for (const key of keys) {
		const value = getConfigValue(config, key);

		if (value !== undefined) {
			record[key] = value;
		}
	}

	return record;
}

/**
 * Whether a setting is listed on this device: one for a desktop or tablet layout is left off a
 * phone's settings, and one for a phone is left off the others
 */
export function isSettingShownOn(
	field: { platform?: SettingsPlatform },
	isMobile: boolean
): boolean {
	if (field.platform === SettingsPlatform.DESKTOP) return !isMobile;

	if (field.platform === SettingsPlatform.MOBILE) return isMobile;

	return true;
}
