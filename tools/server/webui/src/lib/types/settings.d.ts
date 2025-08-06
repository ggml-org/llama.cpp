import type { SETTING_CONFIG_DEFAULT } from "$lib/constants/settings-config";

export type ConfigValue = string | number | boolean;

export interface FieldConfig {
	key: string;
	label: string;
	type: 'input' | 'textarea' | 'checkbox';
	help?: string;
}

export type SettingsConfigType = typeof SETTING_CONFIG_DEFAULT & {
	[key: string]: string | number | boolean;
};
