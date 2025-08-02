export type ConfigValue = string | number | boolean;

export interface FieldConfig {
	key: string;
	label: string;
	type: 'input' | 'textarea' | 'checkbox';
	help?: string;
}
