/**
 * Parameter source - indicates whether a parameter uses default or custom value
 */
export enum ParameterSource {
	CUSTOM = 'custom',
	DEFAULT = 'default'
}

/**
 * Syncable parameter type - data types for parameters that can be synced with server
 */
export enum SyncableParameterType {
	BOOLEAN = 'boolean',
	NUMBER = 'number',
	STRING = 'string'
}

/**
 * Where a setting applies: every device, or only a desktop or tablet layout, or only a phone's.
 * A setting is listed only where it applies.
 */
export enum SettingsPlatform {
	ALL = 'all',
	DESKTOP = 'desktop',
	MOBILE = 'mobile'
}

/**
 * Settings field type - defines the input type for settings fields
 */
export enum SettingsFieldType {
	CHECKBOX = 'checkbox',
	INPUT = 'input',
	RADIO = 'radio',
	SELECT = 'select',
	TEXTAREA = 'textarea'
}

/**
 * How a phone's settings list lays a field out: as a row of a group it shares with its
 * neighbours, or as a group of its own (a prompt, a choice of one).
 */
export enum SettingsFieldBlock {
	OWN = 'own',
	ROWS = 'rows'
}
