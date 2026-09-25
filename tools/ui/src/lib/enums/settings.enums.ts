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
 * Settings field type - defines the input type for settings fields
 */
export enum SettingsFieldType {
	CHECKBOX = 'checkbox',
	INPUT = 'input',
	RADIO = 'radio',
	SELECT = 'select',
	TEXTAREA = 'textarea'
}
