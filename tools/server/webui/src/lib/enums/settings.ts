/**
 * Parameter source - indicates whether a parameter uses default or custom value
 */
export enum ParameterSource {
	DEFAULT = 'default',
	CUSTOM = 'custom'
}

/**
 * Syncable parameter type - data types for parameters that can be synced with server
 */
export enum SyncableParameterType {
	NUMBER = 'number',
	STRING = 'string',
	BOOLEAN = 'boolean'
}

/**
 * Settings field type - defines the input type for settings fields
 */
export enum SettingsFieldType {
	INPUT = 'input',
	TEXTAREA = 'textarea',
	CHECKBOX = 'checkbox',
	SELECT = 'select'
}
